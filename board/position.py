"""A position as one array of 64-bit words, with moves made and unmade in place."""

from __future__ import annotations

import re

import numpy as np
from numpy.typing import NDArray

from ..board.compiled import compiled, compiled_inline, signed

# Squares count from a1 = 0 along the ranks to h8 = 63, as in python-chess.
type Square = int
# A move packed as from | to << 6 | promotion << 12, the form compiled code passes.
type PackedMove = int
# Words 0 to 14: the bitboard of each piece, by its code from `piece`. Then the
# words at the offsets below. The undo stack holds a record of UNDO_SIZE words per ply.
type Position = NDArray[np.uint64]

WHITE, BLACK = 0, 1
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(1, 7)

OCCUPANCY = 15  # + color
_MAILBOX = 17  # + square: the piece code, or 0
TURN = 81
CASTLING = 82  # one flag per entry of CASTLING_RIGHTS
EN_PASSANT = 83  # the square a pawn skipped on the last move, or NO_SQUARE
PLY = 84
UNDO = 85
UNDO_SIZE = 4  # the move, the captured piece, castling, en passant
MAX_PLY = 1024
NO_SQUARE = 64

STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Each right's FEN letter, its king's start and target, and its rook's start.
CASTLING_RIGHTS = (
    ("K", "e1", "g1", "h1"),
    ("Q", "e1", "c1", "a1"),
    ("k", "e8", "g8", "h8"),
    ("q", "e8", "c8", "a8"),
)

SQUARE_BITS = np.array([1 << square for square in range(64)], np.uint64)


def parse_fen(fen: str) -> Position:
    """The position of a FEN with all six fields, ignoring the move counters.

    Move generation also needs the side not to move out of check, which only
    `board.from_fen` can tell.

    Raises:
        ValueError: If `fen` is malformed, a side lacks its one king or has more than
            16 pieces, a pawn stands on the first or last rank, a castling right lacks
            its king or rook on their start squares, or the en passant square has no
            pawn that just stepped past it.
    """
    if not (match := _FEN.fullmatch(" ".join(fen.split()))):
        message = f"not a FEN: {fen!r}"
        raise ValueError(message)
    placement, turn, castling, en_passant = match.groups()
    pieces: list[int] = []
    for row in reversed(placement.split("/")):
        rank: list[int] = []
        for char in row:
            if char.isdigit():
                rank.extend([0] * int(char))
            else:
                kind = _PIECE_LETTERS.index(char.lower()) + 1
                rank.append(piece(WHITE if char.isupper() else BLACK, kind))
        if len(rank) != 8:
            message = f"rank {row!r} doesn't have 8 squares"
            raise ValueError(message)
        pieces.extend(rank)

    # Compiled code runs unchecked and relies on these, which also bound the number of
    # legal moves by MAX_MOVES. It also relies on the side not to move not being in
    # check, which `board.from_fen` checks, since telling needs move generation.
    for color, name in ((WHITE, "white"), (BLACK, "black")):
        own = [code for code in pieces if code and code >> 3 == color]
        if own.count(piece(color, KING)) != 1 or len(own) > 16:
            message = f"{name} needs one king and at most 16 pieces"
            raise ValueError(message)
    if PAWN in {code & 7 for code in pieces[:8] + pieces[56:]}:
        message = "a pawn stands on the first or last rank"
        raise ValueError(message)
    mover = WHITE if turn == "w" else BLACK
    if en_passant != "-":
        square = parse_square(en_passant)
        victim, origin = square - forward(mover), square + forward(mover)
        if (
            square // 8 != 5 - 3 * mover
            or pieces[victim] != piece(mover ^ 1, PAWN)
            or pieces[square]
            or pieces[origin]
        ):
            message = f"no pawn just stepped past the en passant square {en_passant}"
            raise ValueError(message)

    position = np.zeros(UNDO + UNDO_SIZE * MAX_PLY, np.uint64)
    for square, code in enumerate(pieces):
        if code:
            position[_MAILBOX + square] = code
            position[code] |= SQUARE_BITS[square]
            position[OCCUPANCY + (code >> 3)] |= SQUARE_BITS[square]
    for flag, (letter, king, _, rook) in enumerate(CASTLING_RIGHTS):
        if letter in castling:
            color = WHITE if letter.isupper() else BLACK
            if (pieces[parse_square(king)], pieces[parse_square(rook)]) != (
                piece(color, KING),
                piece(color, ROOK),
            ):
                message = f"castling right {letter} lacks its king or rook"
                raise ValueError(message)
            position[CASTLING] |= np.uint64(1 << flag)
    position[TURN] = mover
    position[EN_PASSANT] = NO_SQUARE if en_passant == "-" else parse_square(en_passant)
    return position


def parse_square(name: str) -> Square:
    """The square of a name like `e4`."""
    return "abcdefgh".index(name[0]) + 8 * "12345678".index(name[1])


def format_packed_uci(move: PackedMove) -> str:
    """The UCI notation of `move`, like `e2e4` or `e7e8q`."""
    promotion = move >> 12
    return (
        _format_square(move & 63)
        + _format_square(move >> 6 & 63)
        + (_PIECE_LETTERS[promotion - 1] if promotion else "")
    )


@compiled
def piece(color: int, kind: int) -> int:
    """The code of `color`'s piece of `kind`, which is also its bitboard's word."""
    return color << 3 | kind


@compiled
def forward(color: int) -> int:
    """The step from a square to the one a rank ahead, as `color` sees it."""
    return 8 - 16 * color


@compiled_inline
def make_move(position: Position, move: PackedMove) -> None:
    """Plays the legal `move` on `position` in place, recording how to undo it."""
    origin, target, promotion = move & 63, move >> 6 & 63, move >> 12
    us = signed(position[TURN])
    them = us ^ 1
    moved = signed(position[_MAILBOX + origin])
    captured = signed(position[_MAILBOX + target])
    ply = signed(position[PLY])

    record = UNDO + UNDO_SIZE * ply
    position[record] = move
    position[record + 1] = captured
    position[record + 2] = position[CASTLING]
    position[record + 3] = position[EN_PASSANT]
    position[PLY] = ply + 1

    if captured:
        position[captured] ^= SQUARE_BITS[target]
        position[OCCUPANCY + them] ^= SQUARE_BITS[target]
    _move_piece(position, moved, origin, target)

    en_passant = NO_SQUARE
    if moved & 7 == PAWN:
        if target == signed(position[EN_PASSANT]):
            victim = target - forward(us)
            position[piece(them, PAWN)] ^= SQUARE_BITS[victim]
            position[OCCUPANCY + them] ^= SQUARE_BITS[victim]
            position[_MAILBOX + victim] = 0
        elif abs(target - origin) == 16:
            en_passant = (origin + target) >> 1
        elif promotion:
            position[moved] ^= SQUARE_BITS[target]
            position[piece(us, promotion)] |= SQUARE_BITS[target]
            position[_MAILBOX + target] = piece(us, promotion)
    elif moved & 7 == KING and abs(target - origin) == 2:
        # The rook jumps from its corner to the square the king crossed.
        _move_piece(
            position,
            piece(us, ROOK),
            target + 1 if target > origin else target - 2,
            (origin + target) >> 1,
        )

    position[CASTLING] &= _CASTLING_KEPT[origin] & _CASTLING_KEPT[target]
    position[EN_PASSANT] = en_passant
    position[TURN] = them


@compiled_inline
def unmake_move(position: Position) -> None:
    """Takes back the last move made on `position`, in place."""
    ply = signed(position[PLY]) - 1
    record = UNDO + UNDO_SIZE * ply
    move = signed(position[record])
    origin, target, promotion = move & 63, move >> 6 & 63, move >> 12
    captured = signed(position[record + 1])
    position[CASTLING] = position[record + 2]
    position[EN_PASSANT] = position[record + 3]
    position[PLY] = ply
    them = signed(position[TURN])
    us = them ^ 1
    position[TURN] = us
    moved = signed(position[_MAILBOX + target])
    if promotion:
        position[moved] ^= SQUARE_BITS[target]
        moved = piece(us, PAWN)
        position[moved] |= SQUARE_BITS[target]
    _move_piece(position, moved, target, origin)
    if captured:
        position[captured] |= SQUARE_BITS[target]
        position[OCCUPANCY + them] |= SQUARE_BITS[target]
        position[_MAILBOX + target] = captured
    if moved & 7 == PAWN and target == signed(position[EN_PASSANT]):
        victim = target - forward(us)
        position[piece(them, PAWN)] |= SQUARE_BITS[victim]
        position[OCCUPANCY + them] |= SQUARE_BITS[victim]
        position[_MAILBOX + victim] = piece(them, PAWN)
    elif moved & 7 == KING and abs(target - origin) == 2:
        _move_piece(
            position,
            piece(us, ROOK),
            (origin + target) >> 1,
            target + 1 if target > origin else target - 2,
        )


@compiled
def _move_piece(position: Position, code: int, origin: Square, target: Square) -> None:
    """Moves the piece of `code` from `origin` to the empty `target`, in place."""
    squares = SQUARE_BITS[origin] | SQUARE_BITS[target]
    position[code] ^= squares
    position[OCCUPANCY + (code >> 3)] ^= squares
    position[_MAILBOX + origin] = 0
    position[_MAILBOX + target] = code


def _format_square(square: Square) -> str:
    """The name of `square`, like `e4`."""
    return "abcdefgh"[square % 8] + "12345678"[square // 8]


_PIECE_LETTERS = "pnbrqk"
_FEN = re.compile(
    r"""
    ([1-8pnbrqkPNBRQK]+(?:/[1-8pnbrqkPNBRQK]+){7})  # placement
    \ ([wb])  # side to move
    \ (-|K?Q?k?q?)  # castling
    \ (-|[a-h][36])  # en passant
    \ \d+\ \d+  # move counters
    """,
    re.VERBOSE,
)
# The castling flags a move keeps when it starts or ends on a square, so moving a king
# or rook, or capturing a rook, drops the rights that depend on it.
_CASTLING_KEPT = np.array(
    [
        ~sum(
            1 << flag
            for flag, (_, king, _, rook) in enumerate(CASTLING_RIGHTS)
            if square in {parse_square(king), parse_square(rook)}
        )
        & 15
        for square in range(64)
    ],
    np.uint64,
)
