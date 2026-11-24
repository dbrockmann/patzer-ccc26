"""Legal move generation on bitboards, compiled."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..board.compiled import (
    Bits,
    compiled,
    compiled_inline,
    deposit,
    extract,
    lowest_square,
    popcount,
    signed,
)
from ..board.position import (
    BISHOP,
    BLACK,
    CASTLING,
    CASTLING_RIGHTS,
    EN_PASSANT,
    KING,
    KNIGHT,
    NO_SQUARE,
    OCCUPANCY,
    PAWN,
    QUEEN,
    ROOK,
    SQUARE_BITS,
    TURN,
    WHITE,
    Position,
    Square,
    forward,
    parse_square,
    piece,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

# Packed moves, as `PackedMove` describes them.
type Moves = NDArray[np.int64]

# More than the legal moves of any position `parse_fen` accepts: no piece has more
# moves than a queen's 27, so 16 pieces a side stay below 15 * 27 + 8 + 2 castlings.
MAX_MOVES = 512


@compiled
def king_attacked(position: Position, color: int) -> bool:
    """Whether an enemy piece attacks the king of `color`."""
    return bool(
        _checkers(position, color, position[OCCUPANCY] | position[OCCUPANCY + 1])
    )


@compiled_inline
def generate(position: Position, moves: Moves, count_only: bool) -> int:
    """Writes the legal moves of `position` to `moves` and returns their number.

    With `count_only`, only counts them and leaves `moves` as it is.
    """
    us = signed(position[TURN])
    them = us ^ 1
    own = position[OCCUPANCY + us]
    occupied = own | position[OCCUPANCY + them]
    king = lowest_square(_pieces(position, us, KING))
    checkers = _checkers(position, us, occupied)
    attacked = _attacked(position, them, occupied ^ SQUARE_BITS[king])
    count = _add(moves, 0, count_only, king, _KING_ATTACKS[king] & ~own & ~attacked)
    if popcount(checkers) > 1:
        return count  # only the king can escape a double check

    # The squares the other pieces may move to: in check, capturing or blocking the
    # checker, otherwise anywhere but onto their own pieces.
    if checkers:
        allowed = checkers | _BETWEEN[king, lowest_square(checkers)]
    else:
        allowed = ~own
        for flag, origin, target, empty, safe in _CASTLINGS[us]:
            if (
                position[CASTLING] & np.uint64(flag)
                and not occupied & np.uint64(empty)
                and not attacked & np.uint64(safe)
            ):
                count = _add(moves, count, count_only, origin, SQUARE_BITS[target])

    pinned = _pinned(position, them, king, own, occupied)
    pieces = _pieces(position, us, KNIGHT) & ~pinned  # a pinned knight can't move
    while pieces:
        origin = lowest_square(pieces)
        count = _add(
            moves, count, count_only, origin, _KNIGHT_ATTACKS[origin] & allowed
        )
        pieces &= pieces - _ONE  # drops the lowest piece

    for kind in (BISHOP, ROOK, QUEEN):
        pieces = _pieces(position, us, kind)
        while pieces:
            origin = lowest_square(pieces)
            targets = _EMPTY
            if kind != ROOK:
                targets |= _bishop_attacks(origin, occupied)
            if kind != BISHOP:
                targets |= _rook_attacks(origin, occupied)
            if pinned & SQUARE_BITS[origin]:
                targets &= _LINE[king, origin]  # stays on the line to its king
            count = _add(moves, count, count_only, origin, targets & allowed)
            pieces &= pieces - _ONE

    pawns = _pieces(position, us, PAWN)
    en_passant = signed(position[EN_PASSANT])
    if en_passant != NO_SQUARE:
        capturers = _PAWN_ATTACKS[them, en_passant] & pawns
        while capturers:
            origin = lowest_square(capturers)
            if _en_passant_legal(position, us, origin, en_passant):
                count = _add(moves, count, count_only, origin, SQUARE_BITS[en_passant])
            capturers &= capturers - _ONE

    # Pinned pawns go one at a time, each along its own line to the king, and the
    # free ones all at once.
    enemy = occupied & ~own
    pinned_pawns = pawns & pinned
    while pinned_pawns:
        origin = lowest_square(pinned_pawns)
        count = _add_pawns(
            moves,
            count,
            count_only,
            us,
            SQUARE_BITS[origin],
            occupied,
            enemy,
            allowed & _LINE[king, origin],
        )
        pinned_pawns &= pinned_pawns - _ONE
    return _add_pawns(
        moves, count, count_only, us, pawns & ~pinned, occupied, enemy, allowed
    )


@compiled_inline
def _pieces(position: Position, color: int, kind: int) -> Bits:
    """The bitboard of `color`'s pieces of `kind`."""
    return position[piece(color, kind)]


@compiled_inline
def _checkers(position: Position, us: int, occupied: Bits) -> Bits:
    """The enemy pieces that attack the king of `us`."""
    them = us ^ 1
    king = lowest_square(_pieces(position, us, KING))
    queens = _pieces(position, them, QUEEN)
    return (
        (_KNIGHT_ATTACKS[king] & _pieces(position, them, KNIGHT))
        | (_PAWN_ATTACKS[us, king] & _pieces(position, them, PAWN))
        | (_rook_attacks(king, occupied) & (_pieces(position, them, ROOK) | queens))
        | (_bishop_attacks(king, occupied) & (_pieces(position, them, BISHOP) | queens))
    )


@compiled_inline
def _attacked(position: Position, them: int, occupied: Bits) -> Bits:
    """The squares the pieces of `them` attack.

    `occupied` leaves out the defending king, so that it can't step back along a ray.
    """
    toward_a, toward_h = _pawn_captures(_pieces(position, them, PAWN), them)
    attacked = (
        _KING_ATTACKS[lowest_square(_pieces(position, them, KING))]
        | toward_a
        | toward_h
    )
    pieces = _pieces(position, them, KNIGHT)
    while pieces:
        attacked |= _KNIGHT_ATTACKS[lowest_square(pieces)]
        pieces &= pieces - _ONE
    queens = _pieces(position, them, QUEEN)
    pieces = _pieces(position, them, ROOK) | queens
    while pieces:
        attacked |= _rook_attacks(lowest_square(pieces), occupied)
        pieces &= pieces - _ONE
    pieces = _pieces(position, them, BISHOP) | queens
    while pieces:
        attacked |= _bishop_attacks(lowest_square(pieces), occupied)
        pieces &= pieces - _ONE
    return attacked


@compiled_inline
def _pinned(
    position: Position, them: int, king: Square, own: Bits, occupied: Bits
) -> Bits:
    """The own pieces that alone stand between `king` and an enemy slider's line."""
    queens = _pieces(position, them, QUEEN)
    pinners = (
        _rook_attacks(king, _EMPTY) & (_pieces(position, them, ROOK) | queens)
    ) | (_bishop_attacks(king, _EMPTY) & (_pieces(position, them, BISHOP) | queens))
    pinned = _EMPTY
    while pinners:
        between = _BETWEEN[king, lowest_square(pinners)] & occupied
        if between & own and popcount(between) == 1:
            pinned |= between
        pinners &= pinners - _ONE
    return pinned


@compiled
def _en_passant_legal(
    position: Position, us: int, origin: Square, target: Square
) -> bool:
    """Whether capturing en passant from `origin` to `target` leaves the king safe.

    Rare enough to check the whole position after the capture, which also takes away
    the captured pawn and can open the king's rank.
    """
    victim = target - forward(us)
    occupied = (
        (position[OCCUPANCY] | position[OCCUPANCY + 1])
        ^ SQUARE_BITS[origin]
        ^ SQUARE_BITS[target]
        ^ SQUARE_BITS[victim]
    )
    return not _checkers(position, us, occupied) & ~SQUARE_BITS[victim]


@compiled
def _add(
    moves: Moves,
    count: int,
    count_only: bool,
    origin: Square,
    targets: Bits,
) -> int:
    """Adds the moves from `origin` to each of `targets`, and returns the new count."""
    if count_only:
        return count + popcount(targets)
    while targets:
        moves[count] = origin | lowest_square(targets) << 6
        count += 1
        targets &= targets - _ONE
    return count


@compiled
def _add_pawns(
    moves: Moves,
    count: int,
    count_only: bool,
    us: int,
    pawns: Bits,
    occupied: Bits,
    enemy: Bits,
    allowed: Bits,
) -> int:
    """Adds the pushes and captures of `pawns` to squares in `allowed`."""
    step = forward(us)
    single = _advance(pawns, us) & ~occupied
    double = _advance(single & _DOUBLE_STEP_RANKS[us], us) & ~occupied
    toward_a, toward_h = _pawn_captures(pawns, us)
    count = _add_pawn_targets(moves, count, count_only, us, single & allowed, step)
    count = _add_pawn_targets(moves, count, count_only, us, double & allowed, 2 * step)
    count = _add_pawn_targets(
        moves, count, count_only, us, toward_a & enemy & allowed, step - 1
    )
    return _add_pawn_targets(
        moves, count, count_only, us, toward_h & enemy & allowed, step + 1
    )


@compiled_inline
def _advance(pawns: Bits, color: int) -> Bits:
    """`color`'s `pawns` moved one rank forward."""
    return pawns << _ONE_RANK if color == WHITE else pawns >> _ONE_RANK


@compiled_inline
def _pawn_captures(pawns: Bits, color: int) -> tuple[Bits, Bits]:
    """The squares `color`'s `pawns` capture on, toward the a-file and the h-file."""
    if color == WHITE:
        return (pawns & _NOT_FILE_A) << _SEVEN, (pawns & _NOT_FILE_H) << _NINE
    return (pawns & _NOT_FILE_A) >> _NINE, (pawns & _NOT_FILE_H) >> _SEVEN


@compiled
def _add_pawn_targets(
    moves: Moves,
    count: int,
    count_only: bool,
    us: int,
    targets: Bits,
    step: int,
) -> int:
    """Adds the pawn moves onto `targets`, each from `step` squares behind."""
    promotions = targets & _LAST_RANKS[us]
    if count_only:
        return count + popcount(targets) + 3 * popcount(promotions)
    while targets:
        target = lowest_square(targets)
        move = target - step | target << 6
        if SQUARE_BITS[target] & promotions:
            for promotion in (QUEEN, ROOK, BISHOP, KNIGHT):
                moves[count] = move | promotion << 12
                count += 1
        else:
            moves[count] = move
            count += 1
        targets &= targets - _ONE
    return count


@compiled
def _rook_attacks(square: Square, occupied: Bits) -> Bits:
    return _ROOK_TABLE[
        int(_ROOK_OFFSETS[square]) + extract(occupied, _ROOK_MASKS[square])
    ]


@compiled
def _bishop_attacks(square: Square, occupied: Bits) -> Bits:
    return _BISHOP_TABLE[
        int(_BISHOP_OFFSETS[square]) + extract(occupied, _BISHOP_MASKS[square])
    ]


def _rays(square: Square, steps: Iterable[tuple[int, int]]) -> list[list[Square]]:
    """The squares in each (file, rank) step direction from `square` to the edge."""
    file, rank = square % 8, square // 8
    return [
        [
            (rank + rank_step * distance) * 8 + file + file_step * distance
            for distance in range(1, 8)
            if 0 <= file + file_step * distance < 8
            and 0 <= rank + rank_step * distance < 8
        ]
        for file_step, rank_step in steps
    ]


def _bits(squares: Iterable[Square]) -> int:
    return sum(1 << square for square in set(squares))


def _table(bitboards: Iterable[int]) -> NDArray[np.uint64]:
    return np.array(list(bitboards), np.uint64)


@compiled
def _fill_slider_table(
    masks: NDArray[np.uint64],
    offsets: NDArray[np.int64],
    attacks: NDArray[np.uint64],
    steps: tuple[tuple[int, int], ...],
) -> None:
    """Fills `attacks` with each square's attacks for every occupancy of its mask."""
    for square in range(64):
        for index in range(offsets[square + 1] - offsets[square]):
            occupied = deposit(index, masks[square])
            targets = _EMPTY
            for file_step, rank_step in steps:
                file, rank = square % 8 + file_step, square // 8 + rank_step
                while 0 <= file < 8 and 0 <= rank < 8:
                    targets |= SQUARE_BITS[rank * 8 + file]
                    if occupied & SQUARE_BITS[rank * 8 + file]:
                        break
                    file += file_step
                    rank += rank_step
            attacks[offsets[square] + index] = targets


def _slider_tables(
    steps: tuple[tuple[int, int], ...],
) -> tuple[NDArray[np.uint64], NDArray[np.int64], NDArray[np.uint64]]:
    """For each square, the blocker mask, the table offset, and all attack tables.

    The mask leaves out each ray's last square, since a piece there doesn't change the
    attacks, and PEXT turns the occupancy under the mask into the table index.
    """
    masks = _table(
        _bits(square for ray in _rays(origin, steps) for square in ray[:-1])
        for origin in range(64)
    )
    offsets = np.cumsum([0] + [1 << int(mask).bit_count() for mask in masks]).astype(
        np.int64
    )
    attacks = np.empty(offsets[-1], np.uint64)
    _fill_slider_table(masks, offsets, attacks, steps)
    return masks, offsets[:-1].copy(), attacks


_EMPTY = np.uint64(0)
_ONE = np.uint64(1)
_SEVEN = np.uint64(7)
_ONE_RANK = np.uint64(8)
_NINE = np.uint64(9)
_ORTHOGONAL = ((1, 0), (-1, 0), (0, 1), (0, -1))
_DIAGONAL = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_KNIGHT_STEPS = ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2))

_KNIGHT_ATTACKS = _table(
    _bits(ray[0] for ray in _rays(square, _KNIGHT_STEPS) if ray) for square in range(64)
)
_KING_ATTACKS = _table(
    _bits(ray[0] for ray in _rays(square, _ORTHOGONAL + _DIAGONAL) if ray)
    for square in range(64)
)
# The squares a pawn of each color attacks from each square.
_PAWN_ATTACKS = np.array(
    [
        [
            _bits(
                ray[0]
                for ray in _rays(square, ((-1, rank_step), (1, rank_step)))
                if ray
            )
            for square in range(64)
        ]
        for rank_step in (1, -1)
    ],
    np.uint64,
)
_BETWEEN = np.zeros((64, 64), np.uint64)
# From a square through another one in line to the edge, the line a pinned piece keeps.
_LINE = np.zeros((64, 64), np.uint64)
for _origin in range(64):
    for _ray in _rays(_origin, _ORTHOGONAL + _DIAGONAL):
        for _distance, _square in enumerate(_ray):
            _BETWEEN[_origin, _square] = _bits(_ray[:_distance])
            _LINE[_origin, _square] = _bits(_ray)
_ROOK_MASKS, _ROOK_OFFSETS, _ROOK_TABLE = _slider_tables(_ORTHOGONAL)
_BISHOP_MASKS, _BISHOP_OFFSETS, _BISHOP_TABLE = _slider_tables(_DIAGONAL)

_FILE_A = _bits(range(0, 64, 8))
_NOT_FILE_A = np.uint64(~_FILE_A & (1 << 64) - 1)
_NOT_FILE_H = np.uint64(~(_FILE_A << 7) & (1 << 64) - 1)
_DOUBLE_STEP_RANKS = _table(_bits(range(rank * 8, rank * 8 + 8)) for rank in (2, 5))
_LAST_RANKS = _table(_bits(range(rank * 8, rank * 8 + 8)) for rank in (7, 0))
# Per color: the castling flag, the king's start and target, the squares between king
# and rook that have to be empty, and the squares the king crosses or lands on, which
# have to be safe.
_CASTLINGS = tuple(
    tuple(
        (
            1 << flag,
            parse_square(king),
            parse_square(target),
            int(_BETWEEN[parse_square(king), parse_square(rook)]),
            int(_BETWEEN[parse_square(king), parse_square(target)])
            | 1 << parse_square(target),
        )
        for flag, (letter, king, target, rook) in enumerate(CASTLING_RIGHTS)
        if letter.isupper() == (color == WHITE)
    )
    for color in (WHITE, BLACK)
)
