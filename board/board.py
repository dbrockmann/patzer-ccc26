"""Playing legal moves on a position from Python."""

from __future__ import annotations

from typing import NewType, cast

import numpy as np

from ..board import movegen
from ..board.compiled import CHECKED, compiled, signed
from ..board.position import (
    MAX_PLY,
    PLY,
    TURN,
    Position,
    make_move,
    parse_fen,
    unmake_move,
)

# A packed move that `legal_moves` gave for the position it is played on.
LegalMove = NewType("LegalMove", int)


def from_fen(fen: str) -> Position:
    """The position of a FEN with all six fields, ignoring the move counters.

    Raises:
        ValueError: If `parse_fen` rejects `fen`, or the side not to move is in check.
    """
    position = parse_fen(fen)
    if movegen.king_attacked(position, int(position[TURN]) ^ 1):
        message = "the side not to move is in check"
        raise ValueError(message)
    return position


def legal_moves(position: Position) -> list[LegalMove]:
    """The legal moves of `position`."""
    moves = np.empty(movegen.MAX_MOVES, np.int64)
    count = movegen.generate(position, moves, count_only=False)
    return cast("list[LegalMove]", moves[:count].tolist())


@compiled
def push(position: Position, move: LegalMove) -> None:
    """Plays `move` on `position` in place.

    Raises:
        IndexError: If MAX_PLY moves have been played.
        ValueError: If checked, and `move` isn't legal on `position`.
    """
    if signed(position[PLY]) == MAX_PLY:
        message = "too many moves for the undo stack"
        raise IndexError(message)
    if CHECKED and not _is_legal(position, move):
        message = "not a legal move of this position"
        raise ValueError(message)
    make_move(position, move)


@compiled
def pop(position: Position) -> None:
    """Takes back the last move played on `position`, in place.

    Raises:
        IndexError: If no move has been played.
    """
    if signed(position[PLY]) == 0:
        message = "no move to take back"
        raise IndexError(message)
    unmake_move(position)


def is_check(position: Position) -> bool:
    """Whether the side to move is in check."""
    return movegen.king_attacked(position, int(position[TURN]))


def is_checkmate(position: Position) -> bool:
    """Whether the side to move is in check and has no legal move."""
    return is_check(position) and not legal_moves(position)


@compiled
def _is_legal(position: Position, move: LegalMove) -> bool:
    moves = np.empty(movegen.MAX_MOVES, np.int64)
    return move in moves[: movegen.generate(position, moves, count_only=False)]
