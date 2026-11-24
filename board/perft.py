"""Counting a position's legal move sequences, compiled, to test move generation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..board.board import LegalMove, legal_moves, pop, push
from ..board.compiled import compiled
from ..board.movegen import MAX_MOVES, generate
from ..board.position import MAX_PLY, PLY, Position, make_move, unmake_move


def divide(position: Position, depth: int) -> dict[LegalMove, int]:
    """The number of legal move sequences of `depth` plies, per move.

    Raises:
        ValueError: If `depth` is below 1.
        IndexError: If the moves made to count, all but the last ply, would pass
            MAX_PLY.
    """
    if depth < 1:
        message = f"depth {depth} is below 1"
        raise ValueError(message)
    if int(position[PLY]) + depth - 1 > MAX_PLY:
        message = "depth too deep for the undo stack"
        raise IndexError(message)
    moves = np.empty((depth, MAX_MOVES), np.int64)
    counts: dict[LegalMove, int] = {}
    for move in legal_moves(position):
        push(position, move)
        counts[move] = _perft(position, depth - 1, moves) if depth > 1 else 1
        pop(position)
    return counts


@compiled
def _perft(position: Position, depth: int, moves: NDArray[np.int64]) -> int:
    """The number of legal move sequences of `depth` plies.

    Row k of `moves` is scratch space for the call at depth k.
    """
    if depth == 1:
        return generate(position, moves[1], count_only=True)
    total = 0
    for index in range(generate(position, moves[depth], count_only=False)):
        make_move(position, int(moves[depth, index]))
        total += _perft(position, depth - 1, moves)
        unmake_move(position)
    return total
