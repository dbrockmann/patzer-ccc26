"""The board behind the course's `AbstractChessBoard`."""

from typing import final, override

from ccc26.board import AbstractChessBoard

from .board import board
from .board.position import STARTING_FEN, Position, format_packed_uci


@final
class _Move(int):
    """A packed legal move whose `str` gives its UCI notation."""

    __slots__ = ()

    @override
    def __str__(self) -> str:
        return format_packed_uci(self)


@final
class PatzerBoard(AbstractChessBoard):
    """A board for the course's perft framework, whose moves print as UCI."""

    _position: Position

    def __init__(self, fen: str | None = None) -> None:
        super().__init__(fen)
        self._position = board.from_fen(fen or STARTING_FEN)

    @override
    def get_legal_moves(self) -> list[_Move]:
        return [_Move(move) for move in board.legal_moves(self._position)]

    @override
    def push_move(self, move: _Move) -> None:
        # Numba takes a plain int faster than a subclass of one.
        board.push(self._position, board.LegalMove(int(move)))

    @override
    def pop_move(self) -> None:
        board.pop(self._position)
