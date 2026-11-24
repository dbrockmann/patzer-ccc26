from __future__ import annotations
import time

import click

from ..board import board, perft
from ..board.position import STARTING_FEN, format_packed_uci


@click.command()
@click.option("--fen", default=STARTING_FEN, help="Position to count from.")
@click.option("--depth", type=click.IntRange(min=1), required=True, help="Plies.")
@click.option("--divide", is_flag=True, help="Count each root move on its own.")
def main(*, fen: str, depth: int, divide: bool) -> None:
    """Count the legal move sequences of a given depth, as a move generator test."""
    try:
        root = board.from_fen(fen)
    except ValueError as error:
        raise click.BadParameter(str(error), param_hint="--fen") from error
    # A shallow count first loads the compiled code, so the clock times counting only.
    perft.divide(root, min(depth, 2))
    start = time.perf_counter()
    counts = perft.divide(root, depth)
    seconds = time.perf_counter() - start
    if divide:
        for move, nodes in sorted(
            (format_packed_uci(move), nodes) for move, nodes in counts.items()
        ):
            click.echo(f"{move}: {nodes}")
    click.echo(f"nodes: {sum(counts.values())}")
    click.echo(f"seconds: {seconds:.6f}")


if __name__ == "__main__":
    main()
