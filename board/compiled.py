"""The boundary to Numba: compiling functions, and operations compiled code needs."""

from __future__ import annotations

# llvmlite ships no types and Numba's intrinsic and config APIs are untyped, which
# this module wraps.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import hashlib
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numba
import numpy as np
from llvmlite import binding, ir
from numba import types
from numba.core import config
from numba.extending import intrinsic

if TYPE_CHECKING:
    from collections.abc import Callable

type Bits = np.uint64

if not binding.get_host_cpu_features().get("bmi2"):
    _message = "the board needs an x86-64 CPU with BMI2 for PEXT and PDEP"
    raise ImportError(_message)

# Numba's cache misses changes to other modules and to bounds checking, so it is keyed
# by all of the sources and the bounds checking, and only the current one is kept.
_PACKAGE = Path(__file__).parent
_SOURCES_CACHE = (
    _PACKAGE
    / "__pycache__"
    / "numba"
    / hashlib.sha256(
        b"".join(path.read_bytes() for path in sorted(_PACKAGE.glob("*.py")))
    ).hexdigest()[:16]
)
if not _SOURCES_CACHE.exists():
    shutil.rmtree(_SOURCES_CACHE.parent, ignore_errors=True)
# Whether compiled code checks its inputs, which follows Numba's bounds checking.
# Compiled code reads this as a constant, so unchecked code carries no trace of the
# checks.
CHECKED = bool(config.BOUNDSCHECK)  # pyright: ignore[reportAttributeAccessIssue]
config.CACHE_DIR = str(  # pyright: ignore[reportAttributeAccessIssue]
    _SOURCES_CACHE / ("bounds-checked" if CHECKED else "unchecked")
)


def compiled[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """`function` compiled to machine code for this CPU, cached between runs."""
    return numba.njit(cache=True)(function)


def compiled_inline[**P, R](function: Callable[P, R]) -> Callable[P, R]:
    """Like `compiled`, with its body copied into every compiled caller."""
    return numba.njit(cache=True, inline="always")(function)


@compiled
def lowest_square(bits: Bits) -> int:
    """The square of the lowest set bit of nonzero `bits`."""
    return int(_trailing_zeros(bits))


@compiled
def popcount(bits: Bits) -> int:
    """The number of set bits."""
    return int(_population(bits))


@compiled
def extract(bits: Bits, mask: Bits) -> int:
    """The bits of `bits` under `mask`, packed into the low bits (PEXT)."""
    return int(_extract(bits, mask))


@compiled
def deposit(bits: int, mask: Bits) -> Bits:
    """The low bits of `bits` spread over the set bits of `mask` (PDEP)."""
    return _deposit(np.uint64(bits), mask)


@compiled
def signed(word: Bits) -> int:
    """`word` as a signed 64-bit integer."""
    # Numba keeps int() of a uint64 unsigned, and mixing it with ints gives floats.
    return int(np.int64(word))


def _unary(name: str) -> Callable[[Bits], Bits]:
    """The LLVM bit operation `name` on a uint64, callable from compiled code."""

    def typer(_context: object, _bits: object) -> tuple[object, object]:
        def codegen(
            _context: object,
            builder: ir.IRBuilder,
            _signature: object,
            args: list[object],
        ) -> object:
            if name == "ctpop":
                return builder.ctpop(args[0])
            # Leaving a zero input undefined lets x86 emit tzcnt.
            return getattr(builder, name)(args[0], ir.Constant(ir.IntType(1), 1))

        return types.uint64(types.uint64), codegen

    return cast("Callable[[Bits], Bits]", intrinsic(typer))


def _bmi2(name: str) -> Callable[[Bits, Bits], Bits]:
    """The x86 BMI2 instruction `name` on two uint64, callable from compiled code."""

    def typer(
        _context: object, _source: object, _mask: object
    ) -> tuple[object, object]:
        def codegen(
            _context: object,
            builder: ir.IRBuilder,
            _signature: object,
            args: list[object],
        ) -> object:
            word = ir.IntType(64)
            function = builder.module.declare_intrinsic(
                f"llvm.x86.bmi.{name}.64", fnty=ir.FunctionType(word, [word, word])
            )
            return builder.call(function, args)

        return types.uint64(types.uint64, types.uint64), codegen

    return cast("Callable[[Bits, Bits], Bits]", intrinsic(typer))


_trailing_zeros = _unary("cttz")
_population = _unary("ctpop")
# PEXT turns the blockers into a table index with one instruction and no magic numbers,
# at the cost of ARM and of x86 without BMI2.
_extract = _bmi2("pext")
_deposit = _bmi2("pdep")
