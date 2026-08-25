"""Argument validation shared by every planning entry point.

The project's one definition of "an integer input" and of "a rank sequence". Not to be
confused with :mod:`resihp.verify`, which checks the numerical Principle A contracts:
this module checks *arguments*, before any planning happens.

``bool`` is deliberately not an integer here. ``True`` is otherwise indistinguishable
from the degree ``1`` or the rank ``1``, and would silently plan a topology nobody
asked for.
"""

from numbers import Integral
from typing import Iterable


def _integer(name: str, value: object, *, minimum: int, qualifier: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be a {qualifier} integer")
    return int(value)


def positive_int(name: str, value: object) -> int:
    """``value`` as an ``int``, rejecting anything that is not an integer ``>= 1``."""
    return _integer(name, value, minimum=1, qualifier="positive")


def non_negative_int(name: str, value: object) -> int:
    """``value`` as an ``int``, rejecting anything that is not an integer ``>= 0``."""
    return _integer(name, value, minimum=0, qualifier="non-negative")


def unique_ranks(name: str, values: Iterable[int]) -> tuple[int, ...]:
    """A rank set as a sorted tuple, rejecting non-integers and repeats.

    Sorting here is what makes a planner result independent of the order its caller
    happened to discover the ranks in.
    """
    ranks = tuple(non_negative_int(name, value) for value in values)
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"{name} must not contain duplicate ranks")
    return tuple(sorted(ranks))
