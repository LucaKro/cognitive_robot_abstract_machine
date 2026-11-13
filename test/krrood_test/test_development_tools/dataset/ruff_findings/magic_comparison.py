"""
A module comparing with a bare number, next to the trivial ones that are allowed.
"""

from __future__ import annotations


def is_crowded(people: int) -> bool:
    """
    A threshold written as a literal.
    """
    return people > 12  # reported


def is_alone(people: int) -> bool:
    """
    A comparison with one, which needs no name.
    """
    return people == 1
