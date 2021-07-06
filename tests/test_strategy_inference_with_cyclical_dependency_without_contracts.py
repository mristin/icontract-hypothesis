"""Test for a bug when classes introduce recursivity and depend on themselves in constructors."""

import unittest
from typing import Optional

import icontract
import hypothesis.strategies

import icontract_hypothesis


class Cup(icontract.DBC):
    """
    Represent a cup with a label and the cup next to it clockwise.

    From:
    https://github.com/mristin/python-by-contract-corpus/commit/f302812ad6bac20b97bde1b6b06ced2983e1f6dc
    """

    label: int  #: label of the cup
    next_cup: "Cup"  #: the next cup clockwise

    def __init__(self, label: int, next_cup: Optional["Cup"] = None) -> None:
        """Initialize with the given values."""
        self.label = label
        if next_cup:
            self.next_cup = next_cup
        else:
            self.next_cup = self


class TestStrategyInference(unittest.TestCase):
    def test_with_cup(self) -> None:
        strategy = hypothesis.strategies.from_type(Cup)
        self.assertEqual(
            "fixed_dictionaries({"
            "'label': integers(), "
            "'next_cup': one_of(none(), builds(Cup))})"
            ".map(lambda d: Cup(**d))",
            str(strategy),
        )

        # We can not execute this strategy, as ``builds`` is not handling the recursivity well.
        # Please see this Hypothesis issue:
        # https://github.com/HypothesisWorks/hypothesis/issues/3026


if __name__ == "__main__":
    unittest.main()
