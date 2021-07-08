"""Test inference of the function to assume preconditions."""

# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=unused-argument
# pylint: disable=no-value-for-parameter

import unittest
from typing import Optional, List, Any

import hypothesis.errors
import hypothesis.strategies
import icontract

import icontract_hypothesis


class TestAssumePreconditions(unittest.TestCase):
    def test_assumed_preconditions_pass(self) -> None:
        @icontract.require(lambda x: x > 0)
        def some_func(x: int) -> None:
            pass

        assume_preconditions = icontract_hypothesis.make_assume_preconditions(some_func)

        assume_preconditions(x=100)

    def test_assumed_preconditions_fail(self) -> None:
        @icontract.require(lambda x: x > 0)
        def some_func(x: int) -> None:
            pass

        assume_preconditions = icontract_hypothesis.make_assume_preconditions(some_func)

        unsatisfied_assumption = (
            None
        )  # type: Optional[hypothesis.errors.UnsatisfiedAssumption]
        try:
            assume_preconditions(x=-100)
        except hypothesis.errors.UnsatisfiedAssumption as err:
            unsatisfied_assumption = err

        self.assertIsNotNone(unsatisfied_assumption)

    def test_without_contracts(self) -> None:
        recorded_inputs = []  # type: List[Any]

        def some_func(x: int) -> None:
            recorded_inputs.append(x)

        assume_preconditions = icontract_hypothesis.make_assume_preconditions(some_func)

        @hypothesis.given(x=hypothesis.strategies.integers())
        def execute(x: int) -> None:
            assume_preconditions(x)
            some_func(x)

        execute()

        # 10 is an arbitrary, but plausible value.
        self.assertGreater(len(recorded_inputs), 10)

    def test_with_a_single_precondition(self) -> None:
        recorded_inputs = []  # type: List[int]

        @icontract.require(lambda x: x > 0)
        def some_func(x: int) -> None:
            recorded_inputs.append(x)

        assume_preconditions = icontract_hypothesis.make_assume_preconditions(some_func)

        samples = [-1, 1]

        @hypothesis.given(x=hypothesis.strategies.sampled_from(samples))
        def execute(x: int) -> None:
            samples.append(x)
            assume_preconditions(x)
            some_func(x)

        execute()

        self.assertSetEqual({1}, set(recorded_inputs))

    def test_with_two_preconditions(self) -> None:
        recorded_inputs = []  # type: List[int]

        @icontract.require(lambda x: x > 0)
        @icontract.require(lambda x: x % 3 == 0)
        def some_func(x: int) -> None:
            recorded_inputs.append(x)

        assume_preconditions = icontract_hypothesis.make_assume_preconditions(some_func)

        samples = [-1, 1, 3]

        @hypothesis.given(x=hypothesis.strategies.sampled_from(samples))
        def execute(x: int) -> None:
            samples.append(x)
            assume_preconditions(x)
            some_func(x)

        execute()

        self.assertSetEqual({3}, set(recorded_inputs))

    def test_with_only_postconditions(self) -> None:
        recorded_inputs = []  # type: List[Any]

        @icontract.ensure(lambda result: result > 0)
        def some_func(x: int) -> int:
            recorded_inputs.append(x)
            return 1

        assume_preconditions = icontract_hypothesis.make_assume_preconditions(some_func)

        @hypothesis.given(x=hypothesis.strategies.integers())
        def execute(x: int) -> None:
            assume_preconditions(x)
            some_func(x)

        execute()

        self.assertGreater(len(recorded_inputs), 10)


class TestAssumeWeakenedPreconditions(unittest.TestCase):
    def test_with_a_single_precondition(self) -> None:
        class A(icontract.DBC):
            @icontract.require(lambda x: x % 3 == 0)
            def some_func(self, x: int) -> None:
                pass

        recorded_inputs = []  # type: List[int]

        class B(A):
            @icontract.require(lambda x: x % 7 == 0)
            def some_func(self, x: int) -> None:
                # The inputs from B.some_func need to satisfy either their own preconditions or
                # the preconditions of A.some_func ("require else").
                recorded_inputs.append(x)

        b = B()
        assume_preconditions = icontract_hypothesis.make_assume_preconditions(
            b.some_func
        )

        @hypothesis.given(x=hypothesis.strategies.sampled_from([-14, -3, 5, 7, 9]))
        def execute(x: int) -> None:
            assume_preconditions(x)
            b.some_func(x)

        execute()

        self.assertSetEqual({-14, -3, 7, 9}, set(recorded_inputs))

    def test_with_two_preconditions(self) -> None:
        class A(icontract.DBC):
            @icontract.require(lambda x: x % 3 == 0)
            def some_func(self, x: int) -> None:
                pass

        recorded_inputs = []  # type: List[int]

        class B(A):
            @icontract.require(lambda x: x > 0)
            @icontract.require(lambda x: x % 7 == 0)
            def some_func(self, x: int) -> None:
                # The inputs from B.some_func need to satisfy either their own preconditions or
                # the preconditions of A.some_func ("require else").
                recorded_inputs.append(x)

        b = B()
        assume_preconditions = icontract_hypothesis.make_assume_preconditions(
            b.some_func
        )

        @hypothesis.given(x=hypothesis.strategies.sampled_from([-14, 3, 7, 9, 10, 14]))
        def execute(x: int) -> None:
            assume_preconditions(x)
            b.some_func(x)

        execute()

        self.assertSetEqual({3, 7, 9, 14}, set(recorded_inputs))


if __name__ == "__main__":
    unittest.main()
