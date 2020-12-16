"""Test tests.pyicontract_hypothesis.samples.sample_module with inferred Hypothesis strategies."""

import unittest

import hypothesis.strategies as st
from hypothesis import assume, given
import icontract_hypothesis

import tests.pyicontract_hypothesis.samples.sample_module


class TestWithInferredStrategies(unittest.TestCase):
    """Test all functions from tests.pyicontract_hypothesis.samples.sample_module with inferred Hypothesis strategies."""

    def test_square_greater_than_zero(self) -> None:
        assume_preconditions = icontract_hypothesis.make_assume_preconditions(
            func=tests.pyicontract_hypothesis.samples.sample_module.square_greater_than_zero)

        @given(
            x=st.integers())
        def execute(x) -> None:
            assume_preconditions(x)
            tests.pyicontract_hypothesis.samples.sample_module.square_greater_than_zero(x)

    def test_some_func(self) -> None:
        assume_preconditions = icontract_hypothesis.make_assume_preconditions(
            func=tests.pyicontract_hypothesis.samples.sample_module.some_func)

        @given(
            x=st.integers(min_value=1))
        def execute(x) -> None:
            assume_preconditions(x)
            tests.pyicontract_hypothesis.samples.sample_module.some_func(x)

    def test_another_func(self) -> None:
        assume_preconditions = icontract_hypothesis.make_assume_preconditions(
            func=tests.pyicontract_hypothesis.samples.sample_module.another_func)

        @given(
            x=st.integers(min_value=1).filter(lambda x: square_greater_than_zero(x)))
        def execute(x) -> None:
            assume_preconditions(x)
            tests.pyicontract_hypothesis.samples.sample_module.another_func(x)


if __name__ == '__main__':
    unittest.main()
