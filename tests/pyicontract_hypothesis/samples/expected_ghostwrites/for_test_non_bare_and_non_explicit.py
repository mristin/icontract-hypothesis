"""Test tests.pyicontract_hypothesis.samples.sample_module with inferred Hypothesis strategies."""

import unittest

import icontract_hypothesis

import tests.pyicontract_hypothesis.samples.sample_module


class TestWithInferredStrategies(unittest.TestCase):
    """Test all functions from tests.pyicontract_hypothesis.samples.sample_module with inferred Hypothesis strategies."""

    def test_square_greater_than_zero(self) -> None:
        icontract.integration.with_hypothesis.test_with_inferred_strategies(
                func=tests.pyicontract_hypothesis.samples.sample_module.square_greater_than_zero)

    def test_some_func(self) -> None:
        icontract.integration.with_hypothesis.test_with_inferred_strategies(
                func=tests.pyicontract_hypothesis.samples.sample_module.some_func)

    def test_another_func(self) -> None:
        icontract.integration.with_hypothesis.test_with_inferred_strategies(
                func=tests.pyicontract_hypothesis.samples.sample_module.another_func)


if __name__ == '__main__':
    unittest.main()
