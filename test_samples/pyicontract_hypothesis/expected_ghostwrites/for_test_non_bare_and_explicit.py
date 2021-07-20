"""Test test_samples.pyicontract_hypothesis.sample_module with inferred Hypothesis strategies."""

import unittest

from hypothesis import given

import test_samples.pyicontract_hypothesis.sample_module


class TestWithInferredStrategies(unittest.TestCase):
    """Test all functions from test_samples.pyicontract_hypothesis.sample_module with inferred Hypothesis strategies."""

    def test_some_func(self) -> None:
        @given(
            x=integers(min_value=1)
        )
        def execute(**kwargs) -> None:
            test_samples.pyicontract_hypothesis.sample_module.some_func(**kwargs)

        execute()

    def test_another_func(self) -> None:
        @given(
            x=integers(min_value=1).filter(lambda x: square_greater_than_zero(x))
        )
        def execute(**kwargs) -> None:
            test_samples.pyicontract_hypothesis.sample_module.another_func(**kwargs)

        execute()

    def test_yet_another_func(self) -> None:
        @given(
            fixed_dictionaries({'x': integers(), 'y': integers()}).filter(lambda d: d['x'] < d['y'])
        )
        def execute(kwargs) -> None:
            test_samples.pyicontract_hypothesis.sample_module.yet_another_func(**kwargs)

        execute()


if __name__ == '__main__':
    unittest.main()
