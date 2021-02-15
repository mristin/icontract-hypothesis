"""Test tests.pyicontract_hypothesis.samples.well_formatted_with_two_arguments with inferred Hypothesis strategies."""

import unittest

from hypothesis import given

import tests.pyicontract_hypothesis.samples.well_formatted_with_two_arguments


class TestWithInferredStrategies(unittest.TestCase):
    """Test all functions from tests.pyicontract_hypothesis.samples.well_formatted_with_two_arguments with inferred Hypothesis strategies."""

    def test_add(self) -> None:
        @given(
            a=integers(),
            b=integers()
        )
        def execute(kwargs) -> None:
            tests.pyicontract_hypothesis.samples.well_formatted_with_two_arguments.add(**kwargs)

        execute()


if __name__ == '__main__':
    unittest.main()
