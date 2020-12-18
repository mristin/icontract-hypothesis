import datetime
import decimal
import fractions
import math
import sys
import unittest

import icontract

import icontract_hypothesis


class TestMatchingBounds(unittest.TestCase):
    def test_different_ops(self) -> None:
        hundred = 100

        @icontract.require(lambda x: x > 0)
        @icontract.require(lambda x: x >= 1)
        @icontract.require(lambda x: x < 100)
        @icontract.require(lambda x: x <= 90)
        @icontract.require(lambda y: 0 < y <= 100)
        @icontract.require(lambda y: 1 <= y < 90)
        @icontract.require(lambda z: 0 > z >= -math.sqrt(hundred))
        def some_func(x: int, y: int, z: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries("
            "{'x': integers(min_value=2, max_value=89),\n"
            " 'y': integers(min_value=2, max_value=89),\n"
            " 'z': integers(min_value=-9.0, max_value=-1)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_with_dates(self) -> None:
        SOME_DATE = datetime.date(2014, 3, 2)

        # The preconditions were picked s.t. to also test that we can recompute everything.
        @icontract.require(lambda a: a < SOME_DATE + datetime.timedelta(days=3))
        @icontract.require(lambda b: b < SOME_DATE + datetime.timedelta(days=2))
        @icontract.require(lambda c: c < max(SOME_DATE, datetime.date(2020, 1, 1)))
        @icontract.require(
            lambda d: d
            < (
                SOME_DATE
                if SOME_DATE > datetime.date(2020, 1, 1)
                else datetime.date(2020, 12, 5)
            )
        )
        def some_func(
            a: datetime.date, b: datetime.date, c: datetime.date, d: datetime.date
        ) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries("
            "{'a': dates(max_value=datetime.date(2014, 3, 4)),\n"
            " 'b': dates(max_value=datetime.date(2014, 3, 3)),\n"
            " 'c': dates(max_value=datetime.date(2019, 12, 31)),\n"
            " 'd': dates(max_value=datetime.date(2020, 12, 4))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_with_datetimes(self) -> None:
        SOME_DATETIME = datetime.datetime(2014, 3, 2, 10, 20, 30)

        @icontract.require(lambda a: a < SOME_DATETIME)
        def some_func(a: datetime.datetime) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries("
            "{'a': datetimes(max_value=datetime.datetime(2014, 3, 2, 10, 20, 29, 999999))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_with_times(self) -> None:
        SOME_TIME = datetime.time(1, 2, 3)

        @icontract.require(lambda a: a < SOME_TIME)
        def some_func(a: datetime.time) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'a': times(max_value=datetime.time(1, 2, 2, 999999))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_with_timedeltas(self) -> None:
        SOME_TIMEDELTA = datetime.timedelta(days=3)

        @icontract.require(lambda a: a < SOME_TIMEDELTA)
        def some_func(a: datetime.timedelta) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        if sys.version_info < (3, 7):
            self.assertEqual(
                "fixed_dictionaries("
                "{'a': timedeltas(max_value=datetime.timedelta(2, 86399, 999999))})",
                str(strategy),
            )
        else:
            self.assertEqual(
                "fixed_dictionaries({"
                "'a': timedeltas(max_value=datetime.timedelta("
                "days=2, seconds=86399, microseconds=999999))})",
                str(strategy),
            )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_with_fractions(self) -> None:
        SOME_FRACTION = fractions.Fraction(3, 2)

        @icontract.require(lambda a: a < SOME_FRACTION)
        def some_func(a: fractions.Fraction) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries("
            "{'a': fractions(max_value=Fraction(3, 2))"
            ".filter(lambda a: a < SOME_FRACTION)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_with_decimals(self) -> None:
        SOME_DECIMAL = decimal.Decimal(10)

        @icontract.require(lambda a: a < SOME_DECIMAL)
        @icontract.require(lambda a: not decimal.Decimal.is_nan(a))
        def some_func(a: decimal.Decimal) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries("
            "{'a': decimals(max_value=Decimal('10'))"
            ".filter(lambda a: not decimal.Decimal.is_nan(a))"
            ".filter(lambda a: a < SOME_DECIMAL)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)


if __name__ == "__main__":
    unittest.main()
