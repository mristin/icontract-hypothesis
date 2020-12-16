import datetime
import decimal
import fractions
import math
import sys
import unittest
from typing import List, Any

import icontract

import icontract_hypothesis


class TestMatchingBounds(unittest.TestCase):
    def test_different_ops(self) -> None:
        recorded_inputs = []  # type: List[Any]

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

        strategies = icontract_hypothesis.infer_strategies(some_func)
        self.assertEqual(
            "{'x': integers(min_value=2, max_value=89), 'y': integers(min_value=2, max_value=89), "
            "'z': integers(min_value=-9.0, max_value=-1)}",
            str(strategies),
        )

        icontract_hypothesis.test_with_inferred_strategies(some_func)

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

        strategies = icontract_hypothesis.infer_strategies(some_func)
        self.assertEqual(
            "{'a': dates(max_value=datetime.date(2014, 3, 5)), "
            "'b': dates(max_value=datetime.date(2014, 3, 4)), "
            "'c': dates(max_value=datetime.date(2020, 1, 1)), "
            "'d': dates(max_value=datetime.date(2020, 12, 5))}",
            str(strategies),
        )

        icontract_hypothesis.test_with_inferred_strategies(some_func)

    def test_with_datetimes(self) -> None:
        SOME_DATETIME = datetime.datetime(2014, 3, 2, 10, 20, 30)

        @icontract.require(lambda a: a < SOME_DATETIME)
        def some_func(a: datetime.datetime) -> None:
            pass

        strategies = icontract_hypothesis.infer_strategies(some_func)
        self.assertEqual(
            "{'a': datetimes(max_value=datetime.datetime(2014, 3, 2, 10, 20, 30))}",
            str(strategies),
        )

        icontract_hypothesis.test_with_inferred_strategies(some_func)

    def test_with_times(self) -> None:
        SOME_TIME = datetime.time(1, 2, 3)

        @icontract.require(lambda a: a < SOME_TIME)
        def some_func(a: datetime.time) -> None:
            pass

        strategies = icontract_hypothesis.infer_strategies(some_func)
        self.assertEqual(
            "{'a': times(max_value=datetime.time(1, 2, 3))}", str(strategies)
        )

        icontract_hypothesis.test_with_inferred_strategies(some_func)

    def test_with_timedeltas(self) -> None:
        SOME_TIMEDELTA = datetime.timedelta(days=3)

        @icontract.require(lambda a: a < SOME_TIMEDELTA)
        def some_func(a: datetime.timedelta) -> None:
            pass

        strategies = icontract_hypothesis.infer_strategies(some_func)
        if sys.version_info < (3, 7):
            self.assertEqual(
                "{'a': timedeltas(max_value=datetime.timedelta(3))}", str(strategies)
            )
        else:
            self.assertEqual(
                "{'a': timedeltas(max_value=datetime.timedelta(days=3))}",
                str(strategies),
            )

        icontract_hypothesis.test_with_inferred_strategies(some_func)

    def test_with_fractions(self) -> None:
        SOME_FRACTION = fractions.Fraction(3, 2)

        @icontract.require(lambda a: a < SOME_FRACTION)
        def some_func(a: fractions.Fraction) -> None:
            pass

        strategies = icontract_hypothesis.infer_strategies(some_func)
        self.assertEqual("{'a': fractions(max_value=Fraction(3, 2))}", str(strategies))

        icontract_hypothesis.test_with_inferred_strategies(some_func)

    def test_with_decimals(self) -> None:
        SOME_DECIMAL = decimal.Decimal(10)

        @icontract.require(lambda a: not decimal.Decimal.is_nan(a))
        @icontract.require(lambda a: a < SOME_DECIMAL)
        def some_func(a: decimal.Decimal) -> None:
            pass

        strategies = icontract_hypothesis.infer_strategies(some_func)
        self.assertEqual(
            "{'a': decimals(max_value=Decimal('10'))"
            ".filter(lambda a: not decimal.Decimal.is_nan(a))}",
            str(strategies),
        )

        icontract_hypothesis.test_with_inferred_strategies(some_func)

    def test_with_weakened_preconditions(self) -> None:
        class A(icontract.DBC):
            @icontract.require(lambda x: 0 < x < 20)
            @icontract.require(lambda x: x % 3 == 0)
            def some_func(self, x: int) -> None:
                pass

        class B(A):
            @icontract.require(lambda x: 0 < x < 20)
            @icontract.require(lambda x: x % 7 == 0)
            def some_func(self, x: int) -> None:
                # The inputs from B.some_func need to satisfy either their own preconditions or
                # the preconditions of A.some_func ("require else").
                pass

        b = B()

        strategies = icontract_hypothesis.infer_strategies(b.some_func)
        self.assertEqual(
            "{'x': one_of("
            "integers(min_value=1, max_value=19).filter(lambda x: x % 3 == 0), "
            "integers(min_value=1, max_value=19).filter(lambda x: x % 7 == 0))}",
            str(strategies),
        )

        icontract_hypothesis.test_with_inferred_strategies(b.some_func)


if __name__ == "__main__":
    unittest.main()
