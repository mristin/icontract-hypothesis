import datetime
import decimal
import fractions
import math
import sys
import unittest
from typing import Optional

import icontract

import icontract_hypothesis


class TestGeneral(unittest.TestCase):
    def test_argument_comparator_constant(self) -> None:
        @icontract.require(lambda x: x > 0)
        @icontract.require(lambda x: x >= 1)
        @icontract.require(lambda x: x < 100)
        @icontract.require(lambda x: x <= 90)
        def some_func(x: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': integers(min_value=1, max_value=90)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_constant_comparator_argument(self) -> None:
        @icontract.require(lambda x: 0 < x)
        @icontract.require(lambda x: 1 <= x)
        @icontract.require(lambda x: 100 > x)
        @icontract.require(lambda x: 90 >= x)
        def some_func(x: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': integers(min_value=1, max_value=90)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_constant_lt_argument_lt_argument(self) -> None:
        @icontract.require(lambda x: 1 < x < 10)
        def some_func(x: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': integers(min_value=2, max_value=9)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_constant_le_argument_lt_argument(self) -> None:
        @icontract.require(lambda x: 1 <= x < 10)
        def some_func(x: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': integers(min_value=1, max_value=9)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_constant_lt_argument_le_argument(self) -> None:
        @icontract.require(lambda x: 1 < x <= 10)
        def some_func(x: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': integers(min_value=2, max_value=10)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_constant_gt_argument_gt_argument(self) -> None:
        @icontract.require(lambda x: 100 > x > 10)
        def some_func(x: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': integers(min_value=11, max_value=99)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_constant_ge_argument_gt_argument(self) -> None:
        @icontract.require(lambda x: 100 >= x > 10)
        def some_func(x: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': integers(min_value=11, max_value=100)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_constant_gt_argument_ge_argument(self) -> None:
        @icontract.require(lambda x: 100 > x >= 10)
        def some_func(x: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': integers(min_value=10, max_value=99)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_comparison_with_recomputed_value(self) -> None:
        hundred = 100

        @icontract.require(lambda x: 0 > x >= -math.sqrt(hundred))
        def some_func(x: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': integers(min_value=-10.0, max_value=-1)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_multiple_arguments(self) -> None:
        hundred = 100

        @icontract.require(lambda x: x > 0)
        @icontract.require(lambda y: 0 < y <= 100)
        @icontract.require(lambda z: 0 > z >= -math.sqrt(hundred))
        def some_func(x: int, y: int, z: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries("
            "{'x': integers(min_value=1),\n"
            " 'y': integers(min_value=1, max_value=100),\n"
            " 'z': integers(min_value=-10.0, max_value=-1)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_complex_multi_line_condition(self) -> None:
        SOME_DATE = datetime.date(2014, 3, 2)

        # The preconditions were picked s.t. to also test that we can recompute everything.
        @icontract.require(
            lambda x: x
            < (
                SOME_DATE
                if SOME_DATE > datetime.date(2020, 1, 1)
                else datetime.date(2020, 12, 5)
            )
        )
        def some_func(x: datetime.date) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': dates(max_value=datetime.date(2020, 12, 4))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_correct_stacked_lower_bound(self) -> None:
        @icontract.require(lambda x: 3 <= x)
        @icontract.require(lambda x: 0 < x)
        def some_func(x: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': integers(min_value=3)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_correct_stacked_upper_bound(self) -> None:
        @icontract.require(lambda x: x <= 90)
        @icontract.require(lambda x: x < 100)
        def some_func(x: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': integers(max_value=90)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_unsatisfiable(self) -> None:
        @icontract.require(lambda x: x < 100)
        @icontract.require(lambda x: x > 1000)
        def some_func(x: int) -> None:
            pass

        value_error = None  # type: Optional[ValueError]
        try:
            _ = icontract_hypothesis.infer_strategy(some_func)
        except ValueError as error:
            value_error = error

        assert value_error is not None
        self.assertEqual(
            "The min and max values inferred for the argument x could not be satisfied: "
            "inferred min is 1000, inferred max is 100. Are your preconditions correct?",
            str(value_error),
        )


class TestInt(unittest.TestCase):
    def test_min_inclusive(self) -> None:
        @icontract.require(lambda x: 0 <= x)
        def some_func(x: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': integers(min_value=0)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_min_exclusive(self) -> None:
        @icontract.require(lambda x: 0 < x)
        def some_func(x: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': integers(min_value=1)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_max_inclusive(self) -> None:
        @icontract.require(lambda x: x <= 100)
        def some_func(x: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': integers(max_value=100)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_max_exclusive(self) -> None:
        @icontract.require(lambda x: x < 100)
        def some_func(x: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': integers(max_value=99)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_tightly_satisfiable_with_min_inclusive_and_max_exclusive(self) -> None:
        @icontract.require(lambda x: x < 100)
        @icontract.require(lambda x: x >= 99)
        def some_func(x: int) -> None:
            assert isinstance(x, int)

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': just(99)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_tightly_satisfiable_with_min_exclusive_and_max_inclusive(self) -> None:
        @icontract.require(lambda x: x <= 100)
        @icontract.require(lambda x: x > 99)
        def some_func(x: int) -> None:
            assert isinstance(x, int)

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': just(100)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)


class TestFloat(unittest.TestCase):
    def test_min_inclusive(self) -> None:
        @icontract.require(lambda x: 0 <= x)
        def some_func(x: float) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': floats(min_value=0)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_min_exclusive(self) -> None:
        @icontract.require(lambda x: 0 < x)
        def some_func(x: float) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': floats(min_value=0, exclude_min=True)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_max_inclusive(self) -> None:
        @icontract.require(lambda x: x <= 100)
        def some_func(x: float) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': floats(max_value=100)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_max_exclusive(self) -> None:
        @icontract.require(lambda x: x < 100)
        def some_func(x: float) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': floats(max_value=100, exclude_max=True)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)


class TestFraction(unittest.TestCase):
    def test_min_inclusive(self) -> None:
        @icontract.require(lambda x: fractions.Fraction(1, 2) <= x)
        def some_func(x: fractions.Fraction) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({" "'x': fractions(min_value=Fraction(1, 2))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_min_exclusive(self) -> None:
        @icontract.require(lambda x: fractions.Fraction(1, 2) < x)
        def some_func(x: fractions.Fraction) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({"
            "'x': fractions(min_value=Fraction(1, 2))"
            ".filter(lambda x: fractions.Fraction(1, 2) < x)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_max_inclusive(self) -> None:
        @icontract.require(lambda x: x <= fractions.Fraction(1, 2))
        def some_func(x: fractions.Fraction) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({" "'x': fractions(max_value=Fraction(1, 2))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_max_exclusive(self) -> None:
        @icontract.require(lambda x: x < fractions.Fraction(1, 2))
        def some_func(x: fractions.Fraction) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({"
            "'x': fractions(max_value=Fraction(1, 2))"
            ".filter(lambda x: x < fractions.Fraction(1, 2))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)


class TestDecimal(unittest.TestCase):
    def test_min_inclusive(self) -> None:
        @icontract.require(lambda x: decimal.Decimal("6.0") <= x)
        @icontract.require(lambda x: not x.is_nan())
        def some_func(x: decimal.Decimal) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({"
            "'x': decimals(min_value=Decimal('6.0'))"
            ".filter(lambda x: not x.is_nan())})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_min_exclusive(self) -> None:
        @icontract.require(lambda x: decimal.Decimal("6.0") < x)
        @icontract.require(lambda x: not x.is_nan())
        def some_func(x: decimal.Decimal) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({"
            "'x': decimals(min_value=Decimal('6.0'))"
            ".filter(lambda x: not x.is_nan())"
            '.filter(lambda x: decimal.Decimal("6.0") < x)})',
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_max_inclusive(self) -> None:
        @icontract.require(lambda x: x <= decimal.Decimal("6.0"))
        @icontract.require(lambda x: not x.is_nan())
        def some_func(x: decimal.Decimal) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({"
            "'x': decimals(max_value=Decimal('6.0'))"
            ".filter(lambda x: not x.is_nan())})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_max_exclusive(self) -> None:
        @icontract.require(lambda x: x < decimal.Decimal("6.0"))
        @icontract.require(lambda x: not x.is_nan())
        def some_func(x: decimal.Decimal) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({"
            "'x': decimals(max_value=Decimal('6.0'))"
            ".filter(lambda x: not x.is_nan())"
            '.filter(lambda x: x < decimal.Decimal("6.0"))})',
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)


class TestDate(unittest.TestCase):
    def test_min_inclusive(self) -> None:
        @icontract.require(lambda x: datetime.date(2014, 3, 2) <= x)
        def some_func(x: datetime.date) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': dates(min_value=datetime.date(2014, 3, 2))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_min_exclusive(self) -> None:
        @icontract.require(lambda x: datetime.date(2014, 3, 2) < x)
        def some_func(x: datetime.date) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': dates(min_value=datetime.date(2014, 3, 3))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_max_inclusive(self) -> None:
        @icontract.require(lambda x: x <= datetime.date(2014, 3, 2))
        def some_func(x: datetime.date) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': dates(max_value=datetime.date(2014, 3, 2))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_max_exclusive(self) -> None:
        @icontract.require(lambda x: x < datetime.date(2014, 3, 2))
        def some_func(x: datetime.date) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': dates(max_value=datetime.date(2014, 3, 1))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)


class TestDatetime(unittest.TestCase):
    def test_min_inclusive(self) -> None:
        @icontract.require(lambda x: datetime.datetime(2014, 3, 2, 1, 2, 3) <= x)
        def some_func(x: datetime.datetime) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({"
            "'x': datetimes(min_value=datetime.datetime(2014, 3, 2, 1, 2, 3))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_min_exclusive(self) -> None:
        @icontract.require(lambda x: datetime.datetime(2014, 3, 2, 1, 2, 3) < x)
        def some_func(x: datetime.datetime) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({"
            "'x': datetimes(min_value=datetime.datetime(2014, 3, 2, 1, 2, 3, 1))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_max_inclusive(self) -> None:
        @icontract.require(lambda x: x <= datetime.datetime(2014, 3, 2, 1, 2, 3))
        def some_func(x: datetime.datetime) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({"
            "'x': datetimes(max_value=datetime.datetime(2014, 3, 2, 1, 2, 3))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_max_exclusive(self) -> None:
        @icontract.require(lambda x: x < datetime.datetime(2014, 3, 2, 1, 2, 3))
        def some_func(x: datetime.datetime) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({"
            "'x': datetimes(max_value=datetime.datetime(2014, 3, 2, 1, 2, 2, 999999))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)


class TestTime(unittest.TestCase):
    def test_min_inclusive(self) -> None:
        @icontract.require(lambda x: datetime.time(1, 2, 3, 4) <= x)
        def some_func(x: datetime.time) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({" "'x': times(min_value=datetime.time(1, 2, 3, 4))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_min_exclusive(self) -> None:
        @icontract.require(lambda x: datetime.time(1, 2, 3, 4) < x)
        def some_func(x: datetime.time) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({" "'x': times(min_value=datetime.time(1, 2, 3, 5))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_max_inclusive(self) -> None:
        @icontract.require(lambda x: x <= datetime.time(1, 2, 3, 4))
        def some_func(x: datetime.time) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({" "'x': times(max_value=datetime.time(1, 2, 3, 4))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_max_exclusive(self) -> None:
        @icontract.require(lambda x: x < datetime.time(1, 2, 3, 4))
        def some_func(x: datetime.time) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({" "'x': times(max_value=datetime.time(1, 2, 3, 3))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_min_exclusive_too_high(self) -> None:
        @icontract.require(lambda x: datetime.time.max < x)
        def some_func(x: datetime.time) -> None:
            pass

        value_error = None  # type: Optional[ValueError]
        try:
            _ = icontract_hypothesis.infer_strategy(some_func)
        except ValueError as error:
            value_error = error

        assert value_error is not None
        self.assertEqual(
            "The inferred exclusive lower bound for the time "
            "is equal datetime.time.max (23:59:59.999999) "
            "so we can not compute the next greater time.",
            str(value_error),
        )

    def test_max_exclusive_too_low(self) -> None:
        @icontract.require(lambda x: x < datetime.time.min)
        def some_func(x: datetime.time) -> None:
            pass

        value_error = None  # type: Optional[ValueError]
        try:
            _ = icontract_hypothesis.infer_strategy(some_func)
        except ValueError as error:
            value_error = error

        assert value_error is not None
        self.assertEqual(
            "The inferred exclusive upper bound for the time "
            "is equal datetime.time.min (00:00:00) "
            "so we can not compute the previous less-than time.",
            str(value_error),
        )


class TestTimedelta(unittest.TestCase):
    def test_min_inclusive(self) -> None:
        @icontract.require(lambda x: datetime.timedelta(10) <= x)
        def some_func(x: datetime.timedelta) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)

        if sys.version_info < (3, 7):
            self.assertEqual(
                "fixed_dictionaries({"
                "'x': timedeltas(min_value=datetime.timedelta(10))})",
                str(strategy),
            )
        else:
            self.assertEqual(
                "fixed_dictionaries({"
                "'x': timedeltas(min_value=datetime.timedelta(days=10))})",
                str(strategy),
            )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_min_exclusive(self) -> None:
        @icontract.require(lambda x: datetime.timedelta(10) < x)
        def some_func(x: datetime.timedelta) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)

        if sys.version_info < (3, 7):
            self.assertEqual(
                "fixed_dictionaries({"
                "'x': timedeltas(min_value=datetime.timedelta(10, 0, 1))})",
                str(strategy),
            )
        else:
            self.assertEqual(
                "fixed_dictionaries({"
                "'x': timedeltas(min_value=datetime.timedelta(days=10, microseconds=1))})",
                str(strategy),
            )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_max_inclusive(self) -> None:
        @icontract.require(lambda x: x <= datetime.timedelta(10))
        def some_func(x: datetime.timedelta) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)

        if sys.version_info < (3, 7):
            self.assertEqual(
                "fixed_dictionaries({"
                "'x': timedeltas(max_value=datetime.timedelta(10))})",
                str(strategy),
            )
        else:
            self.assertEqual(
                "fixed_dictionaries({"
                "'x': timedeltas(max_value=datetime.timedelta(days=10))})",
                str(strategy),
            )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_max_exclusive(self) -> None:
        @icontract.require(lambda x: x < datetime.timedelta(10))
        def some_func(x: datetime.timedelta) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)

        if sys.version_info < (3, 7):
            self.assertEqual(
                "fixed_dictionaries({"
                "'x': timedeltas("
                "max_value=datetime.timedelta(9, 86399, 999999))})",
                str(strategy),
            )
        else:
            self.assertEqual(
                "fixed_dictionaries({"
                "'x': timedeltas("
                "max_value=datetime.timedelta(days=9, seconds=86399, microseconds=999999))})",
                str(strategy),
            )

        icontract_hypothesis.test_with_inferred_strategy(some_func)


if __name__ == "__main__":
    unittest.main()
