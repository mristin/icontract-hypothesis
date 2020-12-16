"""Test the automatic testing with inferred strategies."""

# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=unused-argument
# pylint: disable=no-value-for-parameter
import abc
import dataclasses
import enum
import math
import re
import sys
import unittest

if sys.version_info < (3, 8):
    from typing import List, NamedTuple, Union, Optional
else:
    from typing import TypedDict, List, NamedTuple, Union, Optional

import icontract

import icontract_hypothesis


class TestWithInferredStrategies(unittest.TestCase):
    def test_fail_without_type_hints(self) -> None:
        @icontract.require(lambda x: x > 0)
        def some_func(x) -> None:  # type: ignore
            pass

        type_error = None  # type: Optional[TypeError]
        try:
            icontract_hypothesis.test_with_inferred_strategies(some_func)
        except TypeError as err:
            type_error = err

        assert type_error is not None
        self.assertTrue(
            str(type_error).startswith(
                "No strategies could be inferred for the function: "
            )
        )

    def test_without_preconditions(self) -> None:
        def some_func(x: int) -> None:
            pass

        strategies = icontract_hypothesis.infer_strategies(some_func)
        self.assertEqual("{'x': integers()}", str(strategies))

        icontract_hypothesis.test_with_inferred_strategies(some_func)

    def test_unmatched_pattern(self) -> None:
        @icontract.require(lambda x: x > 0 and x > math.sqrt(x))
        def some_func(x: float) -> None:
            pass

        strategies = icontract_hypothesis.infer_strategies(some_func)
        self.assertEqual(
            "{'x': floats().filter(lambda x: x > 0 and x > math.sqrt(x))}",
            str(strategies),
        )

        icontract_hypothesis.test_with_inferred_strategies(some_func)


class TestWithInferredStrategiesOnClasses(unittest.TestCase):
    def test_no_preconditions_and_no_argument_init(self) -> None:
        class A:
            def __repr__(self) -> str:
                return "A()"

        def some_func(a: A) -> None:
            pass

        strategies = icontract_hypothesis.infer_strategies(some_func)
        self.assertEqual("{'a': builds(A)}", str(strategies))

        icontract_hypothesis.test_with_inferred_strategies(some_func)

    def test_no_preconditions_and_init(self) -> None:
        class A:
            def __init__(self, x: int):
                self.x = x

            def __repr__(self) -> str:
                return "A(x={})".format(self.x)

        def some_func(a: A) -> None:
            pass

        strategies = icontract_hypothesis.infer_strategies(some_func)
        self.assertEqual("{'a': builds(A)}", str(strategies))

        icontract_hypothesis.test_with_inferred_strategies(some_func)

    def test_preconditions_with_heuristics(self) -> None:
        class A(icontract.DBC):
            @icontract.require(lambda x: x > 0)
            def __init__(self, x: int):
                self.x = x

            def __repr__(self) -> str:
                return "A(x={})".format(self.x)

        def some_func(a: A) -> None:
            pass

        strategies = icontract_hypothesis.infer_strategies(some_func)
        self.assertEqual("{'a': builds(A, x=integers(min_value=1))}", str(strategies))

        icontract_hypothesis.test_with_inferred_strategies(some_func)

    def test_preconditions_without_heuristics(self) -> None:
        class A(icontract.DBC):
            @icontract.require(lambda x: x > 0)
            @icontract.require(lambda x: x > math.sqrt(x))
            def __init__(self, x: float):
                self.x = x

            def __repr__(self) -> str:
                return "A(x={})".format(self.x)

        def some_func(a: A) -> None:
            pass

        strategies = icontract_hypothesis.infer_strategies(some_func)
        self.assertEqual(
            "{'a': builds(A, x=floats(min_value=0, exclude_min=True)"
            ".filter(lambda x: x > math.sqrt(x)))}",
            str(strategies),
        )

        icontract_hypothesis.test_with_inferred_strategies(some_func)

    def test_composition(self) -> None:
        class A(icontract.DBC):
            @icontract.require(lambda x: x > 0)
            def __init__(self, x: int):
                self.x = x

            def __repr__(self) -> str:
                return "A(x={})".format(self.x)

        class B(icontract.DBC):
            @icontract.require(lambda y: y > 2020)
            def __init__(self, a: A, y: int):
                self.a = a
                self.y = y

            def __repr__(self) -> str:
                return "B(a={!r}, y={})".format(self.a, self.y)

        def some_func(b: B) -> None:
            pass

        strategies = icontract_hypothesis.infer_strategies(some_func)
        self.assertEqual(
            "{'b': builds(B, a=builds(A, x=integers(min_value=1)), y=integers(min_value=2021))}",
            str(strategies),
        )

        icontract_hypothesis.test_with_inferred_strategies(some_func)

    def test_abstract_class(self) -> None:
        class A(icontract.DBC):
            @abc.abstractmethod
            def do_something(self) -> None:
                pass

        class B(A):
            @icontract.require(lambda x: x > 0)
            def __init__(self, x: int):
                self.x = x

            def __repr__(self) -> str:
                return "B(x={})".format(self.x)

            def do_something(self) -> None:
                pass

        def some_func(a: A) -> None:
            pass

        strategies = icontract_hypothesis.infer_strategies(some_func)

        # The strategies inferred for B do not reflect the preconditions of A.
        # This is by design as A is automatically registered with Hypothesis, so Hypothesis
        # will instantiate a B only at run time.

        self.assertEqual("{'a': builds(B, x=integers(min_value=1))}", str(strategies))

        icontract_hypothesis.test_with_inferred_strategies(some_func)

    def test_enum(self) -> None:
        class A(enum.Enum):
            SOMETHING = 1
            ELSE = 2

        def some_func(a: A) -> None:
            pass

        strategies = icontract_hypothesis.infer_strategies(some_func)

        self.assertTrue(
            re.match(r"{'a': sampled_from\([a-zA-Z_0-9.]+\.A\)}", str(strategies)),
            str(strategies),
        )

        icontract_hypothesis.test_with_inferred_strategies(some_func)

    def test_composition_in_data_class(self) -> None:
        class A(icontract.DBC):
            @icontract.require(lambda x: x > 0)
            def __init__(self, x: int):
                self.x = x

            def __repr__(self) -> str:
                return "A(x={})".format(self.x)

        @dataclasses.dataclass
        class B:
            a: A

        def some_func(b: B) -> None:
            pass

        strategies = icontract_hypothesis.infer_strategies(some_func)

        # The strategies inferred for data classes do not reflect the preconditions of A.
        # This is by design as A is automatically registered with Hypothesis, so Hypothesis
        # will instantiate an A only at run time when creating B.
        self.assertEqual("{'b': builds(B)}", str(strategies))

        icontract_hypothesis.test_with_inferred_strategies(some_func)

    def test_typed_dict(self) -> None:
        # TypedDict not available below Python version 3.8.
        if sys.version_info >= (3, 8):

            class A(icontract.DBC):
                @icontract.require(lambda x: x > 0)
                def __init__(self, x: int):
                    self.x = x

                def __repr__(self) -> str:
                    return "A(x={})".format(self.x)

            class B(TypedDict):
                a: A

            def some_func(b: B) -> None:
                pass

            strategies = icontract_hypothesis.infer_strategies(some_func)
            self.assertEqual(
                "{'b': fixed_dictionaries({'a': builds(A, x=integers(min_value=1))}, optional={})}",
                str(strategies),
            )

            icontract_hypothesis.test_with_inferred_strategies(some_func)

    def test_list(self) -> None:
        class A(icontract.DBC):
            @icontract.require(lambda x: x > 0)
            def __init__(self, x: int):
                self.x = x

            def __repr__(self) -> str:
                return "A(x={})".format(self.x)

        def some_func(aa: List[A]) -> None:
            pass

        strategies = icontract_hypothesis.infer_strategies(some_func)
        self.assertEqual(
            "{'aa': lists(builds(A, x=integers(min_value=1)))}", str(strategies)
        )

        icontract_hypothesis.test_with_inferred_strategies(some_func)

    def test_named_tuples(self) -> None:
        class A(icontract.DBC):
            @icontract.require(lambda x: x > 0)
            def __init__(self, x: int):
                self.x = x

            def __repr__(self) -> str:
                return "A(x={})".format(self.x)

        class B(NamedTuple):
            a: A

        def some_func(b: B) -> None:
            pass

        strategies = icontract_hypothesis.infer_strategies(some_func)

        # The strategies inferred for named tuples do not reflect the preconditions of A.
        # This is by design as A is automatically registered with Hypothesis, so Hypothesis
        # will instantiate an A only at run time when creating B.
        self.assertEqual("{'b': builds(B)}", str(strategies))

        icontract_hypothesis.test_with_inferred_strategies(some_func)

    def test_union(self) -> None:
        class A(icontract.DBC):
            @icontract.require(lambda x: x > 0)
            def __init__(self, x: int):
                self.x = x

            def __repr__(self) -> str:
                return "A(x={})".format(self.x)

        class B(icontract.DBC):
            @icontract.require(lambda x: x < 0)
            def __init__(self, x: int):
                self.x = x

            def __repr__(self) -> str:
                return "B(x={})".format(self.x)

        def some_func(a_or_b: Union[A, B]) -> None:
            pass

        strategies = icontract_hypothesis.infer_strategies(some_func)
        self.assertEqual(
            "{'a_or_b': one_of(builds(A, x=integers(min_value=1)), "
            "builds(B, x=integers(max_value=-1)))}",
            str(strategies),
        )

        icontract_hypothesis.test_with_inferred_strategies(some_func)


if __name__ == "__main__":
    unittest.main()
