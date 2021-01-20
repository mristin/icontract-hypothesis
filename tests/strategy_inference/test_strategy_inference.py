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
from typing import List, NamedTuple, Union, Optional, Any, Mapping

if sys.version_info >= (3, 8):
    from typing import TypedDict

import hypothesis
import hypothesis.strategies as st
import icontract

import icontract_hypothesis

SOME_GLOBAL_CONST = 0


class TestWithInferredStrategies(unittest.TestCase):
    def test_fail_without_type_hints(self) -> None:
        @icontract.require(lambda x: x > 0)
        def some_func(x) -> None:  # type: ignore
            pass

        type_error = None  # type: Optional[TypeError]
        try:
            icontract_hypothesis.test_with_inferred_strategy(some_func)
        except TypeError as err:
            type_error = err

        assert type_error is not None
        self.assertTrue(
            re.match(
                r"No search strategy could be inferred for the function: <function .*>; "
                r"the following arguments are missing the type annotations: \['x'\]",
                str(type_error),
            ),
            str(type_error),
        )

    def test_without_preconditions(self) -> None:
        def some_func(x: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual("fixed_dictionaries({'x': integers()})", str(strategy))

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_unmatched_pattern(self) -> None:
        @icontract.require(lambda x: x > 0 and x > math.sqrt(x))
        def some_func(x: float) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': floats().filter(lambda x: x > 0 and x > math.sqrt(x))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_resorting_to_from_type(self) -> None:
        # We can not handle ``.startswith`` at the moment, so we expect
        # ``from_type`` Hypothesis strategy followed by a filter.
        @icontract.require(lambda x: x.startswith("oioi"))
        def some_func(x: str) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': text().filter(lambda x: x.startswith(\"oioi\"))})",
            str(strategy),
        )

        # We explicitly do not test with this strategy as it will not pass the health check.

    def test_snippet_given(self) -> None:
        @icontract.require(lambda x, y: x < y)
        def some_func(x: float, y: float) -> None:
            pass

        @hypothesis.given(
            st.fixed_dictionaries({"x": st.floats(), "y": st.floats()}).filter(
                lambda d: d["x"] < d["y"]
            )
        )
        def test(kwargs: Mapping[str, Any]) -> None:
            some_func(**kwargs)

        test()

    def test_multi_argument_contract(self) -> None:
        @icontract.require(lambda x, y: x < y)
        def some_func(x: float, y: float) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': floats(), 'y': floats()}).filter(lambda d: d['x'] < d['y'])",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_multi_argument_contract_with_closure(self) -> None:
        SOME_CONSTANT = -math.inf
        ANOTHER_CONSTANT = math.inf

        # We must have -512 and +512 so that EXTENDED_ARG opcode is tested as well.
        @icontract.require(
            lambda x, y: SOME_CONSTANT
            < x - 512
            < SOME_GLOBAL_CONST
            < y + 512
            < ANOTHER_CONSTANT
        )
        def some_func(x: float, y: float) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': floats(), 'y': floats()}).filter(lambda d: SOME_CONSTANT\n"
            "    < d['x'] - 512\n"
            "    < SOME_GLOBAL_CONST\n"
            "    < d['y'] + 512\n"
            "    < ANOTHER_CONSTANT)",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_condition_with_function(self) -> None:
        def some_precondition(x: int) -> bool:
            return x > 0

        @icontract.require(some_precondition)
        def some_func(x: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': integers().filter(some_precondition)})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_condition_with_function_on__KWARGS(self) -> None:
        def some_precondition(_KWARGS: Mapping[str, Any]) -> bool:
            return len(_KWARGS) == 1

        @icontract.require(some_precondition)
        def some_func(x: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': integers()}).filter(lambda d: some_precondition(_KWARGS=d))",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_condition_with_function_on_argument_and_KWARGS(self) -> None:
        def some_precondition(x: int, _KWARGS: Mapping[str, Any]) -> bool:
            return x > len(_KWARGS)

        @icontract.require(some_precondition)
        def some_func(x: int, **kwargs: Any) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': integers()})"
            ".filter(lambda d: some_precondition(x=d['x'], _KWARGS=d))",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_condition_on_kwargs(self) -> None:
        @icontract.require(lambda _KWARGS: len(_KWARGS) == 1)
        def some_func(x: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': integers()}).filter(lambda d: len(d) == 1)",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_condition_without_arguments(self) -> None:
        @icontract.require(lambda: SOME_GLOBAL_CONST >= 0)
        def some_func(x: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'x': integers()}).filter(lambda: SOME_GLOBAL_CONST >= 0)",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_initial_kwargs_conflicts_with_local_but_not_with_any_global_or_closure(
        self,
    ) -> None:
        @icontract.require(lambda d, e: d < e)
        def some_func(d: int, e: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'d': integers(), 'e': integers()})"
            ".filter(lambda d: d['d'] < d['e'])",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_initial_kwargs_conflicts_with_the_closure(self) -> None:
        d = 0

        @icontract.require(lambda x: x + d >= 0)
        @icontract.require(lambda x, y: math.sqrt(x + d) < y)
        def some_func(x: int, y: int) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries("
            "{'x': integers().filter(lambda x: x + d >= 0), "
            "'y': integers()})"
            ".filter(lambda _d: math.sqrt(_d['x'] + d) < _d['y'])",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)


class TestWithInferredStrategiesOnClasses(unittest.TestCase):
    def test_no_preconditions_and_no_argument_init(self) -> None:
        class A:
            def __repr__(self) -> str:
                return "A()"

        def some_func(a: A) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual("fixed_dictionaries({'a': builds(A)})", str(strategy))

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_no_preconditions_and_init(self) -> None:
        class A:
            def __init__(self, x: int):
                self.x = x

            def __repr__(self) -> str:
                return "A(x={})".format(self.x)

        def some_func(a: A) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual("fixed_dictionaries({'a': builds(A)})", str(strategy))

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_map_snippet(self) -> None:
        import hypothesis.strategies as st

        class A(icontract.DBC):
            def __init__(self, x: int, y: int):
                self.x = x
                self.y = y

        def some_func(a: A) -> None:
            pass

        @hypothesis.given(
            a=st.fixed_dictionaries({"x": st.integers(), "y": st.integers()})
            .filter(lambda d: d["x"] < d["y"])
            .map(lambda d: A(**d))
        )
        def test(a: A) -> None:
            some_func(a)

        test()

    def test_from_type(self) -> None:
        class A(icontract.DBC):
            @icontract.require(lambda x, y: x < y)
            def __init__(self, x: int, y: int):
                self.x = x
                self.y = y

            def __repr__(self) -> str:
                return "A(x={}, y={})".format(self.x, self.y)

        strategy = hypothesis.strategies.from_type(A)
        self.assertEqual(
            "fixed_dictionaries("
            "{'x': integers(), 'y': integers()})"
            ".filter(lambda d: d['x'] < d['y'])"
            ".map(lambda d: A(**d))",
            str(strategy),
        )

        @hypothesis.given(a=hypothesis.strategies.from_type(A))
        def test(a: A) -> None:
            ...

        test()

    def test_preconditions_with_multiple_arguments(self) -> None:
        class A(icontract.DBC):
            @icontract.require(lambda x, y: x < y)
            def __init__(self, x: int, y: int):
                self.x = x
                self.y = y

            def __repr__(self) -> str:
                return "A(x={}, y={})".format(self.x, self.y)

        def some_func(a: A) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'a': fixed_dictionaries({'x': integers(), 'y': integers()})"
            ".filter(lambda d: d['x'] < d['y'])"
            ".map(lambda d: A(**d))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_preconditions_with_heuristics(self) -> None:
        class A(icontract.DBC):
            @icontract.require(lambda x: x > 0)
            def __init__(self, x: int):
                self.x = x

            def __repr__(self) -> str:
                return "A(x={})".format(self.x)

        def some_func(a: A) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries("
            "{'a': fixed_dictionaries({'x': integers(min_value=1)})"
            ".map(lambda d: A(**d))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

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

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries("
            "{'a': fixed_dictionaries("
            "{'x': floats(min_value=0, exclude_min=True)"
            ".filter(lambda x: x > math.sqrt(x))})"
            ".map(lambda d: A(**d))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

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

        strategy = icontract_hypothesis.infer_strategy(b.some_func)
        self.assertEqual(
            "one_of("
            "fixed_dictionaries("
            "{'x': integers(min_value=1, max_value=19)"
            ".filter(lambda x: x % 3 == 0)}), "
            "fixed_dictionaries({'x': integers(min_value=1, max_value=19)"
            ".filter(lambda x: x % 7 == 0)}))",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(b.some_func)

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

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries("
            "{'b': "
            "fixed_dictionaries("
            "{'a': "
            "fixed_dictionaries("
            "{'x': integers(min_value=1)}).map(lambda d: A(**d)),\n"
            "  'y': integers(min_value=2021)}).map(lambda d: B(**d))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

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

        strategy = icontract_hypothesis.infer_strategy(some_func)

        # The strategies inferred for B do not reflect the preconditions of A.
        # This is by design as A is automatically registered with Hypothesis, so Hypothesis
        # will instantiate a B only at run time.

        self.assertEqual(
            "fixed_dictionaries("
            "{'a': fixed_dictionaries("
            "{'x': integers(min_value=1)})"
            ".map(lambda d: B(**d))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_enum(self) -> None:
        class A(enum.Enum):
            SOMETHING = 1
            ELSE = 2

        def some_func(a: A) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)

        self.assertTrue(
            re.match(
                r"fixed_dictionaries\({'a': sampled_from\([a-zA-Z_0-9.]+\.A\)}\)",
                str(strategy),
            ),
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

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

        strategy = icontract_hypothesis.infer_strategy(some_func)

        # The strategies inferred for data classes do not reflect the preconditions of A.
        # This is by design as A is automatically registered with Hypothesis, so Hypothesis
        # will instantiate an A only at run time when creating B.
        self.assertEqual("fixed_dictionaries({'b': builds(B)})", str(strategy))

        icontract_hypothesis.test_with_inferred_strategy(some_func)

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

            strategy = icontract_hypothesis.infer_strategy(some_func)
            self.assertEqual(
                "fixed_dictionaries("
                "{'b': fixed_dictionaries("
                "{'a': fixed_dictionaries("
                "{'x': integers(min_value=1)})"
                ".map(lambda d: A(**d))}, optional={})})",
                str(strategy),
            )

            icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_list(self) -> None:
        class A(icontract.DBC):
            @icontract.require(lambda x: x > 0)
            def __init__(self, x: int):
                self.x = x

            def __repr__(self) -> str:
                return "A(x={})".format(self.x)

        def some_func(aa: List[A]) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries("
            "{'aa': lists("
            "fixed_dictionaries({'x': integers(min_value=1)})"
            ".map(lambda d: A(**d)))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

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

        strategy = icontract_hypothesis.infer_strategy(some_func)

        # The strategies inferred for named tuples do not reflect the preconditions of A.
        # This is by design as A is automatically registered with Hypothesis, so Hypothesis
        # will instantiate an A only at run time when creating B.
        self.assertEqual("fixed_dictionaries({'b': builds(B)})", str(strategy))

        icontract_hypothesis.test_with_inferred_strategy(some_func)

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

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries("
            "{'a_or_b': one_of("
            "fixed_dictionaries({'x': integers(min_value=1)}).map(lambda d: A(**d)), "
            "fixed_dictionaries({'x': integers(max_value=-1)}).map(lambda d: B(**d)))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)


if __name__ == "__main__":
    unittest.main()
