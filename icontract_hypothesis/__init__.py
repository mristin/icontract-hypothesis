"""
Integrate icontract with Hypothesis.

You need to install ``hypothesis`` extras in order to use this module.
"""
# pylint: disable=unused-argument
# pylint: disable=protected-access
# pylint: disable=inconsistent-return-statements
# pylint: disable=too-many-lines

import ast
import datetime
import decimal
import dis
import fractions
import inspect
import io
import re
import sys
import types
import typing
from typing import (
    cast,
    TypeVar,
    Callable,
    Any,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Union,
    Dict,
    Tuple,
    Type,
    AnyStr,
    Pattern,
    Set,
)

import hypothesis.errors
import hypothesis.internal.reflection
import hypothesis.strategies
import hypothesis.strategies._internal.types
import hypothesis.strategies._internal.collections
import icontract._checkers
import icontract._metaclass
import icontract._recompute
import icontract._represent
import icontract._types

# Don't forget to update the version in setup.py!
__version__ = "1.1.0"
__author__ = "Marko Ristin"
__copyright__ = "Copyright 2020 Marko Ristin"
__license__ = "MIT"
__status__ = "Production/Stable"

CallableT = TypeVar("CallableT", bound=Callable[..., Any])

T = TypeVar("T")  # pylint: disable=invalid-name


def _assume_preconditions_always_satisfied(
    *args: Tuple[Any, ...], **kwargs: Dict[str, Any]
) -> None:
    """Assume that the preconditions are always satisfied (say, when there are no preconditions)."""


def make_assume_preconditions(func: CallableT) -> Callable[..., None]:
    """
    Create a function that assumes the preconditions are satisfied given the positional and keyword arguments.

    Here is an example test case which tests a function ``some_func``:

    >>> import unittest
    >>> import hypothesis.strategies as st
    >>> import icontract_hypothesis
    >>> import icontract

    >>> @icontract.require(lambda x: x > 0)
    ... def some_func(x: int) -> None:
    ...     ...

    >>> class TestSomething(unittest.TestCase):
    ...     def test_that_it_works(self) -> None:
    ...         assume_preconditions = icontract_hypothesis.make_assume_preconditions(
    ...             some_func)
    ...
    ...         @hypothesis.given(x=st.integers())
    ...         def run(x: int) -> None:
    ...             assume_preconditions(x)
    ...             some_func(x)
    ...         run()

    >>> TestSomething().test_that_it_works()
    """
    # The implementation follows tightly icontract._checkers.decorate_with_checker and
    # icontract._checkers._assert_precondition.

    checker = icontract._checkers.find_checker(func)
    if checker is None:
        return _assume_preconditions_always_satisfied

    preconditions = getattr(checker, "__preconditions__", None)
    if preconditions is None:
        return _assume_preconditions_always_satisfied

    sign = inspect.signature(func)
    param_names = list(sign.parameters.keys())
    kwdefaults = icontract._checkers.resolve_kwdefaults(sign=sign)

    def assume_preconditions(*args, **kwargs) -> None:  # type: ignore
        """Accept only positional and keyword arguments that satisfy one of the precondition groups."""
        resolved_kwargs = icontract._checkers.kwargs_from_call(
            param_names=param_names, kwdefaults=kwdefaults, args=args, kwargs=kwargs
        )

        success = True

        for group in preconditions:  # pylint: disable=not-an-iterable
            success = True
            for contract in group:
                condition_kwargs = icontract._checkers.select_condition_kwargs(
                    contract=contract, resolved_kwargs=resolved_kwargs
                )

                check = contract.condition(**condition_kwargs)

                if icontract._checkers.not_check(check=check, contract=contract):
                    success = False
                    break

            if success:
                return

        if not success:
            raise hypothesis.errors.UnsatisfiedAssumption()

    return assume_preconditions


class _InferredMinMax:
    """Represent the inference result of boundaries on an argument."""

    def __init__(
        self,
        min_value: Optional[Any] = None,
        min_inclusive: bool = False,
        max_value: Optional[Any] = None,
        max_inclusive: bool = False,
    ) -> None:
        """Initialize with the given values."""
        self.min_value = min_value
        self.min_inclusive = min_inclusive
        self.max_value = max_value
        self.max_inclusive = max_inclusive


def _no_name_in_descendants(root: ast.expr, name: str) -> bool:
    """Check whether a ``ast.Name`` node with ``root`` identifier is present in the descendants of the node."""
    found = False

    class Visitor(ast.NodeVisitor):
        """Search for the name node."""

        def visit_Name(  # pylint: disable=invalid-name,no-self-use,missing-docstring
            self, node: ast.Name
        ) -> None:
            if node.id == name:
                nonlocal found
                found = True

        def generic_visit(self, node: Any) -> None:
            if not found:
                super(Visitor, self).generic_visit(node)

    visitor = Visitor()
    visitor.visit(root)

    return not found


def _recompute(condition: Callable[..., Any], node: ast.expr) -> Tuple[Any, bool]:
    """Recompute the value corresponding to the node."""
    recompute_visitor = icontract._recompute.Visitor(
        variable_lookup=icontract._represent.collect_variable_lookup(
            condition=condition, condition_kwargs=None
        )
    )

    recompute_visitor.visit(node=node)

    if node in recompute_visitor.recomputed_values:
        return recompute_visitor.recomputed_values[node], True

    return None, False


def _infer_min_max_from_node(
    condition: Callable[..., bool], node: ast.Compare, arg_name: str
) -> Optional[_InferredMinMax]:
    """Match one of the patterns against the AST compare node."""
    # pylint: disable=too-many-boolean-expressions
    # pylint: disable=too-many-return-statements
    # pylint: disable=too-many-branches
    if len(node.comparators) == 1:
        comparator = node.comparators[0]
        operation = node.ops[0]

        # Match something like "x > 0" and "x < 100"
        if (
            isinstance(node.left, ast.Name)
            and node.left.id == arg_name
            and _no_name_in_descendants(root=comparator, name=arg_name)
        ):
            value, recomputed = _recompute(condition=condition, node=comparator)

            # If we can not recompute the value, we also can not infer the bounds.
            if not recomputed:
                return None

            # Match something like "x < 100"
            if isinstance(operation, ast.Lt):
                return _InferredMinMax(max_value=value, max_inclusive=False)
            # Match something like "x =< 100"
            elif isinstance(operation, ast.LtE):
                return _InferredMinMax(max_value=value, max_inclusive=True)

            # Match something like "x > 0"
            elif isinstance(operation, ast.Gt):
                return _InferredMinMax(min_value=value, min_inclusive=False)
            # Match something like "x >= 0"
            elif isinstance(operation, ast.GtE):
                return _InferredMinMax(min_value=value, min_inclusive=True)

            # We could not infer any bound from this condition.
            else:
                return None

        # Match something like "0 < x" and "100 > x"
        if (
            _no_name_in_descendants(root=node.left, name=arg_name)
            and isinstance(comparator, ast.Name)
            and comparator.id == arg_name
        ):
            value, recomputed = _recompute(condition=condition, node=node.left)

            # If we can not recompute the value, we also can not infer the bounds.
            if not recomputed:
                return None

            # Match something like "0 < x"
            if isinstance(operation, ast.Lt):
                return _InferredMinMax(min_value=value, min_inclusive=False)

            # Match something like "0 =< x"
            if isinstance(operation, ast.LtE):
                return _InferredMinMax(min_value=value, min_inclusive=True)

            # Match something like "100 > x"
            elif isinstance(operation, ast.Gt):
                return _InferredMinMax(max_value=value, max_inclusive=False)

            # Match something like "100 >= x"
            elif isinstance(operation, ast.GtE):
                return _InferredMinMax(max_value=value, max_inclusive=True)

            # We could not infer any bound from this condition.
            else:
                pass

    elif len(node.comparators) == 2:
        # Match something like "0 < x < 100" and "0 > x > -100"
        if (
            _no_name_in_descendants(root=node.left, name=arg_name)
            and isinstance(node.comparators[0], ast.Name)
            and node.comparators[0].id == arg_name
            and _no_name_in_descendants(root=node.comparators[1], name=arg_name)
        ):

            left_value, recomputed = _recompute(condition=condition, node=node.left)

            # If we can not recompute the left value, we also can not infer the bounds.
            if not recomputed:
                return None

            right_value, recomputed = _recompute(
                condition=condition, node=node.comparators[1]
            )

            # If we can not recompute the right value, we also can not infer the bounds.
            if not recomputed:
                return None

            op0, op1 = node.ops

            # Match something like "0 < x < 100"
            if isinstance(op0, (ast.Lt, ast.LtE)) and isinstance(
                op1, (ast.Lt, ast.LtE)
            ):
                return _InferredMinMax(
                    min_value=left_value,
                    min_inclusive=isinstance(op0, ast.LtE),
                    max_value=right_value,
                    max_inclusive=isinstance(op1, ast.LtE),
                )

            # Match something like "0 > x > -100"
            elif isinstance(op0, (ast.Gt, ast.GtE)) and isinstance(
                op1, (ast.Gt, ast.GtE)
            ):
                return _InferredMinMax(
                    min_value=right_value,
                    min_inclusive=isinstance(op0, ast.GtE),
                    max_value=left_value,
                    max_inclusive=isinstance(op1, ast.GtE),
                )

            # We could not infer any bound from this condition.
            else:
                pass

    return None


def _body_node_from_condition(condition: Callable[..., Any]) -> Optional[ast.expr]:
    """Try to extract the body node of the contract's lambda condition."""
    if not icontract._represent.is_lambda(a_function=condition):
        return None

    lines, condition_lineno = inspect.findsource(condition)
    filename = inspect.getsourcefile(condition)
    assert filename is not None

    decorator_inspection = icontract._represent.inspect_decorator(
        lines=lines, lineno=condition_lineno, filename=filename
    )
    lambda_inspection = icontract._represent.find_lambda_condition(
        decorator_inspection=decorator_inspection
    )

    assert (
        lambda_inspection is not None
    ), "Expected lambda_inspection to be non-None if _is_lambda is True on: {}".format(
        condition
    )

    body_node = lambda_inspection.node.body

    return body_node


@icontract.require(
    lambda arg_name, contracts: all(
        len(contract.condition_args) == 1 and contract.condition_args[0] == arg_name
        for contract in contracts
    ),
    "All contracts are single-argument contracts related to this argument.",
)
def _infer_min_max_from_preconditions(
    arg_name: str, type_hint: Type[T], contracts: List[icontract._types.Contract]
) -> Tuple[_InferredMinMax, List[icontract._types.Contract]]:
    """
    Infer the min and max values for the given argument from all related preconditions.

    Return the contracts which could not be interpreted.
    """
    min_value = None  # type: Optional[Union[int, float]]
    max_value = None  # type: Optional[Union[int, float]]

    remaining_contracts = []  # type: List[icontract._types.Contract]

    for contract in contracts:
        body_node = _body_node_from_condition(condition=contract.condition)

        if body_node is None:
            remaining_contracts.append(contract)
            continue

        if isinstance(body_node, ast.Compare):
            inferred = _infer_min_max_from_node(
                condition=contract.condition, node=body_node, arg_name=arg_name
            )

            if inferred is not None:
                # We need to constrain min and max values.
                # Hence we use ``max`` for min and ``min`` for max, respectively.
                # This might be a bit counter-intuitive at the first sight.

                if inferred.min_value is not None:
                    min_value = (
                        inferred.min_value
                        if min_value is None
                        else max(inferred.min_value, min_value)
                    )

                if inferred.max_value is not None:
                    max_value = (
                        inferred.max_value
                        if max_value is None
                        else min(inferred.max_value, max_value)
                    )

                # We can not have exclusive bounds on Fractions and Decimals, so we need to leave
                # the contracts in place.
                if type_hint in [fractions.Fraction, decimal.Decimal]:
                    remaining_contracts.append(contract)
            else:
                remaining_contracts.append(contract)
        else:
            remaining_contracts.append(contract)

    return (
        _InferredMinMax(min_value=min_value, max_value=max_value),
        remaining_contracts,
    )


def _make_strategy_with_min_max_for_type(
    a_type: Type[T], inferred: _InferredMinMax
) -> hypothesis.strategies.SearchStrategy[T]:
    if a_type == int:
        # hypothesis.strategies.integers is always inclusive so we have to cut off the boundaries
        # a bit if they are exclusive.
        min_value = inferred.min_value
        if min_value is not None and not inferred.min_inclusive:
            min_value += 1

        max_value = inferred.max_value
        if max_value is not None and not inferred.max_inclusive:
            max_value -= 1

        strategy = hypothesis.strategies.integers(
            min_value=min_value,
            max_value=max_value,
        )  # type: hypothesis.strategies.SearchStrategy[Any]

    elif a_type == float:
        strategy = hypothesis.strategies.floats(
            min_value=inferred.min_value,
            max_value=inferred.max_value,
            exclude_min=inferred.min_value is not None and not inferred.min_inclusive,
            exclude_max=inferred.max_value is not None and not inferred.max_inclusive,
        )

    elif a_type == fractions.Fraction:
        # Fractions must also include their contracts as we can not bound them
        # by filtering! This needs to be a special case.
        strategy = hypothesis.strategies.fractions(
            min_value=inferred.min_value, max_value=inferred.max_value
        )

    elif a_type == decimal.Decimal:
        # Decimals must also include their contracts as we can not bound them
        # by filtering! This needs to be a special case.
        strategy = hypothesis.strategies.decimals(
            min_value=inferred.min_value, max_value=inferred.max_value
        )

    elif a_type == datetime.date:
        min_value = datetime.date.min
        if inferred.min_value is not None:
            if inferred.min_inclusive:
                min_value = inferred.min_value
            else:
                min_value = inferred.min_value + datetime.timedelta(days=1)

        max_value = datetime.date.max
        if inferred.max_value is not None:
            if inferred.max_inclusive:
                max_value = inferred.max_value
            else:
                max_value = inferred.max_value - datetime.timedelta(days=1)

        strategy = hypothesis.strategies.dates(min_value=min_value, max_value=max_value)

    elif a_type == datetime.datetime:
        min_value = datetime.datetime.min
        if inferred.min_value is not None:
            if inferred.min_inclusive:
                min_value = inferred.min_value
            else:
                min_value = inferred.min_value + datetime.timedelta(microseconds=1)

        max_value = datetime.datetime.max
        if inferred.max_value is not None:
            if inferred.max_inclusive:
                max_value = inferred.max_value
            else:
                max_value = inferred.max_value - datetime.timedelta(microseconds=1)

        strategy = hypothesis.strategies.datetimes(
            min_value=min_value, max_value=max_value
        )

    elif a_type == datetime.time:
        min_value = datetime.time.min
        if inferred.min_value is not None:
            min_value = inferred.min_value

            if not inferred.min_inclusive:
                if min_value == datetime.time.max:
                    raise ValueError(
                        f"The inferred exclusive lower bound for the time is equal "
                        f"datetime.time.max ({datetime.time.max}) "
                        f"so we can not compute the next greater time."
                    )

                min_value = (
                    datetime.datetime.combine(
                        datetime.date(2000, 1, 1), min_value, min_value.tzinfo
                    )
                    + datetime.timedelta(microseconds=1)
                ).time()

        max_value = datetime.time.max
        if inferred.max_value is not None:
            max_value = inferred.max_value

            if not inferred.max_inclusive:
                if max_value == datetime.time.min:
                    raise ValueError(
                        f"The inferred exclusive upper bound for the time is equal "
                        f"datetime.time.min ({datetime.time.min}) "
                        f"so we can not compute the previous less-than time."
                    )

                max_value = (
                    datetime.datetime.combine(
                        datetime.date(2000, 1, 1), max_value, max_value.tzinfo
                    )
                    - datetime.timedelta(microseconds=1)
                ).time()

        strategy = hypothesis.strategies.times(min_value=min_value, max_value=max_value)

    elif a_type == datetime.timedelta:
        min_value = datetime.timedelta.min
        if inferred.min_value is not None:
            if inferred.min_inclusive:
                min_value = inferred.min_value
            else:
                min_value = inferred.min_value + datetime.timedelta(microseconds=1)

        max_value = datetime.timedelta.max
        if inferred.max_value is not None:
            if inferred.max_inclusive:
                max_value = inferred.max_value
            else:
                max_value = inferred.max_value - datetime.timedelta(microseconds=1)

        strategy = hypothesis.strategies.timedeltas(
            min_value=min_value, max_value=max_value
        )

    else:
        raise AssertionError("Unexpected type hint: {}".format(a_type))

    return strategy


# We need to compile a dummy pattern so that we can compare addresses of re.Pattern.match functions.
_DUMMY_RE = re.compile(r"something")


def _infer_regexp_from_condition(
    arg_name: str, condition: Callable[..., Any]
) -> Optional[Pattern[AnyStr]]:
    """Try to infer the regular expression pattern from a precondition."""
    body_node = _body_node_from_condition(condition=condition)
    if body_node is None:
        return None

    if not isinstance(body_node, ast.Call):
        return None

    if not _no_name_in_descendants(root=body_node.func, name=arg_name):
        return None

    if not isinstance(body_node.func, ast.Attribute):
        return None

    if body_node.func.attr != "match":
        return None

    callee, recomputed = _recompute(condition=condition, node=body_node.func.value)
    if not recomputed:
        return None

    if callee == re:
        # Match "re.match(r'Some pattern', s, *args, *kwargs)
        if (
            len(body_node.args) >= 2
            and _no_name_in_descendants(root=body_node.args[0], name=arg_name)
            and isinstance(body_node.args[1], ast.Name)
            and body_node.args[1].id == arg_name
            and not any(
                _no_name_in_descendants(root=arg, name=arg_name)
                for arg in body_node.args[2:]
            )
        ):
            # Recompute the pattern
            args = []  # type: List[Any]
            for arg in [body_node.args[0]] + body_node.args[2:]:
                value, recomputed = _recompute(condition=condition, node=arg)
                if not recomputed:
                    return None

                args.append(value)

            kwargs = dict()  # type: Dict[str, Any]
            for keyword in body_node.keywords:
                value, recomputed = _recompute(condition=condition, node=keyword.value)
                if not recomputed:
                    return None

                assert (
                    keyword.arg is not None
                ), "Unexpected missing arg for a keyword: {}".format(ast.dump(keyword))
                kwargs[keyword.arg] = value

            pattern = re.compile(*args, **kwargs)
            return pattern

    elif isinstance(callee, Pattern):
        return callee

    return None


@icontract.require(
    lambda arg_name, contracts: all(
        len(contract.condition_args) == 1 and contract.condition_args[0] == arg_name
        for contract in contracts
    ),
    "All contracts are single-argument contracts related to this argument.",
)
def _infer_str_strategy_from_preconditions(
    arg_name: str, contracts: List[icontract._types.Contract]
) -> Tuple[
    Optional[hypothesis.strategies.SearchStrategy[AnyStr]],
    List[icontract._types.Contract],
]:
    """
    Try to match code patterns on AST of the preconditions contracts and infer the string strategy.

    Return (strategy if possible, remaining contracts).
    """
    found_idx = -1  # Index of the contract that defines the pattern, -1 if not found
    re_pattern = None  # type: Optional[Pattern[AnyStr]]

    for i, contract in enumerate(contracts):
        re_pattern = _infer_regexp_from_condition(
            arg_name=arg_name, condition=contract.condition
        )
        if re_pattern is not None:
            found_idx = i
            break

    if found_idx == -1:
        return None, contracts[:]

    assert re_pattern is not None
    return (
        hypothesis.strategies.from_regex(regex=re_pattern),
        contracts[:found_idx] + contracts[found_idx + 1 :],
    )


def _strategy_for_type(
    a_type: Type[T],
) -> hypothesis.strategies.SearchStrategy[T]:
    """Create a strategy for instances to satisfy the preconditions on ``__init__``."""
    init = getattr(a_type, "__init__")

    if inspect.isfunction(init):
        strategy = infer_strategy(init)
    elif isinstance(init, icontract._checkers._SLOT_WRAPPER_TYPE):
        # We have to distinguish this special case which is used by named tuples and
        # possibly other optimized data structures.
        # In those cases, we have to infer the strategy based on __new__ instead of __init__.
        new = getattr(a_type, "__new__")
        assert (
            new is not None
        ), "Expected __new__ in {} if __init__ is a slot wrapper.".format(a_type)

        strategy = infer_strategy(new)
    else:
        raise AssertionError(
            "Expected __init__ to be either a function or a slot wrapper, but got: {}".format(
                type(init)
            )
        )

    pack_repr = f"lambda d: {a_type.__name__}(**d)"

    pack = lambda d: a_type(**d)  # type: ignore
    pack.__icontract_hypothesis_source_code__ = pack_repr  # type: ignore

    return strategy.map(pack=pack)


@icontract.require(
    lambda arg_name, contracts: contracts is None
    or all(
        len(contract.condition_args) == 1 and contract.condition_args[0] == arg_name
        for contract in contracts
    )
)
def _infer_strategy_for_argument(
    arg_name: str, type_hint: Any, contracts: Optional[List[icontract._types.Contract]]
) -> hypothesis.strategies.SearchStrategy[Any]:
    """Infer the initial strategy for the argument."""
    if contracts is None:
        return hypothesis.strategies.from_type(type_hint)

    strategy = None  # type: Optional[hypothesis.strategies.SearchStrategy[Any]]
    remaining_contracts = contracts

    if type_hint in [
        int,
        float,
        fractions.Fraction,
        decimal.Decimal,
        datetime.date,
        datetime.datetime,
        datetime.time,
        datetime.timedelta,
    ]:
        inferred, remaining_contracts = _infer_min_max_from_preconditions(
            arg_name=arg_name, type_hint=type_hint, contracts=contracts
        )

        if (
            inferred.min_value is not None
            and inferred.max_value is not None
            and inferred.min_value > inferred.max_value
        ):
            raise ValueError(
                (
                    "The min and max values inferred for the argument {} could not be satisfied: "
                    "inferred min is {}, inferred max is {}. Are your preconditions correct?"
                ).format(arg_name, inferred.min_value, inferred.max_value)
            )

        strategy = _make_strategy_with_min_max_for_type(
            a_type=type_hint, inferred=inferred
        )

    if strategy is None and type_hint == str:
        strategy, remaining_contracts = _infer_str_strategy_from_preconditions(
            arg_name=arg_name, contracts=contracts
        )

    if strategy is None:
        strategy = hypothesis.strategies.from_type(type_hint)
        remaining_contracts = contracts

    assert strategy is not None

    for contract in remaining_contracts:
        strategy = strategy.filter(contract.condition)

    return strategy


def _rewrite_condition_as_filter(
    contract: icontract._types.Contract,
) -> Callable[..., Any]:
    """
    Parses, rewrites and recompiles the condition so that it can be used as filter on kwargs.

    Return (rewritten condition as filter function, string representation of the condition)
    """
    if "_ARGS" in contract.condition_arg_set:
        raise NotImplementedError(
            "The handling of the special argument _ARGS is not currently handled "
            "by icontract-hypothesis. Please create a new issue on GitHub if you need "
            "this feature: https://github.com/mristin/icontract-hypothesis/issues"
        )

    lambda_inspection = icontract._represent.inspect_lambda_condition(
        condition=contract.condition
    )

    if lambda_inspection is None:
        condition_repr = "lambda d: {}({})".format(
            contract.condition.__name__,
            ", ".join(
                "_KWARGS=d"
                if arg_name == "_KWARGS"
                else "{0}=d['{0}']".format(arg_name)
                for arg_name in contract.condition_args
            ),
        )

        recompiled_code = compile(
            condition_repr,
            filename=f"<rewritten by icontract-hypothesis: {condition_repr}>",
            mode="eval",
        )  # type: types.CodeType

        lambda_code = recompiled_code.co_consts[0]

        rewired_function = types.FunctionType(
            lambda_code,
            {contract.condition.__name__: contract.condition},
            name="<lambda>",
            argdefs=contract.condition.__defaults__,  # type: ignore
            closure=contract.condition.__closure__,  # type: ignore
        )

        rewired_function.__icontract_hypothesis_source_code__ = condition_repr  # type: ignore
        return rewired_function

    name_set = set()  # type: Set[str]

    class FindNamesVisitor(ast.NodeVisitor):
        """Find all the Name nodes."""

        def visit_Name(  # pylint: disable=invalid-name,no-self-use,missing-docstring
            self, node: ast.Name
        ) -> None:
            name_set.add(node.id)

    FindNamesVisitor().visit(lambda_inspection.node)

    # We need to make sure that ``kwargs_name`` does not conflict with any global or closured
    # variable. However, it is OK if it conflicts with the local variables as they are going
    # to be replaced.
    kwargs_name = "d"
    while kwargs_name in name_set.difference(contract.condition_arg_set):
        kwargs_name = "_" + kwargs_name

    class Replace:
        """Instruct to replace the start and end of the source code with the given replacement."""

        def __init__(self, start: int, end: int, replacement: str) -> None:
            self.start = start
            self.end = end
            self.replacement = replacement

        def __repr__(self) -> str:
            return "Replace(start={}, end={}, replacement={!r})".format(
                self.start, self.end, self.replacement
            )

    replacements_in_text = []  # type: List[Replace]

    class ReplaceNameWithSubscriptTransformer(ast.NodeTransformer):
        """Replace condition arguments with kwargs subscript."""

        def visit_Name(  # pylint: disable=invalid-name,no-self-use,missing-docstring
            self, node: ast.Name
        ) -> ast.expr:
            nonlocal replacements_in_text

            if node.id not in contract.condition_arg_set:
                return node

            assert lambda_inspection is not None
            start, end = lambda_inspection.atok.get_text_range(node)

            if node.id == "_KWARGS":
                replacements_in_text.append(
                    Replace(start=start, end=end, replacement=kwargs_name)
                )
                return ast.Name(id=kwargs_name, ctx=ast.Load())

            replacements_in_text.append(
                Replace(
                    start=start,
                    end=end,
                    replacement="{}[{!r}]".format(kwargs_name, node.id),
                )
            )

            return ast.Subscript(
                value=ast.Name(id=kwargs_name, ctx=ast.Load()),
                slice=ast.Index(value=ast.Constant(value=node.id, kind=None)),
                ctx=ast.Load(),
            )

    lambda_node = ReplaceNameWithSubscriptTransformer().visit(lambda_inspection.node)

    lambda_node.args = ast.arguments(
        posonlyargs=[],
        args=[ast.arg(arg=kwargs_name, annotation=None, type_comment=None)],
        vararg=None,
        kwonlyargs=[],
        kw_defaults=[],
        kwarg=None,
        defaults=[],
    )

    expression_node = ast.Expression(body=lambda_node)

    class AddPositionVisitor(ast.NodeVisitor):
        """Add position to all the nodes if they do not have one already."""

        def generic_visit(self, node: Any) -> None:
            assert lambda_inspection is not None
            node.lineno = lambda_inspection.node.lineno
            node.col_offset = lambda_inspection.node.col_offset

            super().generic_visit(node)

    AddPositionVisitor().visit(expression_node)

    ##
    # Rewrite text
    ##

    if len(replacements_in_text) == 0:
        condition_repr = lambda_inspection.atok.get_text(lambda_inspection.node)
    else:
        # We have to use the text of the whole decorator, not just the lambda since ``atok``
        # refers to the decorator.
        text = lambda_inspection.atok.text

        body_start, body_end = lambda_inspection.atok.get_text_range(
            lambda_inspection.node.body
        )

        parts = []  # type: List[str]

        previous_replace = None  # type: Optional[Replace]
        for replace in replacements_in_text:
            if previous_replace is None:
                parts.append(text[body_start : replace.start])
            else:
                parts.append(text[previous_replace.end : replace.start])

            parts.append(replace.replacement)
            previous_replace = replace

        assert previous_replace is not None
        parts.append(text[previous_replace.end : body_end])
        condition_repr = "lambda {}: {}".format(kwargs_name, "".join(parts))

    ##
    # Recompile the code
    ##

    recompiled_code = compile(
        expression_node,
        filename=f"<rewritten by icontract-hypothesis: {condition_repr}>",
        mode="eval",
    )

    ##
    # Patch the recompiled code such that it uses globals and freevars (closures) from the
    # original condition function
    ##

    # We need to patch all the closured vars in the recompiled code
    # to LOAD_DEREF instead of LOAD_GLOBAL
    closured_vars = {
        name: i for i, name in enumerate(contract.condition.__code__.co_freevars)
    }
    if len(closured_vars) > 256:
        raise NotImplementedError(
            "icontract-hypothesis does not handle condition lambdas with more than 256 closures. "
            "Please create an issue to let us know to implement it: "
            "https://github.com/mristin/icontract-hypothesis/issues"
        )

    # The indices of global vars need to change to correspond to the original condition function.
    global_vars = {
        name: i for i, name in enumerate(contract.condition.__code__.co_names)
    }

    # The first const contains the actual code of the lambda function.
    lambda_code = recompiled_code.co_consts[0]

    opcode_load_global = dis.opmap["LOAD_GLOBAL"]
    opcode_load_deref = dis.opmap["LOAD_DEREF"]
    opcode_extended_arg = dis.opmap["EXTENDED_ARG"]

    bytecode = lambda_code.co_code
    new_bytecode = io.BytesIO()

    # The last instruction must be RETURN_VALUE.
    assert (
        bytecode[-2:]
        == dis.opmap["RETURN_VALUE"].to_bytes(1, byteorder="little") + b"\x00"
    )

    # Since we know that the last instruction is RETURN_VALUE,
    # we can always look up the next opcode.
    for i in range(0, len(bytecode) - 2, 2):
        opcode = bytecode[i]
        oparg = bytecode[i + 1]

        next_opcode = bytecode[i + 2]

        if opcode == opcode_extended_arg and next_opcode == opcode_load_global:
            raise NotImplementedError(
                "icontract-hypothesis does not handle lambdas with more than 256 globals. "
                "Please create an issue to let us know to implement it: "
                "https://github.com/mristin/icontract-hypothesis/issues"
            )

        if opcode == opcode_load_global:
            name = lambda_code.co_names[oparg]
            if name in closured_vars:
                new_bytecode.write(
                    opcode_load_deref.to_bytes(1, byteorder="little")
                    + closured_vars[name].to_bytes(1, byteorder="little")
                )
            elif name in global_vars:
                new_bytecode.write(
                    opcode_load_global.to_bytes(1, byteorder="little")
                    + global_vars[name].to_bytes(1, byteorder="little")
                )
            else:
                raise AssertionError(
                    f"Unexpected case where the opcode was {opcode} ({(dis.opmap[opcode])!r}"
                    f"and the name was {name!r}, but the original closures were {closured_vars!r} "
                    f"and the original globals were {global_vars!r}, respectively."
                )
        else:
            new_bytecode.write(bytecode[i : i + 2])

    new_bytecode.write(bytecode[-2:])

    if sys.version_info < (3, 8):
        new_lambda_code = types.CodeType(
            lambda_code.co_argcount,
            lambda_code.co_kwonlyargcount,
            lambda_code.co_nlocals,
            lambda_code.co_stacksize,
            lambda_code.co_flags,
            new_bytecode.getvalue(),
            lambda_code.co_consts,
            contract.condition.__code__.co_names,
            lambda_code.co_varnames,
            "<recompiled by icontract-hypothesis>",
            "<lambda>",
            lambda_code.co_firstlineno,
            lambda_code.co_lnotab,
            contract.condition.__code__.co_freevars,
            lambda_code.co_cellvars,
        )
    else:
        assert isinstance(lambda_code, types.CodeType)
        new_lambda_code = lambda_code.replace(
            co_code=new_bytecode.getvalue(),
            co_names=contract.condition.__code__.co_names,
            co_filename="<recompiled by icontract-hypothesis>",
            co_name="<lambda>",
            co_freevars=contract.condition.__code__.co_freevars,
        )

    recompiled_function = eval(  # pylint: disable=eval-used
        recompiled_code, dict(globals())
    )

    ##
    # Re-wire the globals and the closure
    ##

    rewired_function = types.FunctionType(
        new_lambda_code,
        contract.condition.__globals__,  # type: ignore
        name="<lambda>",
        argdefs=recompiled_function.__defaults__,
        closure=contract.condition.__closure__,  # type: ignore
    )

    rewired_function.__icontract_hypothesis_source_code__ = condition_repr  # type: ignore

    return rewired_function


def _infer_strategy_from_conjunction(
    type_hints: Mapping[str, Any], conjunction: List[icontract._types.Contract]
) -> hypothesis.strategies.SearchStrategy[Any]:
    """Infer the strategy that satisfies the given conjunction of contracts."""
    ##
    # Group single-argument contracts by the argument,
    # keep the list of the zero or multi-argument contracts
    ##

    single_argument_contracts = (
        dict()
    )  # type: MutableMapping[str, List[icontract._types.Contract]]

    non_single_argument_contracts = []  # type: List[icontract._types.Contract]

    for contract in conjunction:
        if len(contract.condition_args) == 1 and contract.condition_args[0] not in (
            "_ARGS",
            "_KWARGS",
        ):
            arg_name = contract.condition_args[0]

            if arg_name not in single_argument_contracts:
                single_argument_contracts[arg_name] = [contract]
            else:
                single_argument_contracts[arg_name].append(contract)
        else:
            non_single_argument_contracts.append(contract)

    strategy = hypothesis.strategies.fixed_dictionaries(
        {
            arg_name: _infer_strategy_for_argument(
                arg_name=arg_name,
                type_hint=type_hint,
                contracts=single_argument_contracts.get(arg_name, None),
            )
            for arg_name, type_hint in type_hints.items()
        }
    )

    for contract in non_single_argument_contracts:
        condition_as_filter = _rewrite_condition_as_filter(contract=contract)

        # Assert these attributes here as we depend on monkey patching the Hypothesis
        # reflection function
        assert hasattr(condition_as_filter, "__icontract_hypothesis_source_code__")
        assert condition_as_filter.__name__ == "<lambda>", condition_as_filter.__name__

        strategy = strategy.filter(condition_as_filter)

    return strategy


def _create_strategy_only_from_type_hints(
    type_hints: Mapping[str, Any]
) -> hypothesis.strategies.SearchStrategy[Any]:
    """
    Create a strategy based only on type hints.

    For example, this is the strategy that you can infer if there are no preconditions.
    """
    return hypothesis.strategies.fixed_dictionaries(
        {
            arg_name: hypothesis.strategies.from_type(arg_type)
            for arg_name, arg_type in type_hints.items()
        }
    )


def infer_strategy(
    func: CallableT,
) -> hypothesis.strategies.SearchStrategy[Any]:
    r"""
    Infer the search strategy of the arguments for the given function.

    Apart from the internal usage, this function is mainly meant for manual inspection
    of the inferred strategies.

    Here is an example how you can debug what strategies will be used to test ``some_func``:

    >>> import icontract
    >>> import icontract_hypothesis

    >>> @icontract.require(lambda x: x > 0)
    ... def some_func(x: int) -> None:
    ...    ...

    >>> strategy = icontract_hypothesis.infer_strategy(some_func)
    >>> str(strategy)
    "fixed_dictionaries({'x': integers(min_value=1)})"
    """
    parameters = inspect.signature(func).parameters
    if len(parameters) == 0:
        return hypothesis.strategies.just(dict())

    for name, parameter in parameters.items():
        if parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise NotImplementedError(
                "Inferring strategy of a function with positional-only arguments is not "
                "currently implemented in icontract-hypothesis. "
                "Please create an issue on GitHub if you need this feature: "
                "https://github.com/mristin/icontract-hypothesis/issues"
            )
        else:
            pass

    type_hints = typing.get_type_hints(func)
    if "return" in type_hints:
        del type_hints["return"]

    typed_args = set(type_hints)
    parameter_set = set(parameters.keys())
    parameter_set.difference_update({"self"})

    for name, parameter in parameters.items():
        if parameter.kind in (
            inspect.Parameter.VAR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        ):
            # Ignore variable keyword and positional arguments in further analysis
            parameter_set.remove(name)

            if name in type_hints:
                typed_args.remove(name)
                del type_hints[name]

    if typed_args != parameter_set:
        raise TypeError(
            (
                "No search strategy could be inferred for the function: {}; "
                "the following arguments are missing the type annotations: {}"
            ).format(func, list(parameter_set.difference(typed_args)))
        )

    checker = icontract._checkers.find_checker(func)

    if checker is None:
        return _create_strategy_only_from_type_hints(type_hints=type_hints)

    maybe_preconditions = getattr(checker, "__preconditions__", None)

    preconditions = None  # type: Optional[List[List[icontract._types.Contract]]]
    if maybe_preconditions is not None:
        assert isinstance(maybe_preconditions, list)
        assert all(isinstance(conjunction, list) for conjunction in maybe_preconditions)
        assert all(
            isinstance(contract, icontract._types.Contract)
            for conjunction in maybe_preconditions
            for contract in conjunction
        )

        preconditions = cast(List[List[icontract._types.Contract]], maybe_preconditions)

    if preconditions is None or len(preconditions) == 0:
        return _create_strategy_only_from_type_hints(type_hints=type_hints)

    strategies = [
        _infer_strategy_from_conjunction(type_hints=type_hints, conjunction=conjunction)
        for conjunction in preconditions
    ]

    if len(strategies) == 1:
        return strategies[0]

    return hypothesis.strategies.one_of(strategies)


def test_with_inferred_strategy(func: CallableT) -> None:
    r"""
    Use type hints and contracts to infer the search strategy and test the function.

    Here is an example test case which tests a function ``some_func``:

    >>> import unittest
    >>> import icontract
    >>> import icontract_hypothesis

    >>> @icontract.require(lambda x: x > 0)
    ... @icontract.require(lambda x: x < 100)
    ... @icontract.require(lambda y: 0 < y < 100)
    ... def some_func(x: int, y: int) -> None:
    ...     ...

    >>> class TestSomething(unittest.TestCase):
    ...     def test_that_it_works(self) -> None:
    ...         icontract_hypothesis.test_with_inferred_strategy(some_func)

    >>> TestSomething().test_that_it_works()
    """
    pass  # for pydocstyle

    def execute(kwargs: Dict[str, Any]) -> None:
        func(**kwargs)

    strategy = infer_strategy(func=func)
    wrapped = hypothesis.given(strategy)(execute)
    wrapped()


def _register_with_hypothesis(cls: Type[T]) -> None:
    """
    Register ``cls`` with Hypothesis based on our custom ``_strategy_for_type``.

    The registration is necessary so that the preconditions on the __init__ are propagated
    in ``hypothesis.strategies.builds``.
    """
    # We should not register abstract classes as this will mislead Hypothesis to instantiate
    # them.
    if inspect.isabstract(cls):
        return

    if cls not in hypothesis.strategies._internal.types._global_type_lookup:
        hypothesis.strategies.register_type_strategy(cls, _strategy_for_type(cls))


def _hook_into_icontract_and_hypothesis() -> None:
    """
    Redirect ``icontract._metaclass._register_for_hypothesis``.

    All the classes previously registered by icontract are now re-registered
    by ``_register_with_hypothesis``.
    """
    if not hasattr(icontract._metaclass, "_CONTRACT_CLASSES"):
        return  # already hooked in

    icontract._metaclass._register_for_hypothesis = _register_with_hypothesis

    for cls in icontract._metaclass._CONTRACT_CLASSES:
        _register_with_hypothesis(cls)

    # Delete in order to fail fast
    del icontract._metaclass._CONTRACT_CLASSES

    # Monkey-patch lambda source so that we do not have to introduce
    # strategy classes just for this functionality.
    # See https://github.com/HypothesisWorks/hypothesis/issues/2713
    upstream_extract_lambda_source = (
        hypothesis.internal.reflection.extract_lambda_source
    )
    hypothesis.internal.reflection.extract_lambda_source = lambda f: (
        getattr(f, "__icontract_hypothesis_source_code__", None)
        or upstream_extract_lambda_source(f)  # type: ignore
    )


_hook_into_icontract_and_hypothesis()
