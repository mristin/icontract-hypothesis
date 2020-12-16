"""Provide functionality for ghostwriting test modules."""
import argparse
import enum
import inspect
import pathlib
import textwrap
from typing import Optional, Tuple, List, overload, Union, Callable, Any

import hypothesis.strategies
import icontract._checkers
import icontract._represent

import icontract_hypothesis
import icontract_hypothesis.pyicontract_hypothesis._general as _general
from icontract_hypothesis.pyicontract_hypothesis import exhaustive


class Explicit(enum.Enum):
    """Specify how explicit the ghostwriter should be."""

    STRATEGIES = "strategies"
    STRATEGIES_AND_ASSUMES = "strategies-and-assumes"


class Params:
    """Represent parameters of the command "ghostwrite"."""

    def __init__(
        self,
        module_name: str,
        output: Optional[pathlib.Path],
        explicit: Optional[Explicit],
        bare: bool,
    ) -> None:
        """Initialize with the given values."""
        self.module_name = module_name
        self.output = output
        self.explicit = explicit
        self.bare = bare


def parse_params(args: argparse.Namespace) -> Tuple[Optional[Params], List[str]]:
    """
    Try to parse the parameters of the command "ghostwrite".

    Return (parsed parameters, errors if any).
    """
    output = pathlib.Path(args.output) if args.output != "-" else None

    return (
        Params(
            module_name=args.module,
            output=output,
            explicit=Explicit(args.explicit) if args.explicit is not None else None,
            bare=args.bare,
        ),
        [],
    )


@overload
def _indent_but_first(
    lines: List[str], level: int = 1
) -> List[str]:  # pylint: disable=all
    ...


@overload
def _indent_but_first(lines: str, level: int = 1) -> str:  # pylint: disable=all
    ...


def _indent_but_first(
    lines: Union[List[str], str], level: int = 1
) -> Union[str, List[str]]:
    r"""
    Indents the text by 4 spaces.

    >>> _indent_but_first([''], 0)
    ['']

    >>> _indent_but_first(['test'], 1)
    ['test']

    >>> _indent_but_first(['test', '', 'me'], 1)
    ['test', '', '    me']

    >>> _indent_but_first('test\n\nme', 1)
    'test\n\n    me'
    """
    if isinstance(lines, str):
        result = lines.splitlines()
        for i in range(1, len(result)):
            if len(result[i]) > 0:
                result[i] = "    " * level + result[i]

        return "\n".join(result)

    elif isinstance(lines, list):
        result = lines[:]
        for i in range(1, len(result)):
            if len(result[i]) > 0:
                result[i] = "    " * level + result[i]

        return result

    else:
        raise AssertionError("Unhandled input: {}".format(lines))


def _ghostwrite_condition_code(condition: Callable[..., Any]) -> str:
    """Ghostwrite the code representing the condition in an assumption."""
    if not icontract._represent.is_lambda(a_function=condition):
        sign = inspect.signature(condition)
        args = ", ".join(param for param in sign.parameters.keys())

        return "{}({})".format(condition.__name__, args)

    # We need to extract the source code corresponding to the decorator since
    # inspect.getsource() is broken with lambdas.

    # Find the line corresponding to the condition lambda
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
    ), "Expected lambda_inspection to be non-None if is_lambda is True on: {}".format(
        condition
    )

    return lambda_inspection.text


@icontract.ensure(
    lambda result: not result.endswith("\n")
    and not result.startswith(" ")
    and not result.startswith("\t"),
    "Not indented and no newline at the end",
)
def _ghostwrite_assumes(func: Callable[..., Any]) -> str:
    """Ghostwrite the assume statements for the given function."""
    checker = icontract._checkers.find_checker(func)
    if checker is None:
        return ""

    preconditions = getattr(checker, "__preconditions__", None)
    if preconditions is None:
        return ""

    # We need to pack all the preconditions in a large boolean expression so that
    # the weakening can be handled easily.

    dnf = []  # type: List[List[str]]
    for group in preconditions:
        conjunctions = []  # type: List[str]
        for contract in group:
            code = _ghostwrite_condition_code(condition=contract.condition)
            conjunctions.append(code)

        dnf.append(conjunctions)

    if len(dnf) == 0:
        return ""

    if len(dnf) == 1:
        if len(dnf[0]) == 1:
            return "assume({})".format(dnf[0][0])
        else:
            formatted_conjunctions = textwrap.dedent(
                """\
                assume(
                    {}
                )"""
            ).format(
                _indent_but_first(
                    " and\n".join("({})".format(code) for code in dnf[0]), level=1
                )
            )

            return formatted_conjunctions

    dnf_formatted = []  # type: List[str]
    for conjunctions in dnf:
        if len(conjunctions) == 1:
            dnf_formatted.append("({})".format(conjunctions[0]))
        else:
            dnf_formatted.append(
                textwrap.dedent(
                    """\
                (
                    {}
                )"""
                ).format(
                    _indent_but_first(
                        " and\n".join("({})".format(code) for code in conjunctions),
                        level=1,
                    )
                )
            )

    return textwrap.dedent(
        """\
        assume(
            {}
        )"""
    ).format(_indent_but_first(" or \n".join(dnf_formatted)))


def _ghostwrite_test_function(
    module_name: str, point: _general.FunctionPoint, explicit: Optional[Explicit]
) -> Tuple[str, List[str]]:
    """
    Ghostwrite a test function for the given function point.

    The result needs to be properly indented afterwards.

    Return (code, errors if any)
    """
    test_func = ""

    if explicit is None:
        test_func = textwrap.dedent(
            """\
            def test_{1}(self) -> None:
                icontract.integration.with_hypothesis.test_with_inferred_strategies(
                        func={0}.{1})
            """.format(
                module_name, point.func.__name__
            )
        ).strip()

    elif explicit is Explicit.STRATEGIES or explicit is Explicit.STRATEGIES_AND_ASSUMES:
        strategies = icontract_hypothesis.infer_strategies(func=point.func)

        if len(strategies) == 0:
            return "", [
                "No strategy could be inferred for the function on line {}: {}".format(
                    point.first_row, point.func.__name__
                )
            ]

        args = ", ".join(strategies.keys())

        given_args_lines = []  # type: List[str]
        for i, (arg_name, strategy) in enumerate(strategies.items()):
            strategy_code = str(strategy)
            for name in hypothesis.strategies.__all__:
                prefix = "{}(".format(name)
                if strategy_code.startswith(prefix):
                    strategy_code = "st." + strategy_code
                    break

            if i < len(strategies) - 1:
                given_args_lines.append("{}={},".format(arg_name, strategy_code))
            else:
                given_args_lines.append("{}={}".format(arg_name, strategy_code))

        if explicit is Explicit.STRATEGIES:
            test_func = (
                textwrap.dedent(
                    """\
                def test_{func}(self) -> None:
                    assume_preconditions = icontract_hypothesis.make_assume_preconditions(
                        func={module}.{func})

                    @given(
                        {given_args})
                    def execute({args}) -> None:
                        assume_preconditions({args})
                        {module}.{func}({args})
                """
                )
                .format(
                    module=module_name,
                    func=point.func.__name__,
                    args=args,
                    given_args="\n".join(_indent_but_first(given_args_lines, 2)),
                )
                .strip()
            )

        elif explicit is Explicit.STRATEGIES_AND_ASSUMES:
            assume_statements = _ghostwrite_assumes(func=point.func)

            if assume_statements == "":
                test_func = (
                    textwrap.dedent(
                        """\
                    def test_{func}(self) -> None:
                        @given(
                            {given_args})
                        def execute({args}) -> None:
                            {module}.{func}({args})
                    """
                    )
                    .format(
                        module=module_name,
                        func=point.func.__name__,
                        args=args,
                        given_args="\n".join(_indent_but_first(given_args_lines, 2)),
                    )
                    .strip()
                )
            else:
                test_func = (
                    textwrap.dedent(
                        """\
                    def test_{func}(self) -> None:
                        @given(
                            {given_args})
                        def execute({args}) -> None:
                            {assumes}
                            {module}.{func}({args})
                    """
                    )
                    .format(
                        module=module_name,
                        func=point.func.__name__,
                        args=args,
                        assumes=_indent_but_first(assume_statements, level=2),
                        given_args="\n".join(_indent_but_first(given_args_lines, 2)),
                    )
                    .strip()
                )

        else:
            exhaustive.assert_never(explicit)
    else:
        exhaustive.assert_never(explicit)

    return test_func, []


def _ghostwrite_for_function_points(
    points: List[_general.FunctionPoint],
    module_name: str,
    explicit: Optional[Explicit],
    bare: bool,
) -> Tuple[str, List[str]]:
    """
    Ghostwrite a test case for the given function points.

    Return (generated code, errors if any).
    """
    errors = []  # type: List[str]
    test_funcs = []  # type: List[str]
    for point in points:
        test_func, test_func_errors = _ghostwrite_test_function(
            module_name=module_name, point=point, explicit=explicit
        )
        if test_func_errors:
            errors.extend(test_func_errors)
        else:
            test_funcs.append(test_func)

    if errors:
        return "", errors

    if bare:
        return "\n\n".join(test_funcs), []

    blocks = []  # type: List[str]

    header = "\n\n".join(
        [
            '"""Test {} with inferred Hypothesis strategies."""'.format(module_name),
            "import unittest",
            (
                (
                    "import hypothesis.strategies as st\n"
                    "from hypothesis import assume, given\n"
                )
                if explicit
                else ""
            )
            + "import icontract_hypothesis",
            "import {}".format(module_name),
        ]
    )
    blocks.append(header)

    if len(points) == 0:
        blocks.append(
            textwrap.dedent(
                '''\
                class TestWithInferredStrategies(unittest.TestCase):
                    """Test all functions from {0} with inferred Hypothesis strategies."""
                    # Either there are no functions in {0} or all the functions were excluded.
                '''.format(
                    module_name
                )
            ).strip()
        )
    else:
        body = "\n\n".join(test_funcs)

        test_case = [
            textwrap.dedent(
                '''\
                    class TestWithInferredStrategies(unittest.TestCase):
                        """Test all functions from {module} with inferred Hypothesis strategies."""

                        {body}
                    '''
            )
            .format(
                module=module_name,
                body="\n".join(_indent_but_first(lines=body.splitlines(), level=1)),
            )
            .strip()
        ]
        blocks.append("".join(test_case))

    blocks.append(
        textwrap.dedent(
            """\
        if __name__ == '__main__':
            unittest.main()
        """
        )
    )

    return "\n\n\n".join(blocks), []


def ghostwrite(general: _general.Params, command: Params) -> Tuple[str, List[str]]:
    """
    Write a unit test module for the specified functions.

    Return (generated code, errors if any).
    """
    mod, errors = _general.load_module_with_name(command.module_name)
    if errors:
        return "", errors

    assert mod is not None

    points, errors = _general.select_function_points(
        source_code=inspect.getsource(mod),
        mod=mod,
        include=general.include,
        exclude=general.exclude,
    )
    if errors:
        return "", errors

    return _ghostwrite_for_function_points(
        points=points,
        module_name=command.module_name,
        explicit=command.explicit,
        bare=command.bare,
    )
