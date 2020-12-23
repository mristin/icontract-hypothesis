"""Automatically test a Python file."""
import argparse
import collections
import datetime
import json
import pathlib
import re
import textwrap
from typing import Mapping, Any, Tuple, Optional, List, MutableMapping, Dict, TextIO

import hypothesis

import icontract_hypothesis
import icontract_hypothesis.pyicontract_hypothesis._general as _general


class Params:
    """Represent parameters of the command "test"."""

    def __init__(
        self, path: pathlib.Path, settings: Mapping[str, Any], inspect: bool
    ) -> None:
        """Initialize with the given values."""
        self.path = path
        self.settings = settings
        self.inspect = inspect


_SETTING_STATEMENT_RE = re.compile(
    r"^(?P<identifier>[a-zA-Z_][a-zA-Z_0-9]*)\s*=\s*(?P<value>.*)\s*$"
)


def parse_params(args: argparse.Namespace) -> Tuple[Optional[Params], List[str]]:
    """
    Try to parse the parameters of the command "test".

    Return (parsed parameters, errors if any).
    """
    errors = []  # type: List[str]

    path = pathlib.Path(args.path)

    settings = collections.OrderedDict()  # type: MutableMapping[str, Any]

    if args.settings is not None:
        for i, statement in enumerate(args.settings):
            mtch = _SETTING_STATEMENT_RE.match(statement)
            if not mtch:
                errors.append(
                    "Invalid setting statement {}. Expected statement to match {}, "
                    "but got: {}".format(
                        i + 1, _SETTING_STATEMENT_RE.pattern, statement
                    )
                )

                return None, errors

            identifier = mtch.group("identifier")
            value_str = mtch.group("value")

            try:
                value = json.loads(value_str)
            except json.decoder.JSONDecodeError as error:
                errors.append(
                    "Failed to parse the value of the setting {}: {}".format(
                        identifier, error
                    )
                )
                return None, errors

            settings[identifier] = value

    if errors:
        return None, errors

    return Params(path=path, settings=settings, inspect=args.inspect), errors


def _test_function_point(
    point: _general.FunctionPoint, settings: Optional[Mapping[str, Any]]
) -> List[str]:
    """
    Test a single function point.

    Return errors if any.
    """
    errors = []  # type: List[str]

    func = point.func  # Optimize the look-up

    try:
        strategy = icontract_hypothesis.infer_strategy(func=func)
    except Exception as error:
        errors.append(
            (
                "Inference of the search strategy failed for the function: {}. "
                "The error was: {}"
            ).format(func, error)
        )
        return errors

    def execute(kwargs: Dict[str, Any]) -> None:
        func(**kwargs)

    wrapped = hypothesis.given(strategy)(execute)
    if settings:
        wrapped = hypothesis.settings(**settings)(wrapped)

    wrapped()

    return []


def _inspect_test_function_point(
    point: _general.FunctionPoint, settings: Optional[Mapping[str, Any]]
) -> Tuple[str, List[str]]:
    """
    Inspect what test will executed to test a single function point.

    Return (inspection, errors if any).
    """
    errors = []  # type: List[str]

    func = point.func  # Optimize the look-up

    try:
        strategy = icontract_hypothesis.infer_strategy(func=func)
    except Exception as error:
        errors.append(
            (
                "Inference of the search strategy failed for the function: {}. "
                "The error was: {}"
            ).format(func, error)
        )
        return "", errors

    parts = [
        textwrap.dedent(
            """\
        hypothesis.given(
        {}
        )"""
        ).format(textwrap.indent(str(strategy), "    "))
    ]

    if settings:
        assert len(settings) != 0

        if len(settings) == 1:
            items_lst = list(settings.items())
            # fmt: off
            settings_str = ''.join(
                ['hypothesis.settings({'] +
                ['{!r}: {!r}'.format(items_lst[0][0], items_lst[0][1])] +
                ['})']
            )
            # fmt: on
        else:
            settings_parts = ["hypothesis.settings({"]
            for i, (key, value) in enumerate(settings.items()):
                if i < len(settings) - 1:
                    settings_parts.append(f"    {key!r}: {value!r},")
                else:
                    settings_parts.append(f"    {key!r}: {value!r}")
            settings_parts.append("})")
            settings_str = "\n".join(settings_parts)

        parts.append(settings_str)

    return "\n".join(parts), []


def test(general: _general.Params, command: Params, stdout: TextIO) -> List[str]:
    """
    Test the specified functions.

    Return errors if any.
    """
    if not command.path.exists():
        return ["The file to be tested does not exist: {}".format(command.path)]

    try:
        source_code = command.path.read_text(encoding="utf-8")
    except Exception as error:
        return ["Failed to read the file {}: {}".format(command.path, error)]

    mod, errors = _general.load_module_from_source_file(path=command.path)
    if errors:
        return errors

    assert mod is not None

    points, errors = _general.select_function_points(
        source_code=source_code,
        mod=mod,
        include=general.include,
        exclude=general.exclude,
    )
    if errors:
        return errors

    if command.inspect:
        printed_previously = False
        for point in points:
            inspection, test_errors = _inspect_test_function_point(
                point=point, settings=command.settings
            )
            errors.extend(test_errors)

            if not test_errors:
                if printed_previously:
                    stdout.write("\n")

                stdout.write(f"{point.func.__name__} at line {point.first_row}:\n")
                stdout.write(textwrap.indent(inspection, "   "))
                stdout.write("\n")
                printed_previously = True
    else:
        for point in points:
            start = datetime.datetime.now()
            test_errors = _test_function_point(point=point, settings=command.settings)
            errors.extend(test_errors)

            duration = datetime.datetime.now() - start

            if not test_errors:
                stdout.write(
                    f"Tested {point.func.__name__} at line {point.first_row} "
                    f"(time delta {duration}).\n"
                )

    if errors:
        return errors

    return []
