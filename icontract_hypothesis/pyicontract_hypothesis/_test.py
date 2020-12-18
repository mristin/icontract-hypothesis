"""Automatically test a Python file."""
import argparse
import collections
import json
import pathlib
import re
from typing import Mapping, Any, Tuple, Optional, List, MutableMapping, Dict

import hypothesis

import icontract_hypothesis
import icontract_hypothesis.pyicontract_hypothesis._general as _general


class Params:
    """Represent parameters of the command "test"."""

    def __init__(self, path: pathlib.Path, settings: Mapping[str, Any]) -> None:
        """Initialize with the given values."""
        self.path = path
        self.settings = settings


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

    return Params(path=path, settings=settings), errors


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


def test(general: _general.Params, command: Params) -> List[str]:
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

    for point in points:
        test_errors = _test_function_point(point=point, settings=command.settings)
        errors.extend(test_errors)

        if errors:
            return errors

    return []
