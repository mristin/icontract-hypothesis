"""Automatically test a Python file."""
import argparse
import collections
import datetime
import inspect
import json
import pathlib
import re
import textwrap
from typing import (
    Mapping,
    Any,
    Tuple,
    Optional,
    List,
    MutableMapping,
    Dict,
    TextIO,
)

import hypothesis
import hypothesis._settings
import hypothesis.errors

import icontract_hypothesis
import icontract_hypothesis.pyicontract_hypothesis._general as _general


class Params:
    """Represent parameters of the command "test"."""

    def __init__(
        self,
        path: pathlib.Path,
        settings_parsing: Optional["SettingsParsing"],
        do_inspect: bool,
    ) -> None:
        """Initialize with the given values."""
        self.path = path
        self.settings_parsing = settings_parsing
        self.inspect = do_inspect


_SETTING_STATEMENT_RE = re.compile(
    r"^(?P<identifier>[a-zA-Z_][a-zA-Z_0-9]*)\s*=\s*(?P<value>.*)\s*$"
)

_HYPOTHESIS_SETTINGS_PARAMETERS = inspect.signature(
    hypothesis.settings.__init__
).parameters
_HYPOTHESIS_HEALTH_CHECK_MAP = {
    health_check.name: health_check for health_check in hypothesis.HealthCheck
}
_HYPOTHESIS_VERBOSITY_MAP = {
    verbosity.name: verbosity for verbosity in hypothesis.Verbosity
}
_HYPOTHESIS_PHASE_MAP = {phase.name: phase for phase in hypothesis.Phase}


class SettingsParsing:
    """Represent hypothesis.settings such that it can be inspected down the line."""

    def __init__(self, parsed_kwargs: Mapping[str, Any], product: hypothesis.settings):
        """Initialize with the given values."""
        self.parsed_kwargs = parsed_kwargs
        self.product = product


def _parse_hypothesis_settings(
    settings: Mapping[str, Any]
) -> Tuple[Optional[SettingsParsing], List[str]]:
    """
    Try to parse the setting as Hypothesis settings.

    Return (settings, errors if any).
    """
    errors = []

    kwargs = collections.OrderedDict()  # type: MutableMapping[str, Any]

    for identifier, value in settings.items():
        if identifier not in _HYPOTHESIS_SETTINGS_PARAMETERS:
            errors.append(f"Invalid Hypothesis setting: {identifier!r}")
        else:
            if identifier == "verbosity":
                if value not in _HYPOTHESIS_VERBOSITY_MAP:
                    errors.append(
                        f"Invalid Hypothesis setting {identifier!r}: {value!r}"
                    )
                else:
                    kwargs[identifier] = _HYPOTHESIS_VERBOSITY_MAP[value]

            elif identifier == "phase":
                if not isinstance(value, list):
                    errors.append(
                        f"Invalid Hypothesis setting {identifier!r}: "
                        f"expected a list, but got {value!r}"
                    )
                else:
                    parsed_items = []  # type: List[Any]
                    for item in value:
                        if item not in _HYPOTHESIS_PHASE_MAP:
                            errors.append(
                                f"Invalid {hypothesis.Phase.__name__} in the Hypothesis setting "
                                f"{identifier!r}: {item!r}"
                            )
                        else:
                            parsed_items.append(_HYPOTHESIS_PHASE_MAP[item])
                    kwargs[identifier] = parsed_items

            elif identifier == "suppress_health_check":
                if not isinstance(value, list):
                    errors.append(
                        f"Invalid Hypothesis setting {identifier!r}: "
                        f"expected a list, but got {value!r}"
                    )
                else:
                    parsed_items = []
                    for item in value:
                        if item not in _HYPOTHESIS_HEALTH_CHECK_MAP:
                            errors.append(
                                f"Invalid {hypothesis.HealthCheck.__name__} in the setting "
                                f"{identifier!r}: {item!r}"
                            )
                        else:
                            parsed_items.append(_HYPOTHESIS_HEALTH_CHECK_MAP[item])
                    kwargs[identifier] = parsed_items
            else:
                kwargs[identifier] = value

    if errors:
        return None, errors

    try:
        return (
            SettingsParsing(
                parsed_kwargs=kwargs, product=hypothesis.settings(**kwargs)
            ),
            [],
        )
    except hypothesis.errors.InvalidArgument as error:
        return None, [f"Invalid Hypothesis settings: {error}"]


def parse_params(args: argparse.Namespace) -> Tuple[Optional[Params], List[str]]:
    """
    Try to parse the parameters of the command "test".

    Return (parsed parameters, errors if any).
    """
    errors = []  # type: List[str]

    path = pathlib.Path(args.path)

    settings = collections.OrderedDict()  # type: MutableMapping[str, Any]

    settings_parsing = None  # type: Optional[SettingsParsing]

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

        settings_parsing, settings_errors = _parse_hypothesis_settings(
            settings=settings
        )
        if settings_errors:
            errors.extend(settings_errors)
        else:
            assert settings_parsing is not None

    if errors:
        return None, errors

    return (
        Params(path=path, settings_parsing=settings_parsing, do_inspect=args.inspect),
        errors,
    )


def _test_function_point(
    point: _general.FunctionPoint, hypothesis_settings: Optional[hypothesis.settings]
) -> List[str]:
    """
    Test a single function point.

    Return errors if any.
    """
    errors = []  # type: List[str]

    func = point.func  # Optimize the look-up

    try:
        strategy = icontract_hypothesis.infer_strategy(func=func)
    except AssertionError:
        raise
    except Exception as error:
        error_as_str = str(error)
        if error_as_str == "":
            error_as_str = str(type(error))

        errors.append(
            f"Inference of the search strategy failed for the function: {func}. "
            f"The error was: {error_as_str}"
        )
        return errors

    def execute(kwargs: Dict[str, Any]) -> None:
        func(**kwargs)

    wrapped = hypothesis.given(strategy)(execute)
    if hypothesis_settings:
        wrapped = hypothesis_settings(wrapped)

    try:
        wrapped()
    except hypothesis.errors.FailedHealthCheck as err:
        return [
            f"Failed to test the function: "
            f"{func.__name__} from {inspect.getfile(func)} "
            f"due to the failed Hypothesis health check: {err}"
        ]

    return []


def _inspect_test_function_point(
    point: _general.FunctionPoint, settings_parsing: Optional[SettingsParsing]
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

    if settings_parsing and len(settings_parsing.parsed_kwargs) > 0:
        if len(settings_parsing.parsed_kwargs) == 1:
            identifier, value = list(settings_parsing.parsed_kwargs.items())[0]

            settings_str = f"hypothesis.settings({identifier}={value})"
        else:
            lines = ["hypothesis.settings("]
            items = list(settings_parsing.parsed_kwargs.items())
            for i, (identifier, value) in enumerate(items):
                if i < len(items) - 1:
                    lines.append(f"    {identifier}={value},")
                else:
                    lines.append(f"    {identifier}={value}")
            lines.append(")")

            settings_str = "\n".join(lines)

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
                point=point, settings_parsing=command.settings_parsing
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
            test_errors = _test_function_point(
                point=point,
                hypothesis_settings=(
                    command.settings_parsing.product
                    if command.settings_parsing is not None
                    else None
                ),
            )
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
