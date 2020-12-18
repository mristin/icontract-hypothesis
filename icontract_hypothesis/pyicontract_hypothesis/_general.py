"""Provide functions common to all the commands."""
import argparse
import importlib
import importlib.machinery
import inspect
import io
import json
import pathlib
import re
import tokenize
import types
from typing import List, Union, Tuple, Optional, Callable, Any, Pattern

import icontract


class _LineRange:
    """Represent a line range (indexed from 1, both first and last inclusive)."""

    def __init__(self, first: int, last: int) -> None:
        """Initialize with the given values."""
        self.first = first
        self.last = last


class Params:
    """Represent general program parameters specified regardless of the command."""

    def __init__(
        self,
        include: List[Union[Pattern[str], _LineRange]],
        exclude: List[Union[Pattern[str], _LineRange]],
    ) -> None:
        """Initialize with the given values."""
        self.include = include  # type: List[Union[Pattern[str], _LineRange]]
        self.exclude = exclude  # type: List[Union[Pattern[str], _LineRange]]


_LINE_RANGE_RE = re.compile(
    r"^\s*(?P<first>[0-9]|[1-9][0-9]+)(\s*-\s*(?P<last>[1-9]|[1-9][0-9]+))?\s*$"
)


def _parse_point_spec(
    text: str,
) -> Tuple[Optional[Union[_LineRange, Pattern[str]]], List[str]]:
    """
    Try to parse the given specification of function point(s).

    Return (parsed point spec, errors if any)
    """
    errors = []  # type: List[str]

    mtch = _LINE_RANGE_RE.match(text)
    if mtch:
        if mtch.group("last") is None:
            first = int(mtch.group("first"))
            if first <= 0:
                errors.append(
                    "Unexpected line index (expected to start from 1): {}".format(text)
                )
                return None, errors

            return _LineRange(first=int(mtch.group("first")), last=first), errors
        else:
            first = int(mtch.group("first"))
            last = int(mtch.group("last"))

            if first <= 0:
                errors.append(
                    "Unexpected line index (expected to start from 1): {}".format(text)
                )
                return None, errors

            if last < first:
                errors.append("Unexpected line range (last < first): {}".format(text))
                return None, errors

            else:
                return (
                    _LineRange(
                        first=int(mtch.group("first")), last=int(mtch.group("last"))
                    ),
                    errors,
                )

    try:
        pattern = re.compile(text)
        return pattern, errors
    except re.error as err:
        errors.append("Failed to parse the pattern {}: {}".format(text, err))
        return None, errors


def parse_params(args: argparse.Namespace) -> Tuple[Optional[Params], List[str]]:
    """
    Try to parse general parameters of the program (regardless of the command).

    Return (parsed parameters, errors if any).
    """
    errors = []  # type: List[str]

    include = []  # type: List[Union[Pattern[str], _LineRange]]
    if args.include is not None:
        for include_str in args.include:
            point_spec, point_spec_errors = _parse_point_spec(text=include_str)
            errors.extend(point_spec_errors)

            if not point_spec_errors:
                assert point_spec is not None
                include.append(point_spec)

    exclude = []  # type: List[Union[Pattern[str], _LineRange]]
    if args.exclude is not None:
        for exclude_str in args.exclude:
            point_spec, point_spec_errors = _parse_point_spec(text=exclude_str)
            errors.extend(point_spec_errors)

            if not point_spec_errors:
                assert point_spec is not None
                exclude.append(point_spec)

    if errors:
        return None, errors

    return Params(include=include, exclude=exclude), errors


def load_module_with_name(name: str) -> Tuple[Optional[types.ModuleType], List[str]]:
    """
    Load the module given its name.

    Example identifier: some.module
    """
    try:
        mod = importlib.import_module(name=name)
        assert isinstance(mod, types.ModuleType)
        return mod, []
    except Exception as error:
        return None, ["Failed to import the module {}: {}".format(name, error)]


def load_module_from_source_file(
    path: pathlib.Path,
) -> Tuple[Optional[types.ModuleType], List[str]]:
    """
    Try to load a module from the source file.

    Return (loaded module, errors if any).
    """
    fullname = re.sub(r"[^A-Za-z0-9_]", "_", path.stem)

    mod = None  # type: Optional[types.ModuleType]
    try:
        loader = importlib.machinery.SourceFileLoader(fullname=fullname, path=str(path))
        mod = types.ModuleType(loader.name)
        loader.exec_module(mod)
    except Exception as error:
        return None, ["Failed to import the file {}: {}".format(path, error)]

    assert mod is not None, "Expected mod to be set before"

    return mod, []


class FunctionPoint:
    """Represent a testable function."""

    @icontract.require(lambda first_row: first_row > 0)
    @icontract.require(lambda last_row: last_row > 0)
    @icontract.require(lambda first_row, last_row: first_row <= last_row)
    def __init__(self, first_row: int, last_row: int, func: Callable[..., Any]) -> None:
        """
        Initialize with the given values.

        First and last row are both inclusive.
        """
        self.first_row = first_row
        self.last_row = last_row
        self.func = func


def _overlap(first: int, last: int, another_first: int, another_last: int) -> bool:
    """
    Return True if the two intervals overlap.

    >>> not any([
    ...     _overlap(1, 1, 2, 2),
    ...     _overlap(2, 2, 1, 1)
    ... ])
    True

    >>> all([
    ...     _overlap(1, 1, 1, 1),
    ...     _overlap(1, 5, 1, 1),
    ...     _overlap(1, 1, 1, 5),
    ...     _overlap(1, 3, 2, 5),
    ...     _overlap(2, 5, 1, 3),
    ...     _overlap(1, 5, 2, 3),
    ...     _overlap(2, 3, 1, 5),
    ...  ])
    True
    """
    return min(last, another_last) - max(first, another_first) >= 0


_DIRECTIVE_RE = re.compile(r"^#\s*pyicontract-hypothesis\s*:\s*(?P<value>[^ \t]*)\s*$")


def select_function_points(
    source_code: str,
    mod: types.ModuleType,
    include: List[Union[_LineRange, Pattern[str]]],
    exclude: List[Union[_LineRange, Pattern[str]]],
) -> Tuple[List[FunctionPoint], List[str]]:
    """Select the function points from the module based on the ``include`` and ``exclude``."""
    included = []  # type: List[FunctionPoint]
    errors = []  # type: List[str]

    for key in dir(mod):
        value = getattr(mod, key)
        if inspect.isfunction(value):
            func = value  # type: Callable[..., Any]

            # Ignore imported functions
            if func.__module__ != mod.__name__:
                continue

            source_lines, srow = inspect.getsourcelines(func)
            point = FunctionPoint(
                first_row=srow, last_row=srow + len(source_lines) - 1, func=func
            )
            included.append(point)

    # The built-in dir() gives us an unsorted directory.
    included = sorted(included, key=lambda point: point.first_row)

    ##
    # Add ranges of lines given by comment directives to the ``exclude``
    ##

    extended_exclude = exclude[:]

    range_start = None  # type: Optional[int]
    reader = io.BytesIO(source_code.encode("utf-8"))
    for toktype, _, (first_row, _), _, line in tokenize.tokenize(reader.readline):
        if toktype == tokenize.COMMENT:
            mtch = _DIRECTIVE_RE.match(line.strip())
            if mtch:
                value = mtch.group("value")

                if value not in ["enable", "disable"]:
                    errors.append(
                        (
                            "Unexpected directive on line {}. "
                            "Expected '# pyicontract-hypothesis: (disable|enable)', "
                            "but got: {}"
                        ).format(first_row, line.strip())
                    )
                    continue

                if value == "disable":
                    if range_start is not None:
                        continue

                    range_start = first_row

                elif value == "enable":
                    if range_start is not None:
                        extended_exclude.append(
                            _LineRange(first=range_start, last=first_row)
                        )

                else:
                    raise AssertionError(
                        "Unexpected value: {}".format(json.dumps(value))
                    )

    exclude = extended_exclude

    if errors:
        return [], errors

    ##
    # Remove ``included`` which do not match ``include``
    ##

    if len(include) > 0:
        incl_line_ranges = [incl for incl in include if isinstance(incl, _LineRange)]
        if len(incl_line_ranges) > 100:
            print(
                (
                    "There are much more --include items then expected: {0}. "
                    "Please consider filing an issue by visiting this link: "
                    "https://github.com/Parquery/icontract/issues/new"
                    "?title=Use+interval+tree"
                    "&body=We+had+{0}+include+line+ranges+in+pyicontract-hypothesis."
                ).format(len(incl_line_ranges))
            )

        if len(incl_line_ranges) > 0:
            filtered_included = []  # type: List[FunctionPoint]
            for point in included:
                overlaps_include = any(
                    _overlap(
                        first=line_range.first,
                        last=line_range.last,
                        another_first=point.first_row,
                        another_last=point.last_row,
                    )
                    for line_range in incl_line_ranges
                )

                if overlaps_include:
                    filtered_included.append(point)

            included = filtered_included

        # Match regular expressions
        patterns = [incl for incl in include if isinstance(incl, Pattern)]
        if len(patterns) > 0:
            filtered_included = []
            for pattern in patterns:
                for point in included:
                    if pattern.match(point.func.__name__):
                        filtered_included.append(point)

            included = filtered_included

    if len(included) == 0:
        return [], []

    ##
    # Exclude all points in ``included`` if matched in ``exclude``
    ##

    if len(exclude) > 0:
        excl_line_ranges = [excl for excl in exclude if isinstance(excl, _LineRange)]
        if len(excl_line_ranges) > 100:
            print(
                (
                    "There are much more --exclude items then expected: {0}. "
                    "Please consider filing an issue by visiting this link: "
                    "https://github.com/Parquery/icontract/issues/new"
                    "?title=Use+interval+tree"
                    "&body=We+had+{0}+exclude+line+ranges+in+pyicontract-hypothesis."
                ).format(len(excl_line_ranges))
            )

        if len(excl_line_ranges) > 0:
            filtered_included = []
            for point in included:
                overlaps_exclude = any(
                    _overlap(
                        first=line_range.first,
                        last=line_range.last,
                        another_first=point.first_row,
                        another_last=point.last_row,
                    )
                    for line_range in excl_line_ranges
                )

                if not overlaps_exclude:
                    filtered_included.append(point)

            included = filtered_included

        patterns = [excl for excl in exclude if isinstance(excl, Pattern)]
        if len(patterns) > 0:
            filtered_included = []
            for pattern in patterns:
                for point in included:
                    if not pattern.match(point.func.__name__):
                        filtered_included.append(point)

            included = filtered_included

    return included, []
