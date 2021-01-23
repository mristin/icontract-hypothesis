# !/usr/bin/env python3

"""Combine property-based testing with contracts of a Python module."""

import argparse
import contextlib
import io
import sys
import textwrap
from typing import List, Optional, Tuple, TextIO, Union, NoReturn

import icontract_hypothesis.pyicontract_hypothesis._general as _general
import icontract_hypothesis.pyicontract_hypothesis._ghostwrite as _ghostwrite
import icontract_hypothesis.pyicontract_hypothesis._test as _test


class Params:
    """Represent the parameters of the program."""

    def __init__(
        self, general: _general.Params, command: Union[_test.Params, _ghostwrite.Params]
    ) -> None:
        """Initialize with the given values."""
        self.general = general
        self.command = command


def _parse_args_to_params(
    args: argparse.Namespace,
) -> Tuple[Optional[Params], List[str]]:
    """
    Parse the parameters from the command-line arguments.

    Return parsed parameters, errors if any
    """
    errors = []  # type: List[str]

    general, general_errors = _general.parse_params(args=args)
    errors.extend(general_errors)

    command = None  # type: Optional[Union[_test.Params, _ghostwrite.Params]]
    if args.command == "test":
        test, command_errors = _test.parse_params(args=args)
        errors.extend(command_errors)

        command = test

    elif args.command == "ghostwrite":
        ghostwrite, command_errors = _ghostwrite.parse_params(args=args)
        errors.extend(command_errors)
        command = ghostwrite

    if errors:
        return None, errors

    assert general is not None
    assert command is not None

    return Params(general=general, command=command), []


def _make_argument_parser(
    prog: str = "pyicontract-hypothesis",
) -> argparse.ArgumentParser:
    """
    Create an instance of the argument parser to parse command-line arguments.

    :param prog: name of the program to be displayed in help
    """
    parser = argparse.ArgumentParser(prog=prog, description=__doc__)
    subparsers = parser.add_subparsers(help="Commands", dest="command")
    subparsers.required = True

    test_parser = subparsers.add_parser(
        "test",
        help="Test the functions automatically by inferring search strategies from preconditions",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    test_parser.add_argument(
        "-p", "--path", help="Path to the Python file to test", required=True
    )

    test_parser.add_argument(
        "--settings",
        help=textwrap.dedent(
            """\
            Specify settings for Hypothesis

            The settings are assigned by '='.
            The value of the setting needs to be encoded as JSON.

            Example: max_examples=500"""
        ),
        nargs="*",
    )

    test_parser.add_argument(
        "--inspect",
        help=textwrap.dedent(
            """\
            Only show the strategy and the settings

            No tests are executed."""
        ),
        action="store_true",
    )

    ghostwriter_parser = subparsers.add_parser(
        "ghostwrite",
        help="Ghostwrite the unit tests with inferred search strategies",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    input_group = ghostwriter_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-m", "--module", help="Module to ghostwrite the unit tests for"
    )
    input_group.add_argument(
        "-p",
        "--path",
        help=textwrap.dedent(
            """\
            Path to the module to ghostwrite the unit tests for.

            If the file represents a module reachable through
            sys.path, use the qualified module name in
            the unit test.

            Otherwise, the module is represented as the stem
            of the path with all non-identifier characters
            replaced with an underscore ("_")."""
        ),
    )

    ghostwriter_parser.add_argument(
        "-o",
        "--output",
        help=textwrap.dedent(
            """\
            Path to the file where the output should be written.

            If '-', writes to STDOUT."""
        ),
        default="-",
    )

    ghostwriter_parser.add_argument(
        "--explicit",
        help=textwrap.dedent(
            """\
            Write the inferred strategies explicitly

            This is practical if you want to tune and
            refine the strategies and just want to use
            ghostwriting as a starting point.

            Mind that pyicontract-hypothesis does not
            automatically fix imports as this is
            usually project-specific. You have to fix imports
            manually after the ghostwriting."""
        ),
        action="store_true",
    )

    ghostwriter_parser.add_argument(
        "--bare",
        help=textwrap.dedent(
            """\
            Print only the body of the tests and omit header/footer
            (such as TestCase class or import statements).

            This is useful when you only want to inspect a single test or
            include a single test function in a custom test suite."""
        ),
        action="store_true",
    )

    for subparser in [test_parser, ghostwriter_parser]:
        subparser.add_argument(
            "-i",
            "--include",
            help=textwrap.dedent(
                """\
                Regular expressions, lines or line ranges of the functions to process

                If a line or line range overlaps the body of a function,
                the function is considered included.

                Example 1: ^do_something.*$
                Example 2: 3
                Example 3: 34-65"""
            ),
            required=False,
            nargs="*",
        )

        subparser.add_argument(
            "-e",
            "--exclude",
            help=textwrap.dedent(
                """\
                Regular expressions, lines or line ranges of the functions to exclude

                If a line or line range overlaps the body of a function,
                the function is considered excluded.

                Example 1: ^do_something.*$
                Example 2: 3
                Example 3: 34-65"""
            ),
            default=["^_.*$"],
            nargs="*",
        )

    return parser


def _parse_args(
    parser: argparse.ArgumentParser, argv: List[str]
) -> Tuple[Optional[argparse.Namespace], str, str]:
    """
    Parse the command-line arguments.

    Return (parsed args or None if failure, captured stdout, captured stderr).
    """
    pass  # for pydocstyle

    # From https://stackoverflow.com/questions/18160078
    @contextlib.contextmanager
    def captured_output():  # type: ignore
        new_out, new_err = io.StringIO(), io.StringIO()
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout, sys.stderr = new_out, new_err
            yield sys.stdout, sys.stderr
        finally:
            sys.stdout, sys.stderr = old_out, old_err

    with captured_output() as (out, err):
        try:
            parsed_args = parser.parse_args(argv)

            err.seek(0)
            out.seek(0)
            return parsed_args, out.read(), err.read()

        except SystemExit:
            err.seek(0)
            out.seek(0)
            return None, out.read(), err.read()


def assert_never(something: NoReturn) -> NoReturn:
    """Enforce exhaustive matching at mypy time."""
    assert False, "Unhandled type: {}".format(type(something).__name__)


def run(
    argv: List[str],
    stdout: TextIO,
    stderr: TextIO,
    prog: str = "pyicontract-hypothesis",
) -> int:
    """
    Execute the run routine.

    :param argv: list of command-line arguments
    :param stdout: stream to write the output to
    :param stderr: stream to write the error output to
    :param prog: name of the program to be displayed in help
    :return: exit code
    """
    parser = _make_argument_parser(prog=prog)
    args, out, err = _parse_args(parser=parser, argv=argv)
    if len(out) > 0:
        stdout.write(out)

    if len(err) > 0:
        stderr.write(err)

    if args is None:
        return 1

    params, errors = _parse_args_to_params(args=args)
    if errors:
        for error in errors:
            print(error, file=stderr)
            return 1

    assert params is not None

    if isinstance(params.command, _test.Params):
        errors = _test.test(
            general=params.general, command=params.command, stdout=stdout
        )
    elif isinstance(params.command, _ghostwrite.Params):
        code, errors = _ghostwrite.ghostwrite(
            general=params.general, command=params.command
        )
        if not errors:
            if params.command.output is None:
                stdout.write(code)
            else:
                params.command.output.write_text(code)
    else:
        assert_never(params.command)

    if errors:
        for error in errors:
            print(error, file=stderr)
            return 1

    return 0


def entry_point() -> int:
    """
    Wrap the entry_point routine wit default arguments.

    :return: exit code
    """
    return run(
        argv=sys.argv[1:],
        stdout=sys.stdout,
        stderr=sys.stderr,
        prog="pyicontract-hypothesis",
    )


if __name__ == "__main__":
    sys.exit(entry_point())
