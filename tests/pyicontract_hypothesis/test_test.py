# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=unused-argument
import io
import os
import pathlib
import re
import textwrap
import unittest
import uuid

import hypothesis

from icontract_hypothesis.pyicontract_hypothesis import _general, main, _test


class TestParsingOfParameters(unittest.TestCase):
    def test_subcommand_test(self) -> None:
        # fmt: off
        argv = [
            "test",
            "--path", "some_module.py",
            "--include", "include-something",
            "--exclude", "exclude-something",
            "--setting",
            'suppress_health_check=["too_slow"]', 'verbosity="verbose"',
        ]
        # fmt: on

        parser = main._make_argument_parser()
        args, out, err = main._parse_args(parser=parser, argv=argv)
        assert args is not None, "Failed to parse argv {!r}: {}".format(argv, err)

        general, errs = _general.parse_params(args=args)

        self.assertListEqual([], errs)
        assert general is not None
        self.assertListEqual(
            [re.compile(pattern) for pattern in ["include-something"]], general.include
        )
        self.assertListEqual(
            [re.compile(pattern) for pattern in ["exclude-something"]], general.exclude
        )

        test, errs = _test.parse_params(args=args)

        self.assertListEqual([], errs)
        assert test is not None
        self.assertEqual(pathlib.Path("some_module.py"), test.path)

        assert test.settings_parsing is not None

        self.assertListEqual(
            [hypothesis.HealthCheck.too_slow],
            test.settings_parsing.product.__dict__["suppress_health_check"],
        )

        self.assertEqual(
            hypothesis.Verbosity.verbose,
            test.settings_parsing.product.__dict__["verbosity"],
        )

    def test_unknown_settings(self) -> None:
        # fmt: off
        argv = [
            "test",
            "--path", "some_module.py",
            "--setting", "totally_invalid=[2, 3]",
        ]
        # fmt: on

        parser = main._make_argument_parser()
        args, out, err = main._parse_args(parser=parser, argv=argv)
        assert args is not None, "Failed to parse argv {!r}: {}".format(argv, err)

        general, errs = _general.parse_params(args=args)
        self.assertListEqual([], errs)

        test, errs = _test.parse_params(args=args)
        self.assertListEqual(["Invalid Hypothesis setting: 'totally_invalid'"], errs)

    def test_invalid_settings(self) -> None:
        # fmt: off
        argv = [
            "test",
            "--path", "some_module.py",
            "--setting", "max_examples=-1",
        ]
        # fmt: on

        parser = main._make_argument_parser()
        args, out, err = main._parse_args(parser=parser, argv=argv)
        assert args is not None, "Failed to parse argv {!r}: {}".format(argv, err)

        general, errs = _general.parse_params(args=args)
        self.assertListEqual([], errs)

        test, errs = _test.parse_params(args=args)
        self.assertListEqual(
            [
                "Invalid Hypothesis settings: max_examples=-1 should be at least one. You can "
                "disable example generation with the `phases` setting instead."
            ],
            errs,
        )


class TestTest(unittest.TestCase):
    def test_default_behavior(self) -> None:
        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        path = (
            this_dir.parent.parent
            / "test_samples/pyicontract_hypothesis/sample_module.py"
        )

        mod, errors = _general.load_module_from_source_file(path=path)
        self.assertListEqual([], errors)
        assert mod is not None

        points, errors = _general.select_function_points(
            source_code=path.read_text(), mod=mod, include=[], exclude=[]
        )
        self.assertListEqual([], errors)

        for point in points:
            test_errors = _test._test_function_point(
                point=point, hypothesis_settings=None
            )
            self.assertListEqual([], test_errors)

        some_func_calls = getattr(mod, "SOME_FUNC_CALLS")
        self.assertEqual(100, some_func_calls)

        another_func_calls = getattr(mod, "ANOTHER_FUNC_CALLS")
        self.assertEqual(100, another_func_calls)

    def test_settings(self) -> None:
        settings_parsing, errors = _test._parse_hypothesis_settings(
            {"max_examples": 10}
        )
        self.assertListEqual([], errors)
        assert settings_parsing is not None

        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        path = (
            this_dir.parent.parent
            / "test_samples/pyicontract_hypothesis/sample_module.py"
        )

        mod, errors = _general.load_module_from_source_file(path=path)
        self.assertListEqual([], errors)
        assert mod is not None

        points, errors = _general.select_function_points(
            source_code=path.read_text(), mod=mod, include=[], exclude=[]
        )
        self.assertListEqual([], errors)

        for point in points:
            test_errors = _test._test_function_point(
                point=point, hypothesis_settings=settings_parsing.product
            )
            self.assertListEqual([], test_errors)

        some_func_calls = getattr(mod, "SOME_FUNC_CALLS")
        self.assertEqual(10, some_func_calls)

        another_func_calls = getattr(mod, "ANOTHER_FUNC_CALLS")
        self.assertEqual(10, another_func_calls)


class TestTestViaSmoke(unittest.TestCase):
    """Perform smoke testing of the "test" command."""

    def test_nonexisting_file(self) -> None:
        path = "doesnt-exist.{}".format(uuid.uuid4())
        argv = ["test", "--path", path]

        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = main.run(argv=argv, stdout=stdout, stderr=stderr)

        self.assertEqual(
            "The file to be tested does not exist: {}".format(path),
            stderr.getvalue().strip(),
        )
        self.assertEqual(exit_code, 1)

    def test_common_case(self) -> None:
        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        pth = (
            this_dir.parent.parent
            / "test_samples/pyicontract_hypothesis/sample_module.py"
        )

        argv = ["test", "--path", str(pth)]

        stdout = io.StringIO()
        stderr = io.StringIO()

        # This is merely a smoke test.
        exit_code = main.run(argv=argv, stdout=stdout, stderr=stderr)

        self.assertEqual("", stderr.getvalue())
        self.assertEqual(exit_code, 0)

        out = re.sub("\(time delta [^)]*\)", "(time delta <erased>)", stdout.getvalue())

        self.assertEqual(
            textwrap.dedent(
                """\
            Tested some_func at line 10 (time delta <erased>).
            Tested another_func at line 19 (time delta <erased>).
            Tested yet_another_func at line 28 (time delta <erased>).
            """
            ),
            out,
        )

    def test_with_settings(self) -> None:
        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        pth = (
            this_dir.parent.parent
            / "test_samples/pyicontract_hypothesis/sample_module.py"
        )

        argv = ["test", "--path", str(pth), "--settings", "max_examples=5"]

        stdout = io.StringIO()
        stderr = io.StringIO()

        # This is merely a smoke test.
        exit_code = main.run(argv=argv, stdout=stdout, stderr=stderr)

        self.assertEqual("", stderr.getvalue())
        self.assertEqual(exit_code, 0)

    def test_with_include_exclude(self) -> None:
        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        pth = (
            this_dir.parent.parent
            / "test_samples/pyicontract_hypothesis/sample_module.py"
        )

        # fmt: off
        argv = [
            "test",
            "--path", str(pth),
            "--include", ".*_func",
            "--exclude", "some.*",
        ]
        # fmt: on

        stdout = io.StringIO()
        stderr = io.StringIO()

        # This is merely a smoke test.
        exit_code = main.run(argv=argv, stdout=stdout, stderr=stderr)

        self.assertEqual("", stderr.getvalue())
        self.assertEqual(exit_code, 0)


class TestInspect(unittest.TestCase):
    """Perform smoke testing of the "test" command."""

    def test_common_case(self) -> None:
        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        pth = (
            this_dir.parent.parent
            / "test_samples/pyicontract_hypothesis/sample_module.py"
        )

        argv = ["test", "--inspect", "--path", str(pth)]

        stdout = io.StringIO()
        stderr = io.StringIO()

        # This is merely a smoke test.
        exit_code = main.run(argv=argv, stdout=stdout, stderr=stderr)

        self.assertEqual("", stderr.getvalue())
        self.assertEqual(exit_code, 0)

        self.assertEqual(
            textwrap.dedent(
                """\
            some_func at line 10:
               hypothesis.given(
                   fixed_dictionaries({'x': integers(min_value=1)})
               )
            
            another_func at line 19:
               hypothesis.given(
                   fixed_dictionaries({'x': integers(min_value=1).filter(lambda x: square_greater_than_zero(x))})
               )
            
            yet_another_func at line 28:
               hypothesis.given(
                   fixed_dictionaries({'x': integers(), 'y': integers()}).filter(lambda d: d['x'] < d['y'])
               )
            """
            ),
            stdout.getvalue(),
        )

    def test_with_settings(self) -> None:
        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        pth = (
            this_dir.parent.parent
            / "test_samples/pyicontract_hypothesis/sample_module.py"
        )

        # fmt: off
        argv = [
            "test",
            "--inspect",
            "--path", str(pth),
            "--settings",
            "max_examples=5",
            'suppress_health_check=["too_slow"]']
        # fmt: on

        stdout = io.StringIO()
        stderr = io.StringIO()

        # This is merely a smoke test.
        exit_code = main.run(argv=argv, stdout=stdout, stderr=stderr)

        self.assertEqual("", stderr.getvalue())
        self.assertEqual(exit_code, 0)

        self.assertEqual(
            textwrap.dedent(
                """\
                some_func at line 10:
                   hypothesis.given(
                       fixed_dictionaries({'x': integers(min_value=1)})
                   )
                   hypothesis.settings(
                       max_examples=5,
                       suppress_health_check=[HealthCheck.too_slow]
                   )
                
                another_func at line 19:
                   hypothesis.given(
                       fixed_dictionaries({'x': integers(min_value=1).filter(lambda x: square_greater_than_zero(x))})
                   )
                   hypothesis.settings(
                       max_examples=5,
                       suppress_health_check=[HealthCheck.too_slow]
                   )
                
                yet_another_func at line 28:
                   hypothesis.given(
                       fixed_dictionaries({'x': integers(), 'y': integers()}).filter(lambda d: d['x'] < d['y'])
                   )
                   hypothesis.settings(
                       max_examples=5,
                       suppress_health_check=[HealthCheck.too_slow]
                   )
            """
            ),
            stdout.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
