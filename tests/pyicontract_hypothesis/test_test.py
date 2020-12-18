# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=unused-argument
import io
import os
import pathlib
import re
import unittest
import uuid

from icontract_hypothesis.pyicontract_hypothesis import _general, main, _test


class TestParsingOfParameters(unittest.TestCase):
    def test_subcommand_test(self) -> None:
        # fmt: off
        argv = [
            "test",
            "--path", "some_module.py",
            "--include", "include-something",
            "--exclude", "exclude-something",
            "--setting", "suppress_health_check=[2, 3]",
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
        self.assertDictEqual({"suppress_health_check": [2, 3]}, dict(test.settings))


class TestTest(unittest.TestCase):
    def test_default_behavior(self) -> None:
        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        path = this_dir / "samples" / "sample_module.py"

        mod, errors = _general.load_module_from_source_file(path=path)
        self.assertListEqual([], errors)
        assert mod is not None

        points, errors = _general.select_function_points(
            source_code=path.read_text(), mod=mod, include=[], exclude=[]
        )
        self.assertListEqual([], errors)

        for point in points:
            test_errors = _test._test_function_point(point=point, settings=None)
            self.assertListEqual([], test_errors)

        some_func_calls = getattr(mod, "SOME_FUNC_CALLS")
        self.assertEqual(100, some_func_calls)

        another_func_calls = getattr(mod, "ANOTHER_FUNC_CALLS")
        self.assertEqual(100, another_func_calls)

    def test_settings(self) -> None:
        settings = {"max_examples": 10}

        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        path = this_dir / "samples" / "sample_module.py"

        mod, errors = _general.load_module_from_source_file(path=path)
        self.assertListEqual([], errors)
        assert mod is not None

        points, errors = _general.select_function_points(
            source_code=path.read_text(), mod=mod, include=[], exclude=[]
        )
        self.assertListEqual([], errors)

        for point in points:
            test_errors = _test._test_function_point(point=point, settings=settings)
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
        pth = this_dir / "samples" / "sample_module.py"

        argv = ["test", "--path", str(pth)]

        stdout = io.StringIO()
        stderr = io.StringIO()

        # This is merely a smoke test.
        exit_code = main.run(argv=argv, stdout=stdout, stderr=stderr)

        self.assertEqual("", stderr.getvalue())
        self.assertEqual(exit_code, 0)

    def test_with_settings(self) -> None:
        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        pth = this_dir / "samples" / "sample_module.py"

        argv = ["test", "--path", str(pth), "--settings", "max_examples=5"]

        stdout = io.StringIO()
        stderr = io.StringIO()

        # This is merely a smoke test.
        exit_code = main.run(argv=argv, stdout=stdout, stderr=stderr)

        self.assertEqual("", stderr.getvalue())
        self.assertEqual(exit_code, 0)

    def test_with_include_exclude(self) -> None:
        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        pth = this_dir / "samples" / "sample_module.py"

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


if __name__ == "__main__":
    unittest.main()
