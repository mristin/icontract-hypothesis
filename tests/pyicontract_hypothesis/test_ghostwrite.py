# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=unused-argument
import io
import os
import pathlib
import re
import shutil
import sys
import tempfile
import unittest
from typing import List

from icontract_hypothesis.pyicontract_hypothesis import _general, _ghostwrite, main


class TestParsingOfParameters(unittest.TestCase):
    def test_subcommand_ghostwrite(self) -> None:
        # fmt: off
        argv = [
            "ghostwrite",
            "--module", "some_module",
            "--include", "include-something",
            "--exclude", "exclude-something",
            "--explicit",
            "--bare",
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

        ghostwrite, errs = _ghostwrite.parse_params(args=args)

        self.assertListEqual([], errs)
        assert ghostwrite is not None
        self.assertEqual("some_module", ghostwrite.module_name)
        self.assertTrue(ghostwrite.explicit)
        self.assertTrue(ghostwrite.bare)

    def test_path_as_input(self) -> None:
        # fmt: off
        argv = [
            "ghostwrite",
            "--path", "path/to/some_module.py"
        ]
        # fmt: on
        parser = main._make_argument_parser()
        args, out, err = main._parse_args(parser=parser, argv=argv)
        assert args is not None, "Failed to parse argv {!r}: {}".format(argv, err)

        general, errs = _general.parse_params(args=args)

        self.assertListEqual([], errs)
        assert general is not None

        ghostwrite, errs = _ghostwrite.parse_params(args=args)

        self.assertListEqual([], errs)
        assert ghostwrite is not None
        assert ghostwrite.path is not None
        assert ghostwrite.module_name is None

        self.assertEqual("path/to/some_module.py", ghostwrite.path.as_posix())


class TestQualifiedModuleNameFromPath(unittest.TestCase):
    def test_non_identifier_stem(self) -> None:
        path = pathlib.Path("some_directory") / "some-script.py"

        qualified_name = _ghostwrite._qualified_module_name_from_path(
            path=path, sys_path=[]
        )

        self.assertEqual("some_script", qualified_name)

    def test_no_parent_in_sys_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = pathlib.Path(tmpdir_str)

            (tmpdir / "some_module").mkdir()
            (tmpdir / "some_module" / "__init__.py").write_text('"""something"""')

            path = tmpdir / "some_module" / "some_submodule.py"
            path.write_text('"""something"""')

            # If tmpdir were added to sys_path, we would have a meaningful qualified name.
            # As sys_path is empty, we have to resort to a stem.
            qualified_name = _ghostwrite._qualified_module_name_from_path(
                path=path, sys_path=[]
            )

            self.assertEqual("some_submodule", qualified_name)

    def test_path_to_module_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = pathlib.Path(tmpdir_str)

            (tmpdir / "some_module").mkdir()
            (tmpdir / "some_module" / "__init__.py").write_text('"""something"""')

            path = tmpdir / "some_module" / "some_submodule.py"
            path.write_text('"""something"""')

            qualified_name = _ghostwrite._qualified_module_name_from_path(
                path=path, sys_path=[tmpdir_str]
            )

            self.assertEqual("some_module.some_submodule", qualified_name)

    def test_path_to_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = pathlib.Path(tmpdir_str)

            (tmpdir / "some_module").mkdir()
            (tmpdir / "some_module" / "__init__.py").write_text('"""something"""')

            (tmpdir / "some_module" / "some_submodule").mkdir()

            path = tmpdir / "some_module" / "some_submodule" / "__init__.py"
            path.write_text('"""something"""')

            qualified_name = _ghostwrite._qualified_module_name_from_path(
                path=path, sys_path=[tmpdir_str]
            )

            self.assertEqual("some_module.some_submodule", qualified_name)

    def test_path_to_module_file_with_ancestor_without_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir_str:
            tmpdir = pathlib.Path(tmpdir_str)

            (tmpdir / "some_module").mkdir()
            # No __init__.py in the directory ``some_module``.

            path = tmpdir / "some_module" / "some_submodule.py"
            path.write_text('"""something"""')

            qualified_name = _ghostwrite._qualified_module_name_from_path(
                path=path, sys_path=[tmpdir_str]
            )

            self.assertEqual("some_submodule", qualified_name)


class TestGhostwrite(unittest.TestCase):
    def test_path(self) -> None:
        this_dir = pathlib.Path(os.path.realpath(__file__)).parent

        path = (
            this_dir.parent.parent
            / "test_samples/pyicontract_hypothesis/sample_module.py"
        )
        # fmt: off
        argv = [
            "ghostwrite",
            "--path", str(path)
        ]
        # fmt: on

        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = main.run(argv=argv, stdout=stdout, stderr=stderr)

        self.assertEqual("", stderr.getvalue())
        self.assertEqual(exit_code, 0)

        # The qualified name of the module depends on the sys.path which, in turn, is modified
        # by the test executor (*e.g.*, IDE can add temporarily directories to sys.path
        # automatically when executing tests).
        #
        # Since the qualified name of a module depends on ``sys.path``, we need to replace it with
        # a generic placeholder to make the test output uniform and reproducible.

        qualified_name = _ghostwrite._qualified_module_name_from_path(
            path=path, sys_path=sys.path
        )
        got = stdout.getvalue()
        got = got.replace(qualified_name, "placeholder_for_sample_module")

        expected_pth = (
            this_dir.parent.parent
            / "test_samples/pyicontract_hypothesis/expected_ghostwrites"
            / ("for_{}.txt".format(TestGhostwrite.test_path.__name__))
        )

        expected = expected_pth.read_text()
        self.assertEqual(expected, got)

    def test_bare_and_explicit(self) -> None:
        # fmt: off
        argv = [
            "ghostwrite",
            "--module", "test_samples.pyicontract_hypothesis.sample_module",
            "--bare",
            "--explicit"
        ]
        # fmt: on

        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = main.run(argv=argv, stdout=stdout, stderr=stderr)

        self.assertEqual("", stderr.getvalue())
        self.assertEqual(exit_code, 0)

        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        expected_pth = (
            this_dir.parent.parent
            / "test_samples/pyicontract_hypothesis/expected_ghostwrites"
            / ("for_{}.txt".format(TestGhostwrite.test_bare_and_explicit.__name__))
        )

        expected = expected_pth.read_text()
        self.assertEqual(expected, stdout.getvalue())

    def test_bare_and_non_explicit(self) -> None:
        # fmt: off
        argv = [
            "ghostwrite",
            "--module", "test_samples.pyicontract_hypothesis.sample_module",
            "--bare",
        ]
        # fmt: on

        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = main.run(argv=argv, stdout=stdout, stderr=stderr)

        self.assertEqual("", stderr.getvalue())
        self.assertEqual(exit_code, 0)

        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        expected_pth = (
            this_dir.parent.parent
            / "test_samples/pyicontract_hypothesis/expected_ghostwrites"
            / ("for_{}.txt".format(TestGhostwrite.test_bare_and_non_explicit.__name__))
        )

        expected = expected_pth.read_text()
        self.assertEqual(expected, stdout.getvalue())

    def test_non_bare_and_explicit(self) -> None:
        # fmt: off
        argv = [
            "ghostwrite",
            "--module", "test_samples.pyicontract_hypothesis.sample_module",
            "--explicit"
        ]
        # fmt: on

        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = main.run(argv=argv, stdout=stdout, stderr=stderr)

        self.assertEqual("", stderr.getvalue())
        self.assertEqual(exit_code, 0)

        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        expected_pth = (
            this_dir.parent.parent
            / "test_samples/pyicontract_hypothesis/expected_ghostwrites"
            / ("for_{}.py".format(TestGhostwrite.test_non_bare_and_explicit.__name__))
        )

        expected = expected_pth.read_text()
        self.assertEqual(expected, stdout.getvalue())

    def test_non_bare_and_non_explicit(self) -> None:
        # fmt: off
        argv = [
            "ghostwrite",
            "--module", "test_samples.pyicontract_hypothesis.sample_module",
        ]
        # fmt: on

        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = main.run(argv=argv, stdout=stdout, stderr=stderr)

        self.assertEqual("", stderr.getvalue())
        self.assertEqual(exit_code, 0)

        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        expected_pth = (
            this_dir.parent.parent
            / "test_samples/pyicontract_hypothesis/expected_ghostwrites"
            / (
                "for_{}.py".format(
                    TestGhostwrite.test_non_bare_and_non_explicit.__name__
                )
            )
        )

        expected = expected_pth.read_text()
        self.assertEqual(expected, stdout.getvalue())

    def test_well_formatted_with_two_arguments(self) -> None:
        # This test is related to the issue:
        # https://github.com/mristin/icontract-hypothesis/issues/29
        # fmt: off
        argv = [
            "ghostwrite",
            "--module",
            "test_samples.pyicontract_hypothesis.well_formatted_with_two_arguments",
            "--explicit"
        ]
        # fmt: on

        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = main.run(argv=argv, stdout=stdout, stderr=stderr)

        self.assertEqual("", stderr.getvalue())
        self.assertEqual(exit_code, 0)

        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        expected_pth = (
            this_dir.parent.parent
            / "test_samples/pyicontract_hypothesis/expected_ghostwrites"
            / (
                "for_{}.py".format(
                    TestGhostwrite.test_well_formatted_with_two_arguments.__name__
                )
            )
        )

        expected = expected_pth.read_text()
        self.assertEqual(expected, stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
