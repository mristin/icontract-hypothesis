# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=unused-argument
import io
import os
import pathlib
import re
import textwrap
import unittest

import icontract

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


class TestGhostwrite(unittest.TestCase):
    def test_bare_and_explicit(self) -> None:
        # fmt: off
        argv = [
            "ghostwrite",
            "--module", "tests.pyicontract_hypothesis.samples.sample_module",
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
            this_dir
            / "samples"
            / "expected_ghostwrites"
            / ("for_{}.txt".format(TestGhostwrite.test_bare_and_explicit.__name__))
        )

        expected = expected_pth.read_text()
        self.assertEqual(expected, stdout.getvalue())

    def test_bare_and_non_explicit(self) -> None:
        # fmt: off
        argv = [
            "ghostwrite",
            "--module", "tests.pyicontract_hypothesis.samples.sample_module",
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
            this_dir
            / "samples"
            / "expected_ghostwrites"
            / ("for_{}.txt".format(TestGhostwrite.test_bare_and_non_explicit.__name__))
        )

        expected = expected_pth.read_text()
        self.assertEqual(expected, stdout.getvalue())

    def test_non_bare_and_explicit(self) -> None:
        # fmt: off
        argv = [
            "ghostwrite",
            "--module", "tests.pyicontract_hypothesis.samples.sample_module",
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
            this_dir
            / "samples"
            / "expected_ghostwrites"
            / ("for_{}.py".format(TestGhostwrite.test_non_bare_and_explicit.__name__))
        )

        expected = expected_pth.read_text()
        self.assertEqual(expected, stdout.getvalue())

    def test_non_bare_and_non_explicit(self) -> None:
        # fmt: off
        argv = [
            "ghostwrite",
            "--module", "tests.pyicontract_hypothesis.samples.sample_module",
        ]
        # fmt: on

        stdout = io.StringIO()
        stderr = io.StringIO()

        exit_code = main.run(argv=argv, stdout=stdout, stderr=stderr)

        self.assertEqual("", stderr.getvalue())
        self.assertEqual(exit_code, 0)

        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        expected_pth = (
            this_dir
            / "samples"
            / "expected_ghostwrites"
            / (
                "for_{}.py".format(
                    TestGhostwrite.test_non_bare_and_non_explicit.__name__
                )
            )
        )

        expected = expected_pth.read_text()
        self.assertEqual(expected, stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
