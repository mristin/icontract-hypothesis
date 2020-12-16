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
        argv = [
            "ghostwrite",
            "--module",
            "some_module",
            "--include",
            "include-something",
            "--exclude",
            "exclude-something",
            "--explicit",
            "strategies",
            "--bare",
        ]
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
        self.assertTrue(ghostwrite.explicit, "strategies")
        self.assertTrue(ghostwrite.bare)


class TestGhostwriteAssumes(unittest.TestCase):
    def test_no_preconditions(self) -> None:
        def some_func(x: int) -> None:
            ...

        text = _ghostwrite._ghostwrite_assumes(func=some_func)

        self.assertEqual("", text)

    def test_lambda_precondition(self) -> None:
        @icontract.require(lambda x: x > 0)
        def some_func(x: int) -> None:
            ...

        text = _ghostwrite._ghostwrite_assumes(func=some_func)

        self.assertEqual("assume(x > 0)", text)

    def test_function_precondition(self) -> None:
        def some_precondition(x: int) -> bool:
            return True

        @icontract.require(some_precondition)
        def some_func(x: int) -> None:
            ...

        text = _ghostwrite._ghostwrite_assumes(func=some_func)

        self.assertEqual("assume(some_precondition(x))", text)

    def test_multiple_preconditions(self) -> None:
        @icontract.require(lambda x: x > 0)
        @icontract.require(lambda x: x < 100)
        def some_func(x: int) -> None:
            ...

        text = _ghostwrite._ghostwrite_assumes(func=some_func)

        self.assertEqual(
            textwrap.dedent(
                """\
            assume(
                (x < 100) and
                (x > 0)
            )"""
            ),
            text,
        )

    def test_weakened_single_precondition(self) -> None:
        class A(icontract.DBC):
            @icontract.require(lambda x: x % 3 == 0)
            def some_func(self, x: int) -> None:
                ...

        class B(A):
            @icontract.require(lambda x: x % 7 == 0)
            def some_func(self, x: int) -> None:
                ...

        b = B()
        text = _ghostwrite._ghostwrite_assumes(func=b.some_func)

        self.assertEqual(
            textwrap.dedent(
                """\
            assume(
                (x % 3 == 0) or 
                (x % 7 == 0)
            )"""
            ),
            text,
        )

    def test_weakened_preconditions(self) -> None:
        class A(icontract.DBC):
            @icontract.require(lambda x: x % 3 == 0)
            @icontract.require(lambda x: x > 100)
            def some_func(self, x: int) -> None:
                ...

        class B(A):
            @icontract.require(lambda x: x % 7 == 0)
            @icontract.require(lambda x: x < 200)
            def some_func(self, x: int) -> None:
                ...

        b = B()
        text = _ghostwrite._ghostwrite_assumes(func=b.some_func)

        self.assertEqual(
            textwrap.dedent(
                """\
            assume(
                (
                    (x > 100) and
                    (x % 3 == 0)
                ) or 
                (
                    (x < 200) and
                    (x % 7 == 0)
                )
            )"""
            ),
            text,
        )


class TestGhostwrite(unittest.TestCase):
    def test_bare_and_explicit_strategies(self) -> None:
        argv = [
            "ghostwrite",
            "--module",
            "tests.pyicontract_hypothesis.samples.sample_module",
            "--bare",
            "--explicit",
            "strategies",
        ]

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
                "for_{}.txt".format(
                    TestGhostwrite.test_bare_and_explicit_strategies.__name__
                )
            )
        )

        expected = expected_pth.read_text()
        self.assertEqual(expected, stdout.getvalue())

    def test_bare_and_explicit_strategies_and_assumes(self) -> None:
        argv = [
            "ghostwrite",
            "--module",
            "tests.pyicontract_hypothesis.samples.sample_module",
            "--bare",
            "--explicit",
            "strategies-and-assumes",
        ]

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
                "for_{}.txt".format(
                    TestGhostwrite.test_bare_and_explicit_strategies_and_assumes.__name__
                )
            )
        )

        expected = expected_pth.read_text()
        self.assertEqual(expected, stdout.getvalue())

    def test_bare_and_non_explicit(self) -> None:
        argv = [
            "ghostwrite",
            "--module",
            "tests.pyicontract_hypothesis.samples.sample_module",
            "--bare",
        ]

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

    def test_non_bare_and_explicit_strategies(self) -> None:
        argv = [
            "ghostwrite",
            "--module",
            "tests.pyicontract_hypothesis.samples.sample_module",
            "--explicit",
            "strategies",
        ]

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
                    TestGhostwrite.test_non_bare_and_explicit_strategies.__name__
                )
            )
        )

        expected = expected_pth.read_text()
        self.assertEqual(expected, stdout.getvalue())

    def test_non_bare_and_non_explicit(self) -> None:
        argv = [
            "ghostwrite",
            "--module",
            "tests.pyicontract_hypothesis.samples.sample_module",
        ]

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
