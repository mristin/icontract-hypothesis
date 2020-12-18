# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=unused-argument
import os
import pathlib
import re
import unittest
from typing import Pattern

import icontract_hypothesis.pyicontract_hypothesis._general as _general


class TestLineRangeRe(unittest.TestCase):
    def test_only_first(self) -> None:
        mtch = _general._LINE_RANGE_RE.match(" 123 ")
        assert mtch is not None

        self.assertEqual("123", mtch.group("first"))
        self.assertIsNone(
            mtch.group("last"), "Unexpected last group: {}".format(mtch.group("last"))
        )

    def test_first_and_last(self) -> None:
        mtch = _general._LINE_RANGE_RE.match(" 123 - 435 ")
        assert mtch is not None

        self.assertEqual("123", mtch.group("first"))
        self.assertEqual("435", mtch.group("last"))

    def test_no_match(self) -> None:
        mtch = _general._LINE_RANGE_RE.match("123aa")
        assert mtch is None, "Expected no match, but got: {}".format(mtch)


class TestParsingOfPointSpecs(unittest.TestCase):
    def test_single_line(self) -> None:
        text = "123"
        point_spec, errors = _general._parse_point_spec(text=text)

        self.assertListEqual([], errors)
        assert isinstance(point_spec, _general._LineRange)
        self.assertEqual(123, point_spec.first)
        self.assertEqual(123, point_spec.last)

    def test_line_range(self) -> None:
        text = "123-345"
        point_spec, errors = _general._parse_point_spec(text=text)

        self.assertListEqual([], errors)
        assert isinstance(point_spec, _general._LineRange)
        self.assertEqual(123, point_spec.first)
        self.assertEqual(345, point_spec.last)

    def test_invalid_line_range(self) -> None:
        text = "345-123"
        point_spec, errors = _general._parse_point_spec(text=text)

        assert point_spec is None
        self.assertListEqual(["Unexpected line range (last < first): 345-123"], errors)

    def test_pattern(self) -> None:
        text = r"^do_.*$"
        point_spec, errors = _general._parse_point_spec(text=text)

        self.assertListEqual([], errors)
        assert isinstance(point_spec, Pattern)
        self.assertEqual(text, point_spec.pattern)


class TestSelectFunctionPoints(unittest.TestCase):
    def test_invalid_module(self) -> None:
        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        path = this_dir / "samples" / "sample_invalid_module.py"

        mod, errors = _general.load_module_from_source_file(path=path)
        self.assertListEqual([], errors)
        assert mod is not None

        points, errors = _general.select_function_points(
            source_code=path.read_text(), mod=mod, include=[], exclude=[]
        )

        self.assertListEqual(
            [
                "Unexpected directive on line 8. "
                "Expected '# pyicontract-hypothesis: (disable|enable)', "
                "but got: # pyicontract-hypothesis: disable-once"
            ],
            errors,
        )

    def test_no_include_and_no_exclude(self) -> None:
        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        path = this_dir / "samples" / "sample_module.py"

        mod, errors = _general.load_module_from_source_file(path=path)
        self.assertListEqual([], errors)
        assert mod is not None

        points, errors = _general.select_function_points(
            source_code=path.read_text(), mod=mod, include=[], exclude=[]
        )
        self.assertListEqual([], errors)

        self.assertListEqual(
            ["some_func", "another_func", "yet_another_func"],
            [point.func.__name__ for point in points],
        )

    def test_include_line_range(self) -> None:
        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        path = this_dir / "samples" / "sample_module.py"

        mod, errors = _general.load_module_from_source_file(path=path)
        self.assertListEqual([], errors)
        assert mod is not None

        points, errors = _general.select_function_points(
            source_code=path.read_text(),
            mod=mod,
            # A single line that overlaps the function should be enough to include it.
            include=[_general._LineRange(first=13, last=13)],
            exclude=[],
        )
        self.assertListEqual([], errors)

        self.assertListEqual(["some_func"], [point.func.__name__ for point in points])

    def test_include_pattern(self) -> None:
        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        path = this_dir / "samples" / "sample_module.py"

        mod, errors = _general.load_module_from_source_file(path=path)
        self.assertListEqual([], errors)
        assert mod is not None

        points, errors = _general.select_function_points(
            source_code=path.read_text(),
            mod=mod,
            include=[re.compile(r"^some_.*$")],
            exclude=[],
        )
        self.assertListEqual([], errors)

        self.assertListEqual(["some_func"], [point.func.__name__ for point in points])

    def test_exclude_line_range(self) -> None:
        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        path = this_dir / "samples" / "sample_module.py"

        mod, errors = _general.load_module_from_source_file(path=path)
        self.assertListEqual([], errors)
        assert mod is not None

        points, errors = _general.select_function_points(
            source_code=path.read_text(),
            mod=mod,
            include=[],
            # A single line that overlaps the function should be enough to exclude it.
            exclude=[_general._LineRange(first=13, last=13)],
        )
        self.assertListEqual([], errors)

        self.assertListEqual(
            ["another_func", "yet_another_func"],
            [point.func.__name__ for point in points],
        )

    def test_exclude_pattern(self) -> None:
        this_dir = pathlib.Path(os.path.realpath(__file__)).parent
        path = this_dir / "samples" / "sample_module.py"

        mod, errors = _general.load_module_from_source_file(path=path)
        self.assertListEqual([], errors)
        assert mod is not None

        points, errors = _general.select_function_points(
            source_code=path.read_text(),
            mod=mod,
            include=[],
            exclude=[re.compile(r"^some_.*$")],
        )
        self.assertListEqual([], errors)

        self.assertListEqual(
            ["another_func", "yet_another_func"],
            [point.func.__name__ for point in points],
        )


if __name__ == "__main__":
    unittest.main()
