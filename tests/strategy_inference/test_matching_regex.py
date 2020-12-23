# pylint: disable=missing-docstring
# pylint: disable=invalid-name
# pylint: disable=unused-argument
# pylint: disable=no-value-for-parameter


import re
import unittest

import icontract

import icontract_hypothesis


class TestMatchingRegex(unittest.TestCase):
    def test_re_match(self) -> None:
        @icontract.require(lambda s: re.match(r"^Start.*End$", s, flags=0))
        def some_func(s: str) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'s': from_regex(re.compile(r'^Start.*End$', re.UNICODE))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_re_renamed_match(self) -> None:
        import re as rerenamed

        @icontract.require(lambda s: rerenamed.match(r"^Start.*End$", s, flags=0))
        def some_func(s: str) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'s': from_regex(re.compile(r'^Start.*End$', re.UNICODE))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_multiple_re_match(self) -> None:
        @icontract.require(lambda s: re.match(r"^Start.*End$", s, flags=0))
        @icontract.require(lambda s: re.match(r"^.*something.*$", s, flags=0))
        def some_func(s: str) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            (
                "fixed_dictionaries("
                "{'s': from_regex(re.compile(r'^.*something.*$', re.UNICODE))"
                '.filter(lambda s: re.match(r"^Start.*End$", s, flags=0))})'
            ),
            str(strategy),
        )

        # We can not test this strategy as it is currently not matched.

    def test_compiled_re(self) -> None:
        START_END_RE = re.compile(r"^Start.*End$")

        @icontract.require(lambda s: START_END_RE.match(s))
        def some_func(s: str) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'s': from_regex(re.compile(r'^Start.*End$', re.UNICODE))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)

    def test_compiled_re_with_logic(self) -> None:
        START_END_RE = re.compile(r"^Start.*End$")
        PREFIX_SUFFIX_RE = re.compile(r"^Prefix.*Suffix")

        SWITCH = False

        # This pre-condition also tests a bit more complicated logic so that we are sure that
        # the re-computation is executed successfully.
        @icontract.require(
            lambda s: (START_END_RE if SWITCH else PREFIX_SUFFIX_RE).match(s)
        )
        def some_func(s: str) -> None:
            pass

        strategy = icontract_hypothesis.infer_strategy(some_func)
        self.assertEqual(
            "fixed_dictionaries({'s': from_regex(re.compile(r'^Prefix.*Suffix', re.UNICODE))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(some_func)


if __name__ == "__main__":
    unittest.main()
