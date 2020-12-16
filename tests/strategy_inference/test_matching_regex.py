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

        strategies = icontract_hypothesis.infer_strategies(some_func)
        self.assertEqual(
            "{'s': from_regex(re.compile(r'^Start.*End$', re.UNICODE))}",
            str(strategies),
        )

    def test_re_renamed_match(self) -> None:
        import re as rerenamed

        @icontract.require(lambda s: rerenamed.match(r"^Start.*End$", s, flags=0))
        def some_func(s: str) -> None:
            pass

        strategies = icontract_hypothesis.infer_strategies(some_func)
        self.assertEqual(
            "{'s': from_regex(re.compile(r'^Start.*End$', re.UNICODE))}",
            str(strategies),
        )

    def test_multiple_re_match(self) -> None:
        @icontract.require(lambda s: re.match(r"^Start.*End$", s, flags=0))
        @icontract.require(lambda s: re.match(r"^.*something.*$", s, flags=0))
        def some_func(s: str) -> None:
            pass

        strategies = icontract_hypothesis.infer_strategies(some_func)
        self.assertEqual(
            (
                "{'s': from_regex(re.compile(r'^.*something.*$', re.UNICODE))"
                '.filter(lambda s: re.match(r"^Start.*End$", s, flags=0))}'
            ),
            str(strategies),
        )

    def test_compiled_re(self) -> None:
        START_END_RE = re.compile(r"^Start.*End$")

        @icontract.require(lambda s: START_END_RE.match(s))
        def some_func(s: str) -> None:
            pass

        strategies = icontract_hypothesis.infer_strategies(some_func)
        self.assertEqual(
            "{'s': from_regex(re.compile(r'^Start.*End$', re.UNICODE))}",
            str(strategies),
        )

    def test_compiled_re_with_logic(self) -> None:
        START_END_RE = re.compile(r"^Start.*End$")
        PREFIX_SUFFIX_RE = re.compile(r"^Prefix.*Suffix")

        SWITCH = False

        # This pre-condition also tests a bit more complicated logic so that we are sure that
        # the recomputation is executed successfully.
        @icontract.require(
            lambda s: (START_END_RE if SWITCH else PREFIX_SUFFIX_RE).match(s)
        )
        def some_func(s: str) -> None:
            pass

        strategies = icontract_hypothesis.infer_strategies(some_func)
        self.assertEqual(
            "{'s': from_regex(re.compile(r'^Prefix.*Suffix', re.UNICODE))}",
            str(strategies),
        )


if __name__ == "__main__":
    unittest.main()
