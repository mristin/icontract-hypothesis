"""Test for a bug when classes introduce recursivity and depend on themselves in constructors."""

import unittest
from typing import Optional

import hypothesis
import icontract
import hypothesis.strategies as st

import icontract_hypothesis


class NodeWithoutDbc:
    """
    Represent a cyclic data structure.

    Based on:
    https://github.com/mristin/python-by-contract-corpus/commit/f302812ad6bac20b97bde1b6b06ced2983e1f6dc
    """

    label: int  #: label of the cup
    next_node: "NodeWithoutDbc"  #: the next cup clockwise

    def __init__(self, label: int, next_node: Optional["NodeWithoutDbc"] = None) -> None:
        """Initialize with the given values."""
        self.label = label
        self.next_node = next_node

    def __repr__(self) -> str:
        """Represent the instance as a string for debugging."""
        return f"{self.__class__.__name__}({self.label!r}, {self.next_node!r})"


class TestManual(unittest.TestCase):
    def test_the_deferred_strategy(self) -> None:
        placeholder = st.just(1977)
        strategy = st.deferred(lambda: placeholder)
        print(f"Step 1: strategy is {strategy!r}")  # TODO: debug

        st.register_type_strategy(
            NodeWithoutDbc,
            strategy)

        strategy = st.from_type(NodeWithoutDbc)
        print(f"Step 2: strategy is {strategy!r}")  # TODO: debug

        placeholder = st.just(2021)
        strategy = st.from_type(NodeWithoutDbc)
        print(f"Step 3: strategy is {strategy!r}")  # TODO: debug


        print("got here 2")  # TODO: rem
        strategy = st.builds(
            NodeWithoutDbc,
            label=st.integers(),
            next_node=st.one_of(st.none(), st.from_type(NodeWithoutDbc)))

        print("got here 3")  # TODO: rem
        st.register_type_strategy(
            NodeWithoutDbc,
            strategy
        )

        placeholder = st.just(1)

        print("got here 4")  # TODO: rem
        registered = st.from_type(NodeWithoutDbc)

        print("got here 5")  # TODO: rem
        print(f"registered is {registered!r}")  # TODO: debug


        #
        # print("got here 6")  # TODO: rem
        #
        # self.assertEqual(
        #     "deferred(lambda: st.fixed_dictionaries({ "
        #     "'label': st.integers(), "
        #     "'next_node': st.one_of(st.none(), strategy) })"
        #     ".map(lambda kwargs: NodeWithoutDbc(**kwargs)))",
        #     str(registered)
        # )

        @hypothesis.given(registered)
        def execute(node: NodeWithoutDbc) -> None:
            pass

        execute()

    @unittest.skip("Results in an endless recursion.")
    def test_builds(self) -> None:
        # This results in an endless recursion.
        strategy = st.builds(NodeWithoutDbc)

        @hypothesis.given(strategy)
        def execute(node: NodeWithoutDbc) -> None:
            print(f"node is {node!r}")  # TODO: debug
            pass

        execute()

    @unittest.skip("Results in an endless recursion.")
    def test_nested_from_type(self) -> None:
        _ = st.fixed_dictionaries(
            {
                'label': st.integers(),
                'next_node': st.one_of(st.none(), st.from_type(NodeWithoutDbc))
            }).map(lambda kwargs: NodeWithoutDbc(**kwargs))


# class Node(icontract.DBC):
#     """
#     Represent a cyclic data structure.
#
#     Based on:
#     https://github.com/mristin/python-by-contract-corpus/commit/f302812ad6bac20b97bde1b6b06ced2983e1f6dc
#     """
#
#     label: int  #: label of the cup
#     next_node: "Node"  #: the next cup clockwise
#
#     def __init__(self, label: int, next_node: Optional["Node"] = None) -> None:
#         """Initialize with the given values."""
#         self.label = label
#         self.next_node = next_node
#
#     def __repr__(self) -> str:
#         """Represent the instance as a string for debugging."""
#         return f"{self.__class__.__name__}({self.label!r}, {self.next_node!r})"


# class TestStrategyInference(unittest.TestCase):
#     def test_case(self) -> None:
#         def printme(*args):
#             print(f"args is {args!r}")  # TODO: debug
#             return 1
#
#         strategy = lambda: printme(strategy)
#
#         strategy = hypothesis.strategies.from_type(Node)
#         self.assertEqual(
#             "",
#             str(strategy),
#         )
#
#         # We can not execute this strategy, as ``builds`` is not handling the recursivity well.
#         # Please see this Hypothesis issue:
#         # https://github.com/HypothesisWorks/hypothesis/issues/3026


if __name__ == "__main__":
    unittest.main()
