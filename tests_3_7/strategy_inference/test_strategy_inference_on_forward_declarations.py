# Resolving type hints at runtime is difficult. In particular, ``typing.get_type_hints`` requires
# the user to specify the global and the local namespace.
#
# This makes it almost impossible to infer the type hints which use forward declaration of nested
# classes, see https://www.python.org/dev/peps/pep-0563/#resolving-type-hints-at-runtime.
#
# However, dealing with classes defined in the global space should work OK. The following tests
# check explicitly that icontract-hypothesis can deal with these cases.

import unittest
from typing import Sequence

import icontract

import icontract_hypothesis


def do_something(a: "A") -> None:
    pass


class A(icontract.DBC):
    def __init__(self, x: int) -> None:
        pass


class B:
    def __init__(self, x: int) -> None:
        self.x = x

    def do_something(self, x1: int) -> "B":
        return B(x=x1)


class C(icontract.DBC, Sequence[int]):
    @icontract.require(lambda xs: all(x > -(2 ** 63) for x in xs))
    def __new__(cls, xs: Sequence[int]) -> "C":
        pass


def some_func_on_c(c: C) -> None:
    pass


class TestForwardDeclarations(unittest.TestCase):
    def test_function(self) -> None:
        strategy = icontract_hypothesis.infer_strategy(do_something)

        self.assertEqual(
            "fixed_dictionaries("
            "{'a': "
            "fixed_dictionaries({'x': integers()})"
            ".map(lambda d: A(**d))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(do_something)

    def test_instance_method(self) -> None:
        b = B(0)

        strategy = icontract_hypothesis.infer_strategy(b.do_something)

        self.assertEqual(
            "fixed_dictionaries({'x1': integers()})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(do_something)

    def test_new(self) -> None:
        strategy = icontract_hypothesis.infer_strategy(some_func_on_c)

        self.assertEqual(
            "fixed_dictionaries("
            "{'c': "
            "fixed_dictionaries("
            "{'xs': one_of(binary(), lists(integers()))"
            ".filter(lambda xs: all(x > -(2 ** 63) for x in xs))})"
            ".map(lambda d: C(**d))})",
            str(strategy),
        )

        icontract_hypothesis.test_with_inferred_strategy(do_something)


if __name__ == "__main__":
    unittest.main()
