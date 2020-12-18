"""Test that DBC classes are correctly registered with Hypothesis."""
import unittest

import hypothesis.strategies._internal.types


def main() -> None:
    """Execute the test script."""
    import icontract

    assert (
        icontract._metaclass._CONTRACT_CLASSES is not None
    ), "icontract has been unexpectedly already monkey-patched"

    class A(icontract.DBC):
        pass

    assert [A] == list(icontract._metaclass._CONTRACT_CLASSES)

    import icontract_hypothesis

    assert not hasattr(
        icontract._metaclass, "_CONTRACT_CLASSES"
    ), "Expected _CONTRACT_CLASSES to be deleted upon monkey-patching"

    assert A in hypothesis.strategies._internal.types._global_type_lookup

    class B(icontract.DBC):
        pass

    assert B in hypothesis.strategies._internal.types._global_type_lookup


if __name__ == "__main__":
    main()
