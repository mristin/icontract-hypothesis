"""
Test Python 3.7-specific features.

For example, we could not test ``test_strategy_inference_on_forward_declarations`` in Python 3.6 as
the multiple inheritance of ``icontract.DBC`` and ``Sequance[int]`` caused a meta-class conflict.
"""

import sys

if sys.version_info < (3, 7):
    def load_tests(loader, suite, pattern):  # pylint: disable=unused-argument
        """Ignore all the tests for lower Python versions."""
        return suite
