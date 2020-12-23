THIS PACKAGE IS STILL ALPHA AND HAS NOT BEEN PUBLISHED!
PLEASE DO NOT USE IT YET!

icontract-hypothesis
====================

.. image:: https://github.com/mristin/icontract-hypothesis/workflows/Test/badge.svg
    :target: https://github.com/mristin/icontract-hypothesis/actions?query=workflow%3ATest
    :alt: Test

.. image:: https://badge.fury.io/py/icontract-hypothesis.svg
    :target: https://badge.fury.io/py/icontract-hypothesis
    :alt: PyPI - version

.. image:: https://img.shields.io/pypi/pyversions/icontract-hypothesis.svg
    :alt: PyPI - Python Version

Icontract-hypothesis combines design-by-contract with automatic testing.

It is an integration between
`icontract <https://github.com/Parquery/icontract>`_
library for design-by-contract and
`Hypothesis <https://github.com/HypothesisWorks/hypothesis>`_ library for
property-based testing.

The result is a powerful combination that allows you to automatically test
your code. Instead of writing manually the Hypothesis search strategies for
a function, icontract-hypothesis infers them based on
the function's precondition. This makes automatic testing as effortless as it
goes.

You can use icontract-hypothesis:

* As a library, to write succinct unit tests,
* As a command-line tool or a tool to integrate it with your IDE.
  This allows you to automatically test functions during the development and
  use it in your continuous integration, and
* As a ghostwriter utility giving you a starting point for your more elaborate
  Hypothesis strategies.

Since the contracts live close to the code, evolving the code also automatically
evolves the tests.

Usage
-----
Library
~~~~~~~
There are two ways to integrate icontract-hypothesis in your tests as a library.

**Only assume.** First, you can use it for defining the assumptions of the test based on the
precondition:

.. code-block:: python

    >>> from hypothesis import given
    >>> import hypothesis.strategies as st

    >>> import icontract
    >>> import icontract_hypothesis

    >>> @icontract.require(lambda x: x > 0)
    ... @icontract.ensure(lambda result: result > 0)
    ... def some_func(x: int) -> int:
    ...     return x - 1000

    >>> assume_preconditions = icontract_hypothesis.make_assume_preconditions(
    ...     some_func)

    >>> @given(x=st.integers())
    ... def test_some_func(x: int) -> None:
    ...    assume_preconditions(x)
    ...    some_func(x)

    >>> test_some_func()
    Traceback (most recent call last):
        ...
    icontract.errors.ViolationError: File <doctest README.rst[4]>, line 2 in <module>:
    result > 0: result was -999

The function ``assume_preconditions`` created by
``icontract_hypothesis.make_assume_preconditions`` will reject all the input
values which do not satisfy the pre-conditions of ``some_func``.

**Infer strategy**. Second, you can automatically infer the strategy and test the function:

.. code-block:: python

    >>> import icontract
    >>> import icontract_hypothesis

    >>> @icontract.require(lambda x: x > 0)
    ... @icontract.ensure(lambda result: result > 0)
    ... def some_func(x: int) -> int:
    ...     return x - 1000

    >>> icontract_hypothesis.test_with_inferred_strategy(some_func)
    Traceback (most recent call last):
        ...
    icontract.errors.ViolationError: File <doctest README.rst[10]>, line 2 in <module>:
    result > 0: result was -999

Which approach to use depends on how you want to write your tests.
The first approach, using ``assume_preconditions``, is practical if you already
defined your search strategy and you only want to exclude a few edge cases.
The second approach, automatically inferring test strategies, is useful if you
just want to test your function without specifying any particular search strategy.

Icontract-hypothesis guarantees that the inferred strategy must satisfy the preconditions.
If it does not, you should consider it a bug in which case
please `create an issue <https://github.com/mristin/icontract-hypothesis/issues/new>`_
so that we can fix it.

If you want to inspect the strategy or further refine it programmatically, use
``icontract_hypothesis.infer_strategy``:

.. code-block:: python

    >>> import math

    >>> import icontract
    >>> import icontract_hypothesis

    >>> @icontract.require(lambda x: x > 0)
    ... @icontract.require(lambda x: x > math.sqrt(x))
    ... def some_func(x: float) -> int:
    ...     pass

    >>> icontract_hypothesis.infer_strategy(some_func)
    fixed_dictionaries({'x': floats(min_value=0, exclude_min=True).filter(lambda x: x > math.sqrt(x))})

Testing Tool
~~~~~~~~~~~~
We provide ``pyicontract-hypothesis test`` command-line tool which you can use
to automatically test a module.

.. Help starts: pyicontract-hypothesis test --help
.. code-block::

    usage: pyicontract-hypothesis test [-h] -p PATH
                                       [--settings [SETTINGS [SETTINGS ...]]]
                                       [-i [INCLUDE [INCLUDE ...]]]
                                       [-e [EXCLUDE [EXCLUDE ...]]]

    optional arguments:
      -h, --help            show this help message and exit
      -p PATH, --path PATH  Path to the Python file to test
      --settings [SETTINGS [SETTINGS ...]]
                            Specify settings for Hypothesis

                            The settings are assigned by '='.
                            The value of the setting needs to be encoded as JSON.

                            Example: max_examples=500
      -i [INCLUDE [INCLUDE ...]], --include [INCLUDE [INCLUDE ...]]
                            Regular expressions, lines or line ranges of the functions to process

                            If a line or line range overlaps the body of a function,
                            the function is considered included.

                            Example 1: ^do_something.*$
                            Example 2: 3
                            Example 3: 34-65
      -e [EXCLUDE [EXCLUDE ...]], --exclude [EXCLUDE [EXCLUDE ...]]
                            Regular expressions, lines or line ranges of the functions to exclude

                            If a line or line range overlaps the body of a function,
                            the function is considered excluded.

                            Example 1: ^do_something.*$
                            Example 2: 3
                            Example 3: 34-65

.. Help ends: pyicontract-hypothesis test --help

Note that ``pyicontract-hypothesis test`` can be trivially integrated with
your IDE if you can pass in the current cursor position and the
current file name.

Ghostwriting Tool
~~~~~~~~~~~~~~~~~
Writing property-based tests by hand is tedious and can be partially automated.
To that end, we implemented a ghostwriter utility ``pyicontract-hypothesis ghostwrite``
that generates a first draft based on pre-conditions that you manually refine further:

.. Help starts: pyicontract-hypothesis ghostwrite --help
.. code-block::

    usage: pyicontract-hypothesis ghostwrite [-h] -m MODULE [-o OUTPUT]
                                             [--explicit] [--bare]
                                             [-i [INCLUDE [INCLUDE ...]]]
                                             [-e [EXCLUDE [EXCLUDE ...]]]

    optional arguments:
      -h, --help            show this help message and exit
      -m MODULE, --module MODULE
                            Module to process
      -o OUTPUT, --output OUTPUT
                            Path to the file where the output should be written.

                            If '-', writes to STDOUT.
      --explicit            Write the inferred strategies explicitly

                            This is practical if you want to tune and
                            refine the strategies and just want to use
                            ghostwriting as a starting point.

                            Mind that pyicontract-hypothesis does not
                            automatically fix imports as this is
                            usually project-specific. You have to fix imports
                            manually after the ghostwriting.
      --bare                Print only the body of the tests and omit header/footer
                            (such as TestCase class or import statements).

                            This is useful when you only want to inspect a single test or
                            include a single test function in a custom test suite.
      -i [INCLUDE [INCLUDE ...]], --include [INCLUDE [INCLUDE ...]]
                            Regular expressions, lines or line ranges of the functions to process

                            If a line or line range overlaps the body of a function,
                            the function is considered included.

                            Example 1: ^do_something.*$
                            Example 2: 3
                            Example 3: 34-65
      -e [EXCLUDE [EXCLUDE ...]], --exclude [EXCLUDE [EXCLUDE ...]]
                            Regular expressions, lines or line ranges of the functions to exclude

                            If a line or line range overlaps the body of a function,
                            the function is considered excluded.

                            Example 1: ^do_something.*$
                            Example 2: 3
                            Example 3: 34-65

.. Help ends: pyicontract-hypothesis ghostwrite --help

The examples of ghostwritten tests are available at:
`tests/pyicontract_hypothesis/samples <https://github.com/mristin/icontract-hypothesis/blob/main/tests/pyicontract_hypothesis/samples>`_

Installation
------------
icontract-hypothesis is available on PyPI at
https://pypi.org/project/icontract-hypothesis, so you can use ``pip``:

.. code-block::

    pip3 install icontract-hypothesis


Search Strategies
-----------------
A naive approach to fuzzy testing is to randomly sample input data, filter it
based on pre-conditions and ensure post-conditions after the run. However,
if your acceptable band of input values is narrow, the rejection sampling
will become impractically slow.

For example, assume a pre-condition ``5 < x < 10``.
Sampling from all possible integers for ``x`` will rarely hit
the pre-condition (if ever) thus wasting valuable computational time.
The problem is exacerbated as the number of arguments grow due to
`the curse of dimensionality <https://en.wikipedia.org/wiki/Curse_of_dimensionality>`_.

Icontract-hypothesis tries to address the search strategies
a bit more intelligently:

* The pre-conditions are matched against common code patterns to define
  the strategies. For example, ``5 < x < 10`` gives a search strategy
  ``hypothesis.strategies.integers(min=6, max=9)``.

  We currently match bounds on all available Hypothesis types
  (``int``, ``float``, ``datetime.date`` *etc*.).
  We also match regular expressions on ``str`` arguments.

* Pre-conditions which could not be matched, but operate on a single argument
  are inferred based on the type hint and composed with Hypothesis
  ``FilteredStrategy``.

* The remainder of the pre-conditions are enforced by filtering on the whole
  fixed dictionary which is finally passed into the function as keyword arguments.

There is an ongoing effort to move the strategy matching code into Hypothesis and
develop it further to include many more cases. See
`this Hypothesis issue <https://github.com/HypothesisWorks/hypothesis/issues/2701>`_.

Classes
~~~~~~~
Hypothesis automatically builds composite input arguments (classes, dataclasses,
named tuples *etc*.). If your class enforces pre-conditions in the constructor
method (``__init__``), make sure that it inherits from ``icontract.DBC``.

That way icontract-hypothesis will use
`hypothesis.strategies.register_type_strategy <https://hypothesis.readthedocs.io/en/latest/data.html#hypothesis.strategies.register_type_strategy>`_
to register your class with Hypothesis and consider pre-conditions when building
its instances.

It is important that you do *not* use ``hypothesis.strategies.builds(.)`` with
the classes using contracts in their constructors as ``builds`` will disregard the registered
strategy. You should use ``hypothesis.strategies.from_type(.)`` instead. See
`this comment on an Hypothesis issue <https://github.com/HypothesisWorks/hypothesis/issues/2708#issuecomment-749393747>`_
and
`the corresponding answer <https://github.com/HypothesisWorks/hypothesis/issues/2708#issuecomment-749397758>`_.

Related Libraries
-----------------
Python design-by-contract libraries
`deal <https://github.com/life4/deal>`_ and
`dpcontracts <https://github.com/deadpixi/contracts>`_
integrate directly with Hypothesis (see
`this page <https://deal.readthedocs.io/basic/tests.html>`_ and
`that page <https://hypothesis.readthedocs.io/en/latest/extras.html#hypothesis-dpcontracts>`_,
respectively).

As of 2020-12-16:

* Neither of the two libraries handles behavioral sub-typing correctly
  (*i.e.*, they do not weaken and strengthen the pre-conditions, and
  post-conditions and invariants, respectively).
  Hence they can not be used with class hierarchies as the contracts are not
  properly inherited.
* They only provide rejection sampling which is insufficient for many practical
  use cases. For example, the computational time grows exponentially with the
  number of arguments (see Section "Search Strategies").
* Finally, the existing libraries do not propagate pre-conditions of
  constructors to Hypothesis so testing with composite inputs (such as instances
  of classes) is currently not possible with these two libraries.

Benchmarks
~~~~~~~~~~
We run benchmarks against `deal` and `dpcontracts` libraries as part of our continuous integration.

We benchmark against functions using 1, 2 and 3 arguments, respectively, with the precondition that
the argument should be positive (*e.g.*, ``a > 0``). We sampled 100 inputs per each run.

.. Benchmark report starts.


The following scripts were run:

* `benchmarks/compare_with_others.py <https://github.com/Parquery/icontract/tree/master/benchmarks/compare_with_others.py>`_

The benchmarks were executed on Intel(R) Xeon(R) E-2276M  CPU @ 2.80GHz.
We used Python 3.8.5, icontract 2.4.1, deal 4.4.0 and dpcontracts 0.6.0.

The following tables summarize the results.

Benchmarking Hypothesis testing:


Argument count: 1

==========================================  ============  ==============  =======================
Case                                          Total time    Time per run    Relative time per run
==========================================  ============  ==============  =======================
`benchmark_icontract_inferred_strategy`           0.48 s        48.29 ms                     100%
`benchmark_icontract_assume_preconditions`        0.79 s        78.75 ms                     163%
`benchmark_dpcontracts`                           1.06 s       106.17 ms                     220%
`benchmark_deal`                                  0.83 s        82.63 ms                     171%
==========================================  ============  ==============  =======================

Argument count: 2

==========================================  ============  ==============  =======================
Case                                          Total time    Time per run    Relative time per run
==========================================  ============  ==============  =======================
`benchmark_icontract_inferred_strategy`           0.63 s        63.45 ms                     100%
`benchmark_icontract_assume_preconditions`        1.65 s       165.05 ms                     260%
`benchmark_dpcontracts`                           2.10 s       209.51 ms                     330%
`benchmark_deal`                                  1.61 s       161.09 ms                     254%
==========================================  ============  ==============  =======================

Argument count: 3

==========================================  ============  ==============  =======================
Case                                          Total time    Time per run    Relative time per run
==========================================  ============  ==============  =======================
`benchmark_icontract_inferred_strategy`           0.72 s        71.66 ms                     100%
`benchmark_icontract_assume_preconditions`        3.30 s       330.20 ms                     461%
`benchmark_dpcontracts`                           4.23 s       423.31 ms                     591%
`benchmark_deal`                                  3.20 s       319.57 ms                     446%
==========================================  ============  ==============  =======================



.. Benchmark report ends.

Versioning
==========
We follow `Semantic Versioning <http://semver.org/spec/v1.0.0.html>`_.
The version X.Y.Z indicates:

* X is the major version (backward-incompatible),
* Y is the minor version (backward-compatible), and
* Z is the patch version (backward-compatible bug fix).
