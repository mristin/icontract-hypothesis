icontract-hypothesis
====================

.. image:: https://travis-ci.com/mristin/icontract-hypothesis.svg?branch=master
    :target: https://travis-ci.com/mristin/icontract-hypothesis

.. image:: https://coveralls.io/repos/github/mristin/icontract-hypothesis/badge.svg?branch=master
    :target: https://coveralls.io/github/mristin/icontract-hypothesis

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
* As a command-line tool or integrate it with your IDE. This allows you to
  automatically test functions while you develop and use it in your continuous
  integration and
* As a ghostwriter utility giving you a starting point for your more elaborate
  Hypothesis test code.

Since the contracts live close to the code, evolving the code automatically
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

**Infer strategies**. Second, you can automatically infer the strategies and test the function:

.. code-block:: python

    >>> import icontract
    >>> import icontract_hypothesis

    >>> @icontract.require(lambda x: x > 0)
    ... @icontract.ensure(lambda result: result > 0)
    ... def some_func(x: int) -> int:
    ...     return x - 1000

    >>> icontract_hypothesis.test_with_inferred_strategies(some_func)
    Traceback (most recent call last):
        ...
    icontract.errors.ViolationError: File <doctest README.rst[10]>, line 2 in <module>:
    result > 0: result was -999

Which approach to use depends on strategy inference. If the strategies can be
inferred, prefer the second. However, if no strategies could be inferred and
rejection sampling fails, you need to resort to the first approach and come up
with appropriate search strategies manually.

Use ``icontract_hypothesis.infer_strategies`` to inspect which strategies were
inferred:

.. code-block:: python

    >>> import math

    >>> import icontract
    >>> import icontract_hypothesis

    >>> @icontract.require(lambda x: x > 0)
    ... @icontract.require(lambda x: x > math.sqrt(x))
    ... def some_func(x: float) -> int:
    ...     pass

    >>> icontract_hypothesis.infer_strategies(some_func)
    {'x': floats(min_value=0, exclude_min=True).filter(lambda x: x > math.sqrt(x))}

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
Writing property-based tests by hand is tedious. To that end, we implemented
a ghostwriter utility ``pyicontract-hypothesis ghostwrite`` that comes up with
a first draft based on pre-conditions that you manually refine later:

.. Help starts: pyicontract-hypothesis ghostwrite --help
.. code-block::

    usage: pyicontract-hypothesis ghostwrite [-h] -m MODULE [-o OUTPUT]
                                             [--explicit {strategies,strategies-and-assumes}]
                                             [--bare] [-i [INCLUDE [INCLUDE ...]]]
                                             [-e [EXCLUDE [EXCLUDE ...]]]

    optional arguments:
      -h, --help            show this help message and exit
      -m MODULE, --module MODULE
                            Module to process
      -o OUTPUT, --output OUTPUT
                            Path to the file where the output should be written.

                            If '-', writes to STDOUT.
      --explicit {strategies,strategies-and-assumes}
                            Write the inferred strategies explicitly

                            This is practical if you want to tune and
                            refine the strategies and just want to use
                            ghostwriting as a starting point.

                            Mind that pyicontract-hypothesis does not
                            automatically fix imports as this is
                            usually project-specific. You have to fix imports
                            manually after the ghostwriting.

                            Possible levels of explicitness:
                            * strategies: Write the inferred strategies

                            * strategies-and-assumes: Write out both the inferred strategies
                                   and the preconditions
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

The examples of ghostwritten tests is available at:
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
  (``int``, ``float``, ``datetime.date`` *etc*.) and regular expressions.

* Pre-conditions which could not be matched, but operate on a single argument
  are inferred based on the type hint and composed with Hypothesis
  ``FilteredStrategy``.

* The remainder of the pre-conditions are enforced using ``hypothesis.assume``,
  basically falling back to rejection sampling as the last resort.

Classes
~~~~~~~
Hypothesis automatically builds composite input arguments (classes, dataclasses,
named tuples *etc*.). If your class enforces pre-conditions in the constructor
method (``__init__``), make sure that it inherits from ``icontract.DBC``.

That way icontract-hypothesis will use
`hypothesis.strategies.register_type_strategy <https://hypothesis.readthedocs.io/en/latest/data.html#hypothesis.strategies.register_type_strategy>`_
to register your class with Hypothesis and consider pre-conditions when building
its instances.

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

* Neither of the two handles behavioral sub-typing correctly
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
We used Python 3.8.5, icontract 2.4.0, deal 4.4.0 and dpcontracts 0.6.0.

The following tables summarize the results.

Benchmarking Hypothesis testing:


Argument count: 1

==========================================  ============  ==============  =======================
Case                                          Total time    Time per run    Relative time per run
==========================================  ============  ==============  =======================
`benchmark_icontract_inferred_strategies`         0.53 s        52.92 ms                     100%
`benchmark_icontract_assume_preconditions`        0.88 s        88.21 ms                     167%
`benchmark_dpcontracts`                           1.19 s       118.99 ms                     225%
`benchmark_deal`                                  0.90 s        90.43 ms                     171%
==========================================  ============  ==============  =======================

Argument count: 2

==========================================  ============  ==============  =======================
Case                                          Total time    Time per run    Relative time per run
==========================================  ============  ==============  =======================
`benchmark_icontract_inferred_strategies`         0.68 s        68.14 ms                     100%
`benchmark_icontract_assume_preconditions`        1.86 s       186.15 ms                     273%
`benchmark_dpcontracts`                           2.31 s       230.61 ms                     338%
`benchmark_deal`                                  1.95 s       195.42 ms                     287%
==========================================  ============  ==============  =======================

Argument count: 3

==========================================  ============  ==============  =======================
Case                                          Total time    Time per run    Relative time per run
==========================================  ============  ==============  =======================
`benchmark_icontract_inferred_strategies`         0.79 s        78.85 ms                     100%
`benchmark_icontract_assume_preconditions`        3.54 s       354.22 ms                     449%
`benchmark_dpcontracts`                           4.45 s       444.93 ms                     564%
`benchmark_deal`                                  3.44 s       344.00 ms                     436%
==========================================  ============  ==============  =======================



.. Benchmark report ends.

Versioning
==========
We follow `Semantic Versioning <http://semver.org/spec/v1.0.0.html>`_.
The version X.Y.Z indicates:

* X is the major version (backward-incompatible),
* Y is the minor version (backward-compatible), and
* Z is the patch version (backward-compatible bug fix).
