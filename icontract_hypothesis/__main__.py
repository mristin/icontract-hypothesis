"""Link to the ``pyicontract_hypothesis`` so that you can execute it with ``python`` CLI."""

import sys

import icontract_hypothesis.pyicontract_hypothesis.main

if __name__ == "__main__":
    sys.exit(
        icontract_hypothesis.pyicontract_hypothesis.main.run(
            argv=sys.argv[1:],
            stdout=sys.stdout,
            stderr=sys.stderr,
            prog="icontract_hypothesis",
        )
    )
