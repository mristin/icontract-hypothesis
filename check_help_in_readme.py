#!/usr/bin/env python3

"""Check that the help snippets in the Readme coincide with the actual output."""
import os
import pathlib
import re
import subprocess
import sys

HELP_STARTS_RE = re.compile(r"^.. Help starts: (?P<command>.*)$")


def main() -> int:
    """Execute the main routine."""
    this_dir = pathlib.Path(os.path.realpath(__file__)).parent
    pth = this_dir / "README.rst"

    text = pth.read_text(encoding="utf-8")

    lines = text.splitlines()

    i = 0
    while i < len(lines):
        mtch = HELP_STARTS_RE.match(lines[i])
        if mtch:
            command = mtch.group("command")
            help_ends = ".. Help ends: {}".format(command)
            try:
                end_index = lines.index(help_ends, i)
            except ValueError:
                end_index = -1

            if end_index == -1:
                print(
                    "Could not find the end marker {!r} in the readme: {}".format(
                        help_ends, pth
                    ),
                    file=sys.stderr,
                )
                return -1

            expected = lines[i + 1 : end_index]

            command_parts = command.split(" ")
            if command_parts[0] in ["python", "python3"]:
                # We need to replace "python" with "sys.executable" on Windows as the environment
                # is not properly inherited.
                command_parts[0] = sys.executable

            proc = subprocess.Popen(
                command_parts,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding="utf-8",
            )
            output, err = proc.communicate()

            output_lines = output.splitlines()
            for i in range(len(output_lines)):
                if len(output_lines[i]) > 0:
                    output_lines[i] = "    " + output_lines[i]
            output_lines.insert(0, ".. code-block::")
            output_lines.insert(1, "")
            output_lines.append("")

            expected = [line.rstrip() for line in expected]
            output_lines = [line.rstrip() for line in output_lines]

            if output_lines != expected:
                print("Expected:", file=sys.stderr)
                for i, line in enumerate(expected):
                    if i >= len(output_lines) or line != output_lines[i]:
                        print("DIFF: {:2d}: {!r}".format(i, line))
                    else:
                        print("OK  : {:2d}: {!r}".format(i, line))

                print("Got:", file=sys.stderr)
                for i, line in enumerate(output_lines):
                    if i >= len(expected) or line != expected[i]:
                        print("DIFF: {:2d}: {!r}".format(i, line))
                    else:
                        print("OK  : {:2d}: {!r}".format(i, line))

                print("Got (raw output):\n", file=sys.stderr)
                print(output, file=sys.stderr)

                print("Got (raw error):\n", file=sys.stderr)
                for line in err.splitlines():
                    print("ERROR: {}".format(line))

                return -1

            i = end_index + 1

        else:
            i += 1


if __name__ == "__main__":
    sys.exit(main())
