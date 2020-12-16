import io
import unittest

from icontract_hypothesis.pyicontract_hypothesis import main


class TestParsingOfParameters(unittest.TestCase):
    def test_no_command(self) -> None:
        argv = ["-m", "some_module"]

        stdout, stderr = io.StringIO(), io.StringIO()

        main.run(argv=argv, stdout=stdout, stderr=stderr)

        self.assertEqual("", stdout.getvalue())
        self.assertEqual(
            """\
usage: pyicontract-hypothesis [-h] {test,ghostwrite} ...
pyicontract-hypothesis: error: argument command: invalid choice: 'some_module' (choose from 'test', 'ghostwrite')
""",
            stderr.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
