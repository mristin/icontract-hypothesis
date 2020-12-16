"""Let mypy check the exhaustive matches."""
from typing import NoReturn

# Based on https://github.com/python/mypy/issues/5818


def assert_never(something: NoReturn) -> NoReturn:
    """Enforce exhaustive matching at mypy time."""
    assert False, "Unhandled type: {}".format(type(something).__name__)
