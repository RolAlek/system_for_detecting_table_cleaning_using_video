import logging
import sys


def configure_logging(*, verbose: bool = False) -> None:
    """
    Messages to the user through logging: levels, uniform format, stderr.
    verbose=True — DEBUG (more noise from libraries if they log).
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stderr,
        force=True,
    )
