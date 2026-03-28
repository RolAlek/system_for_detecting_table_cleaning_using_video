import logging
import sys

from src.cli import parse_args
from src.configs import DetectionConfig
from src.logging_config import configure_logging
from src.pipeline import TableVideoPipeline

logger = logging.getLogger(__name__)


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    try:
        with TableVideoPipeline(args, DetectionConfig()) as app:
            app.run()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user (Ctrl+C).")
        sys.exit(130)


if __name__ == "__main__":
    main()
