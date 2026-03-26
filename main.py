from src.cli import parse_args
from src.configs import DetectionConfig
from src.logging_config import configure_logging
from src.pipeline import TableVideoPipeline


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    with TableVideoPipeline(args, DetectionConfig()) as app:
        app.run()


if __name__ == "__main__":
    main()
