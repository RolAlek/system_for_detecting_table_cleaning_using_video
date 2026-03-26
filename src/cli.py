import argparse


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detection of events at the table by video (people detector + table zone).",
    )
    p.add_argument("--video", type=str, required=True, help="Path to the input video")
    p.add_argument(
        "--output",
        type=str,
        default="output.mp4",
        help="Path to the output video with visualization",
    )
    p.add_argument(
        "--report",
        type=str,
        default="report.txt",
        help="Text report with statistics",
    )
    p.add_argument(
        "--problem-frame",
        type=str,
        default="problem_frame.png",
        help="Save frame with problematic/illustrative moment (fraction of the video length)",
    )
    p.add_argument(
        "--problem-at",
        type=float,
        default=0.35,
        help="Fraction of the video length (0..1) on which to save problem_frame",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose log (DEBUG level)",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Do not show the frame progress bar (e.g. CI or redirected stderr)",
    )
    return p.parse_args()
