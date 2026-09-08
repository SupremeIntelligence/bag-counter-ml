import argparse
from app.cv.pipeline import DEFAULT_CHECKPOINT, process_video


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="input.mp4",
    )

    parser.add_argument(
        "--output",
        default="counted.mp4",
    )

    parser.add_argument(
        "--result",
        default="counted_results.json",
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--det-thr",
        type=float,
        default=0.10,
    )

    args = parser.parse_args()

    process_video(
        input_path=args.input,
        output_path=args.output,
        result_path=args.result,
        checkpoint_path=args.checkpoint,
        det_thr=args.det_thr,
    )


if __name__ == "__main__":
    main()