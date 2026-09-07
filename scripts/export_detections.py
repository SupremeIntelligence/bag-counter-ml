import argparse
import json
from pathlib import Path

import cv2
from mmdet.apis import init_detector, inference_detector


CONFIG_PATH = "configs/rtmdet_bag.py"
DEFAULT_CHECKPOINT = (
    "work_dirs/rtmdet_bag/"
    "best_coco_bbox_mAP_epoch_45.pth"
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="input.mp4",
        help="Path to input video",
    )

    parser.add_argument(
        "--output",
        default="detections.json",
        help="Path to output JSON",
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.05,
        help="Minimum confidence to save",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    model = init_detector(
        CONFIG_PATH,
        args.checkpoint,
        device="cuda:0",
    )

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {input_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output = {
        "video": {
            "path": str(input_path),
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
        },
        "checkpoint": str(args.checkpoint),
        "score_threshold": args.score_thr,
        "frames": [],
    }

    frame_number = 0

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        result = inference_detector(
            model,
            frame,
        )

        instances = result.pred_instances.cpu()

        bboxes = instances.bboxes.numpy()
        scores = instances.scores.numpy()
        labels = instances.labels.numpy()

        detections = []

        for bbox, score, label in zip(
            bboxes,
            scores,
            labels,
        ):
            if score < args.score_thr:
                continue

            x1, y1, x2, y2 = bbox.tolist()

            detections.append(
                {
                    "bbox": [
                        float(x1),
                        float(y1),
                        float(x2),
                        float(y2),
                    ],
                    "score": float(score),
                    "label": int(label),
                }
            )

        output["frames"].append(
            {
                "frame_number": frame_number,
                "timestamp": frame_number / fps,
                "detections": detections,
            }
        )

        frame_number += 1

        if frame_number % 250 == 0:
            print(
                f"Processed "
                f"{frame_number}/{total_frames}"
            )

    cap.release()

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print()
    print(f"Done: {output_path}")
    print(f"Frames processed: {frame_number}")


if __name__ == "__main__":
    main()