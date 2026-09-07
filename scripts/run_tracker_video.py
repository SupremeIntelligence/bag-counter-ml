import argparse
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
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
        default="tracked.mp4",
        help="Path to output video",
    )

    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--det-thr",
        type=float,
        default=0.10,
        help="Minimum detector score passed to ByteTrack",
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

    tracker = sv.ByteTrack(
        track_activation_threshold=0.40,
        lost_track_buffer=75,
        minimum_matching_threshold=0.80,
        frame_rate=int(round(fps)),
        minimum_consecutive_frames=3,
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

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

        keep = scores >= args.det_thr

        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        detections = sv.Detections(
            xyxy=bboxes.astype(np.float32),
            confidence=scores.astype(np.float32),
            class_id=labels.astype(int),
        )

        tracked = tracker.update_with_detections(
            detections
        )

        if tracked.tracker_id is not None:
            for bbox, score, track_id in zip(
                tracked.xyxy,
                tracked.confidence,
                tracked.tracker_id,
            ):
                x1, y1, x2, y2 = bbox.astype(int)

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                text = (
                    f"ID {int(track_id)} "
                    f"{float(score):.2f}"
                )

                cv2.putText(
                    frame,
                    text,
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        frame_number += 1

        cv2.putText(
            frame,
            f"Frame: {frame_number}/{total_frames}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        writer.write(frame)

        if frame_number % 250 == 0:
            print(
                f"Processed "
                f"{frame_number}/{total_frames}"
            )

    cap.release()
    writer.release()

    print()
    print(f"Done: {output_path}")


if __name__ == "__main__":
    main()