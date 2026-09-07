import argparse
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from mmdet.apis import init_detector, inference_detector
import torch


CONFIG_PATH = "configs/rtmdet_bag.py"

DEFAULT_CHECKPOINT = (
    "work_dirs/rtmdet_bag/"
    "best_coco_bbox_mAP_epoch_45.pth"
)

LINE_START = (160, 70) # (160, 70) ((320, 140 + 70)) - новые данные, множитель разрешения для перевода к разрешению мака - 2x
LINE_END = (470, 165) #(470, 165) ((940, 330+70)) - для координаты y еще - 70 для перевода к разрешению ролика
DEAD_ZONE_PX = 10


def signed_distance_to_line(point, line_start, line_end):
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    dx = x2 - x1
    dy = y2 - y1

    numerator = dx * (py - y1) - dy * (px - x1)
    denominator = np.hypot(dx, dy)

    if denominator == 0:
        raise ValueError("Counting line has zero length")

    return numerator / denominator


def get_side(distance, dead_zone):
    if distance > dead_zone:
        return 1

    if distance < -dead_zone:
        return -1

    return 0


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
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--det-thr",
        type=float,
        default=0.10,
    )

    args = parser.parse_args()

    model = init_detector(
        CONFIG_PATH,
        args.checkpoint,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    cap = cv2.VideoCapture(args.input)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {args.input}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )
    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )
    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    tracker = sv.ByteTrack(
        track_activation_threshold=0.40,
        lost_track_buffer=75,
        minimum_matching_threshold=0.80,
        frame_rate=int(round(fps)),
        minimum_consecutive_frames=3,
    )

    last_stable_side = {}

    forward_count = 0
    backward_count = 0

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

        tracked = tracker.update_with_detections(detections)

        if tracked.tracker_id is not None:
            for bbox, score, track_id in zip(tracked.xyxy, tracked.confidence, tracked.tracker_id):
                track_id = int(track_id)

                x1, y1, x2, y2 = bbox

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                point = (cx, cy)

                distance = signed_distance_to_line(point, LINE_START, LINE_END)

                side = get_side(distance, DEAD_ZONE_PX)

                if side != 0:
                    previous_side = last_stable_side.get(track_id)

                    if previous_side is not None and previous_side != side:
                        if previous_side == 1 and side == -1:
                            forward_count += 1
                            print(f"Frame {frame_number}: " f"ID {track_id} FORWARD")

                        elif (previous_side == -1 and side == 1):
                            backward_count += 1
                            print(f"Frame {frame_number}: " f"ID {track_id} BACKWARD")

                    last_stable_side[track_id] = side

                x1i, y1i, x2i, y2i = map(int, bbox)

                cv2.rectangle(
                    frame,
                    (x1i, y1i),
                    (x2i, y2i),
                    (0, 255, 0),
                    2,
                )

                cv2.circle(
                    frame,
                    (int(cx), int(cy)),
                    4,
                    (0, 255, 255),
                    -1,
                )

                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (
                        x1i,
                        max(y1i - 8, 20),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        cv2.line(
            frame,
            LINE_START,
            LINE_END,
            (0, 0, 255),
            2,
        )

        net_count = forward_count - backward_count

        cv2.putText(
            frame,
            f"Forward: {forward_count}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Backward: {backward_count}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Net: {net_count}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        frame_number += 1

        writer.write(frame)

        if frame_number % 250 == 0:
            print(
                f"Processed "
                f"{frame_number}/{total_frames}"
            )

    cap.release()
    writer.release()

    print()
    print("Done")
    print(f"Forward:  {forward_count}")
    print(f"Backward: {backward_count}")
    print(
        f"Net:      "
        f"{forward_count - backward_count}"
    )
    print(f"Output:   {args.output}")


if __name__ == "__main__":
    main()