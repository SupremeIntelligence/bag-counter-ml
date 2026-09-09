import json

import cv2
import numpy as np
import supervision as sv
from mmdet.apis import init_detector, inference_detector
import torch


CONFIG_PATH = "configs/rtmdet_bag.py"

DEFAULT_CHECKPOINT = (
    "models/"
    "rtmdet_bag.pth"
)

LINE_START = (200, 82) #((400, 234)) - новые данные, множитель разрешения для перевода к разрешению мака - 2x
LINE_END = (430, 153) #((860, 376)) - для координаты y еще - 70 для перевода к разрешению ролика
DEAD_ZONE_PX = 10
MOTION_EPS_PX = 1.0
REVERSE_CONFIRM_FRAMES = 10
FORWARD_CONFIRM_FRAMES = 10


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

def crossing_point(previous_point, current_point, previous_distance, current_distance):
    denominator = previous_distance - current_distance

    if abs(denominator) < 1e-8:
        return None

    alpha = previous_distance / denominator

    x0, y0 = previous_point
    x1, y1 = current_point

    x = x0 + alpha * (x1 - x0)
    y = y0 + alpha * (y1 - y0)

    return (x, y)

def point_on_segment(point, line_start, line_end, margin=0.0):
    px, py = point

    ax, ay = line_start
    bx, by = line_end

    ab_x = bx - ax
    ab_y = by - ay

    length_squared = ab_x**2 + ab_y**2

    if length_squared == 0:
        return False

    t = ((px - ax) * ab_x + (py - ay) * ab_y) / length_squared

    segment_length = np.sqrt(length_squared)

    margin_t = margin / segment_length

    return -margin_t <= t <= 1 + margin_t

def get_side(distance, dead_zone):
    if distance > dead_zone:
        return 1

    if distance < -dead_zone:
        return -1

    return 0


def process_video(
    input_path,
    output_path,
    result_path,
    checkpoint_path=DEFAULT_CHECKPOINT,
    det_thr=0.10,
):
    model = init_detector(
        CONFIG_PATH,
        checkpoint_path,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {input_path}"
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
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    tracker = sv.ByteTrack(
        track_activation_threshold=0.40,
        lost_track_buffer=75,
        minimum_matching_threshold=0.80,
        frame_rate=int(round(fps)),
        minimum_consecutive_frames=3,
    )

    last_stable_side = {}
    last_stable_point = {}
    last_stable_distance = {}

    previous_motion_distance = {}
    reverse_streak = 0
    forward_streak = 0
    reverse_motion_detected = False
    reverse_start_frame = None
    anomalies = []

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

        keep = scores >= det_thr

        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        detections = sv.Detections(
            xyxy=bboxes.astype(np.float32),
            confidence=scores.astype(np.float32),
            class_id=labels.astype(int),
        )

        tracked = tracker.update_with_detections(detections)
        frame_motion_samples = []

        if tracked.tracker_id is not None:
            for bbox, score, track_id in zip(tracked.xyxy, tracked.confidence, tracked.tracker_id):
                track_id = int(track_id)

                x1, y1, x2, y2 = bbox

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                point = (cx, cy)

                distance = signed_distance_to_line(point, LINE_START, LINE_END)
                side = get_side(distance, DEAD_ZONE_PX)

                previous_motion = previous_motion_distance.get(track_id)
                if previous_motion is not None:
                    delta_distance = distance - previous_motion

                    if abs(delta_distance) >= MOTION_EPS_PX:
                        frame_motion_samples.append(delta_distance)

                previous_motion_distance[track_id] = distance

                if side != 0:
                    previous_side = last_stable_side.get(track_id)
                    previous_point = last_stable_point.get(track_id)
                    previous_distance = last_stable_distance.get(track_id)

                    if (
                        previous_side is not None
                        and previous_side != side
                        and previous_point is not None
                        and previous_distance is not None
                    ):
                        cross_point = crossing_point(
                            previous_point,
                            point,
                            previous_distance,
                            distance,
                        )

                        if (
                            cross_point is not None
                            and point_on_segment(
                                cross_point,
                                LINE_START,
                                LINE_END,
                            )
                        ):
                            if previous_side == 1 and side == -1:
                                forward_count += 1

                                print(
                                    f"Frame {frame_number}: "
                                    f"ID {track_id} FORWARD"
                                )

                            elif previous_side == -1 and side == 1:
                                backward_count += 1

                                print(
                                    f"Frame {frame_number}: "
                                    f"ID {track_id} BACKWARD"
                                )

                    last_stable_side[track_id] = side
                    last_stable_point[track_id] = point
                    last_stable_distance[track_id] = distance

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
                    (x1i, max(y1i - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        if frame_motion_samples:
            median_motion = float(np.median(frame_motion_samples))

            #Реверс
            if median_motion > MOTION_EPS_PX:
                reverse_streak += 1
                forward_streak = 0

            #Вперед
            elif median_motion < -MOTION_EPS_PX:
                forward_streak += 1
                reverse_streak = 0

            #Ничего
            else:
                reverse_streak = 0
                forward_streak = 0

        if not reverse_motion_detected and reverse_streak >= REVERSE_CONFIRM_FRAMES:
            reverse_motion_detected = True
            reverse_start_frame = frame_number - REVERSE_CONFIRM_FRAMES + 1

            print(f"Frame {frame_number}: ANOMALY - reverse motion detected")
           
        if reverse_motion_detected and forward_streak >= FORWARD_CONFIRM_FRAMES:
            reverse_motion_detected = False

            anomaly = {
                "type": "reverse_motion",
                "start_frame": reverse_start_frame,
                "end_frame": frame_number,
                "start_time": round(reverse_start_frame / fps, 2),
                "end_time": round(frame_number / fps, 2),
            }

            anomalies.append(anomaly)
            print(f"Frame {frame_number}: Reverse motion ended")
            reverse_start_frame = None

        cv2.line(
            frame,
            LINE_START,
            LINE_END,
            (0, 0, 255),
            2,
        )

        net_count = forward_count - backward_count

        cv2.rectangle(
            frame,
            (5, 5),
            (155, 88),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            frame,
            f"Total bags: {net_count}",
            (10,30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2,
            cv2.LINE_AA,
        )
       
        cv2.putText(
            frame,
            f"Forward: {forward_count}",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            f"Backward: {backward_count}",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if reverse_motion_detected:
            anomaly_text = "ANOMALY: REVERSE MOTION"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1

            (text_width, text_height), baseline = cv2.getTextSize(
                anomaly_text,
                font,
                font_scale,
                thickness,
            )

            padding = 6
            x = width - text_width - 2 * padding - 10
            y = 15

            #Фон
            cv2.rectangle(
                frame,
                (x, y),
                (x + text_width + 2 * padding, y + text_height + 2 * padding),
                (0, 0, 0),
                -1,
            )

            #Граница
            cv2.rectangle(
                frame,
                (x, y),
                (x + text_width + 2 * padding, y + text_height + 2 * padding),
                (0, 0, 255),
                1,
            )

            #Текст
            cv2.putText(
                frame,
                anomaly_text,
                (x + padding, y + padding + text_height),
                font,
                font_scale,
                (0, 0, 255),
                thickness,
                cv2.LINE_AA,
            )

        frame_number += 1

        writer.write(frame)

        if frame_number % 250 == 0:
            print(
                f"Processed "
                f"{frame_number}/{total_frames}"
            )

    if reverse_motion_detected and reverse_start_frame is not None:
        end_frame = frame_number - 1
        anomalies.append(
        {
            "type": "reverse_motion",
            "start_frame": reverse_start_frame,
            "end_frame": end_frame,
            "start_time": round(reverse_start_frame / fps, 2),
            "end_time": round(end_frame / fps, 2),
        })

    cap.release()
    writer.release()

    net_count = forward_count - backward_count
   
    processing_result = {
    "total_bags": net_count,
    "forward": forward_count,
    "backward": backward_count,
    "anomalies": anomalies
    }

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(processing_result, file, indent=4, ensure_ascii=False)

    print()
    print("Done")
    print(f"Forward:  {forward_count}")
    print(f"Backward: {backward_count}")
    print(f"Net:      {net_count}")
    print(f"Output:   {output_path}")

    return processing_result