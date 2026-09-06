from pathlib import Path

import cv2


VIDEO_PATH = Path("input.mp4")
OUTPUT_DIR = Path("data/raw_frames")

INTERVALS = [
    (90, 115),   #остановка + реверс
    (165, 185),  #перемещение конвейера
]

STEP_SECONDS = 0.5


def extract_interval(cap: cv2.VideoCapture, fps: float,
    start_sec: float,
    end_sec: float,
) -> None:
    timestamp = start_sec

    while timestamp <= end_sec:
        frame_number = round(timestamp * fps)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()

        if success:
            filename = (f"special_frame_{frame_number:06d}_" f"t{timestamp:07.2f}.jpg")
            cv2.imwrite(str(OUTPUT_DIR / filename), frame)

        timestamp += STEP_SECONDS


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видео")

    fps = cap.get(cv2.CAP_PROP_FPS)

    for start_sec, end_sec in INTERVALS:
        extract_interval(cap, fps, start_sec, end_sec)

    cap.release()


if __name__ == "__main__":
    main()