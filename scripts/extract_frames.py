from pathlib import Path
import cv2

VIDEO_PATH = Path("input.mp4")
OUTPUT_DIR = Path("data/raw_frames")
NUM_FRAMES = 200

def extract_frames(video_path: Path, output_dir: Path, num_frames: int,) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        raise RuntimeError("Не удалось определить число кадров")

    duration = total_frames / fps

    print(f"Всего кадров: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"Длительность: {duration:.2f} сек")
    print(f"Извлекаем кадров: {num_frames}")

    #Равномерно распределяем кадры от начала до конца видео
    frame_indices = [round(i * (total_frames - 1) / (num_frames - 1))for i in range(num_frames)]

    for index, frame_number in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        success, frame = cap.read()

        if not success:
            print(f"Не удалось прочитать frame {frame_number}")
            continue

        timestamp = frame_number / fps

        filename = (f"frame_{frame_number:06d}_"f"t{timestamp:07.2f}.jpg")

        output_path = output_dir / filename

        cv2.imwrite(str(output_path), frame)

        print(
            f"[{index + 1:03d}/{num_frames}] "
            f"frame={frame_number:06d}, "
            f"time={timestamp:7.2f}s "
            f"-> {filename}"
        )

    cap.release()
    print(f"\nКадры сохранены в: {output_dir}")


if __name__ == "__main__":
    extract_frames(
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        num_frames=NUM_FRAMES,
    )