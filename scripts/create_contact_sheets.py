from pathlib import Path

import cv2
import numpy as np


INPUT_DIR = Path("data/raw_frames")
OUTPUT_DIR = Path("data/contact_sheets")

COLS = 5
ROWS = 5
FRAMES_PER_SHEET = COLS * ROWS
THUMB_WIDTH = 320
THUMB_HEIGHT = 180


def create_contact_sheets() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(INPUT_DIR.glob("*.jpg"))

    if not image_paths:
        raise RuntimeError(f"В {INPUT_DIR} нет изображений")

    for sheet_index in range(0, len(image_paths), FRAMES_PER_SHEET):
        batch = image_paths[sheet_index : sheet_index + FRAMES_PER_SHEET]

        cells = []

        for image_path in batch:
            image = cv2.imread(str(image_path))

            if image is None:
                continue

            image = cv2.resize(image,(THUMB_WIDTH, THUMB_HEIGHT))
            label = image_path.stem

            cv2.rectangle(image,(0, 0), (THUMB_WIDTH, 24), (0, 0, 0),thickness=-1,)
            cv2.putText(image,label,(5, 17),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255, 255, 255),1,cv2.LINE_AA,)

            cells.append(image)

        #Пустые ячейки на последней странице
        while len(cells) < FRAMES_PER_SHEET:
            cells.append(np.zeros((THUMB_HEIGHT, THUMB_WIDTH, 3),dtype=np.uint8,))

        rows = []

        for row in range(ROWS):
            start = row * COLS
            end = start + COLS

            rows.append(np.hstack(cells[start:end]))

        sheet = np.vstack(rows)

        output_path = (OUTPUT_DIR /f"sheet_{sheet_index // FRAMES_PER_SHEET + 1:02d}.jpg")

        cv2.imwrite(str(output_path), sheet)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    create_contact_sheets()