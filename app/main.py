from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, UploadFile


app = FastAPI(
    title="Bag Counter API",
)

STORAGE_DIR = Path("storage")
STORAGE_DIR.mkdir(exist_ok=True)


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/videos")
async def upload_video(file: UploadFile = File(...)):
    video_id = str(uuid4())

    suffix = Path(file.filename).suffix

    input_path = STORAGE_DIR / f"{video_id}{suffix}"

    with open(input_path, "wb") as output_file:
        while chunk := await file.read(1024 * 1024):
            output_file.write(chunk)

    return {
        "video_id": video_id,
        "filename": file.filename,
        "path": str(input_path),
    }