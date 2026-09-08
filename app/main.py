from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from app.services.jobs import create_job, get_job

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

@app.post("/jobs")
def start_job(video_id: str):
    matching_files = list(STORAGE_DIR.glob(f"{video_id}.*"))

    if not matching_files:
        raise HTTPException(status_code=404, detail="Video not found")

    input_path = matching_files[0]
    job = create_job(input_path)

    return {"job_id": job["job_id"], "status": job["status"]}


@app.get("/jobs/{job_id}")
def job_status(job_id: str):

    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job

@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    job = get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail="Job is not completed")

    return job["result"]


@app.get("/jobs/{job_id}/video")
def job_video(job_id: str):
    job = get_job(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "completed":
        raise HTTPException(status_code=409, detail="Job is not completed")

    output_path = Path(job["output_path"])

    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Processed video not found")

    return FileResponse(path=output_path, media_type="video/mp4", filename=output_path.name)