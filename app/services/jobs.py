from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4
import json

from app.cv.pipeline import process_video


executor = ThreadPoolExecutor(max_workers=1)
JOBS_PATH = Path("storage") / "jobs.json"

def load_jobs():
    if not JOBS_PATH.exists():
        return {}

    with open(JOBS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def save_jobs():
    with open(JOBS_PATH, "w", encoding="utf-8") as file:
        json.dump(jobs, file, indent=4, ensure_ascii=False)

jobs = load_jobs()

def create_job(input_path):
    job_id = str(uuid4())

    output_path = Path("storage") / f"{job_id}_counted.mp4"
    result_path = Path("storage") / f"{job_id}_results.json"

    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "result_path": str(result_path),
        "result": None,
        "error": None,
    }

    save_jobs()

    executor.submit(run_job, job_id, input_path, output_path, result_path)

    return jobs[job_id]

def run_job(job_id, input_path, output_path, result_path):

    jobs[job_id]["status"] = "processing"
    save_jobs()

    try:
        result = process_video(input_path=input_path, output_path=output_path, result_path=result_path)

        jobs[job_id]["result"] = result
        jobs[job_id]["status"] = "completed"
        save_jobs()

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        save_jobs()

def get_job(job_id):
    return jobs.get(job_id)

def recover_interrupted_jobs():
    changed = False

    for job in jobs.values():
        if job["status"] in ("queued", "processing"):
            job["status"] = "failed"
            job["error"] = "Job was interrupted by server restart"
            changed = True
    if changed:
        save_jobs()

recover_interrupted_jobs()