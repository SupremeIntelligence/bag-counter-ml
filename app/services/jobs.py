from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from app.cv.pipeline import process_video


executor = ThreadPoolExecutor(max_workers=1)
jobs = {}

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

    executor.submit(run_job, job_id, input_path, output_path, result_path)

    return jobs[job_id]


def run_job(job_id, input_path, output_path, result_path):
    jobs[job_id]["status"] = "processing"

    try:
        result = process_video(input_path=input_path, output_path=output_path, result_path=result_path)

        jobs[job_id]["result"] = result
        jobs[job_id]["status"] = "completed"

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


def get_job(job_id):
    return jobs.get(job_id)