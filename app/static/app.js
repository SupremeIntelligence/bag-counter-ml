const videoInput = document.getElementById("video-input");
const processButton = document.getElementById("process-button");
const statusText = document.getElementById("status");
const resultBlock = document.getElementById("result");

const totalBags = document.getElementById("total-bags");
const forward = document.getElementById("forward");
const backward = document.getElementById("backward");
const anomalies = document.getElementById("anomalies");

const downloadLink = document.getElementById("download-link");


processButton.addEventListener("click", async () => {
    const file = videoInput.files[0];

    if (!file) {
        statusText.textContent = "Choose a video first";
        return;
    }

    processButton.disabled = true;
    resultBlock.hidden = true;

    try {
        statusText.textContent = "Uploading video...";

        const formData = new FormData();
        formData.append("file", file);

        const uploadResponse = await fetch(
            "/videos",
            {
                method: "POST",
                body: formData,
            }
        );

        if (!uploadResponse.ok) {
            throw new Error("Video upload failed");
        }

        const uploadData = await uploadResponse.json();

        statusText.textContent = "Starting processing...";

        const jobResponse = await fetch(
            `/jobs?video_id=${encodeURIComponent(uploadData.video_id)}`,
            {
                method: "POST",
            }
        );

        if (!jobResponse.ok) {
            throw new Error("Could not start processing");
        }

        const jobData = await jobResponse.json();

        await waitForJob(jobData.job_id);

    } catch (error) {
        statusText.textContent = error.message;
        processButton.disabled = false;
    }
});


async function waitForJob(jobId) {
    while (true) {
        const response = await fetch(
            `/jobs/${jobId}`
        );

        if (!response.ok) {
            throw new Error("Could not get job status");
        }

        const job = await response.json();

        statusText.textContent = `Status: ${job.status}`;

        if (job.status === "completed") {
            showResult(
                jobId,
                job.result
            );

            processButton.disabled = false;
            return;
        }

        if (job.status === "failed") {
            throw new Error(
                job.error || "Processing failed"
            );
        }

        await new Promise(
            resolve => setTimeout(resolve, 2000)
        );
    }
}


function showResult(jobId, result) {
    totalBags.textContent = result.total_bags;
    forward.textContent = result.forward;
    backward.textContent = result.backward;
    anomalies.textContent = result.anomalies.length;

    downloadLink.href = `/jobs/${jobId}/video`;

    statusText.textContent = "Processing completed";

    resultBlock.hidden = false;
}