const videoInput = document.getElementById("video-input");
const processButton = document.getElementById("process-button");
const statusText = document.getElementById("status");
const resultBlock = document.getElementById("result");

const totalBags = document.getElementById("total-bags");
const forward = document.getElementById("forward");
const backward = document.getElementById("backward");
const anomalies = document.getElementById("anomalies");

const downloadLink = document.getElementById("download-link");


videoInput.addEventListener("change", () => {
    if (videoInput.files[0]) {
        statusText.textContent =
            `Selected: ${videoInput.files[0].name}`;

        setStatusClass("");
    } else {
        statusText.textContent = "Ready";
        setStatusClass("");
    }
});


processButton.addEventListener("click", async () => {
    const file = videoInput.files[0];

    if (!file) {
        statusText.textContent = "Choose a video first";
        setStatusClass("error");
        return;
    }

    processButton.disabled = true;
    processButton.textContent = "Processing...";
    resultBlock.hidden = true;

    try {
        statusText.textContent = "Uploading video...";
        setStatusClass("processing");

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

        const uploadData =
            await uploadResponse.json();

        statusText.textContent =
            "Starting processing...";

        const jobResponse = await fetch(
            `/jobs?video_id=${encodeURIComponent(
                uploadData.video_id
            )}`,
            {
                method: "POST",
            }
        );

        if (!jobResponse.ok) {
            throw new Error(
                "Could not start processing"
            );
        }

        const jobData =
            await jobResponse.json();

        await waitForJob(jobData.job_id);

    } catch (error) {
        statusText.textContent = error.message;
        setStatusClass("error");

        processButton.disabled = false;
        processButton.textContent =
            "Process video";
    }
});


async function waitForJob(jobId) {
    while (true) {
        const response = await fetch(
            `/jobs/${jobId}`
        );

        if (!response.ok) {
            throw new Error(
                "Could not get job status"
            );
        }

        const job = await response.json();

        if (job.status === "queued") {
            statusText.textContent =
                "Status: queued";

            setStatusClass("processing");
        }

        if (job.status === "processing") {
            statusText.textContent =
                "Status: processing";

            setStatusClass("processing");
        }

        if (job.status === "completed") {
            showResult(
                jobId,
                job.result
            );

            processButton.disabled = false;
            processButton.textContent =
                "Process another video";

            return;
        }

        if (job.status === "failed") {
            throw new Error(
                job.error ||
                "Processing failed"
            );
        }

        await new Promise(
            resolve =>
                setTimeout(resolve, 2000)
        );
    }
}


function showResult(jobId, result) {
    totalBags.textContent =
        result.total_bags;

    forward.textContent =
        result.forward;

    backward.textContent =
        result.backward;

    anomalies.textContent =
        result.anomalies.length;

    downloadLink.href =
        `/jobs/${jobId}/video`;

    statusText.textContent =
        "Processing completed";

    setStatusClass("success");

    resultBlock.hidden = false;
}


function setStatusClass(className) {
    statusText.className = className;
}