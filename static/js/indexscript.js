document.addEventListener("DOMContentLoaded", function() {
    let uploadForm = document.getElementById('upload-form');
    let trainButton = document.getElementById('train-button');
    let uploadStatus = document.getElementById('upload-status');
    let trainStatus = document.getElementById('train-status');
    let predictLink = document.getElementById('predict-link');
    let predictSection = document.getElementById('predict-section');

    if (!uploadForm || !trainButton) {
        console.error("Error: Required elements not found in the DOM.");
        return;
    }

    uploadForm.onsubmit = async function(event) {
        event.preventDefault();
        let formData = new FormData(this);
        let response = await fetch('/upload', { method: 'POST', body: formData });
        let result = await response.json();
        uploadStatus.innerText = result.message;
        checkPredictAvailability();
    };

    trainButton.onclick = async function() {
        trainStatus.innerText = "Training started...";
        let response = await fetch('/train', { method: 'POST' });
        let result = await response.json();
        trainStatus.innerText = "Training complete!";
        checkPredictAvailability();

        if (result.message === "Training complete") {
            setTimeout(() => {
                if (confirm("Training is complete! Do you want to go to the prediction page?")) {
                    window.location.href = predictLink.href;
                }
            }, 500);
        }
    };

    async function checkPredictAvailability() {
        let response = await fetch('/check_model_status');
        let result = await response.json();
        if (result.ready) {
            predictSection.style.display = 'block';
        }
    }

    checkPredictAvailability();
});
