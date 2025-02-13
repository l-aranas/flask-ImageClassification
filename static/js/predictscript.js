document.addEventListener("DOMContentLoaded", function() {
    let images = document.querySelectorAll(".predict-image");
    let resultText = document.getElementById("predict-result");
    let loadingSpinner = document.getElementById("loading-spinner");
    let predictionModalElement = document.getElementById("predictionModal");

    // Ensure all required elements exist
    if (!images.length || !resultText || !loadingSpinner || !predictionModalElement) {
        console.error("Error: Required elements not found in predict.html.");
        return;
    }

    let predictionModal = new bootstrap.Modal(predictionModalElement);

    images.forEach(img => {
        img.addEventListener("click", async function() {
            images.forEach(i => i.classList.remove("selected"));
            this.classList.add("selected");

            let filename = this.getAttribute("data-filename");
            loadingSpinner.style.display = "block";
            resultText.innerText = "";

            try {
                let response = await fetch(`/predict_image?filename=${filename}`);
                let result = await response.json();
                loadingSpinner.style.display = "none";

                if (result.error) {
                    resultText.innerText = "❌ Error: " + result.error;
                    resultText.style.color = "red";
                } else {
                    resultText.innerText = `✅ Predicted Class: ${result.prediction}`;
                    resultText.style.color = "green";
                }

                predictionModal.show();
            } catch (error) {
                loadingSpinner.style.display = "none";
                resultText.innerText = "⚠️ Prediction failed!";
                resultText.style.color = "red";
                predictionModal.show();
            }
        });
    });
});
