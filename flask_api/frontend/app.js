const form = document.getElementById("predictionForm");
const resultBox = document.getElementById("result");

form.addEventListener("submit", async (e) => {
    e.preventDefault();

    // Collect form data
    const formData = new FormData(form);
    const data = {};

    formData.forEach((value, key) => {
        data[key] = parseFloat(value);
    });

    try {
        // Send request to backend
        const response = await fetch("http://localhost:5000/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(data)
        });

        const result = await response.json();

        // Display prediction
        resultBox.style.display = "block";
        resultBox.style.background = result.prediction === 1 ? "#28a745" : "#d9534f";
        resultBox.style.color = "white";

        resultBox.innerText =
            result.prediction === 1
            ? "Prediction: GAMMA (1)"
            : "Prediction: ALPHA (0)";

    } catch (error) {
        console.error("Prediction error:", error);

        resultBox.style.display = "block";
        resultBox.style.background = "#6c757d";
        resultBox.style.color = "white";
        resultBox.innerText = "Error contacting prediction server.";
    }
});
