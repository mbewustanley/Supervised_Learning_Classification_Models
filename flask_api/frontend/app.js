const form = document.getElementById("predictionForm");
const resultBox = document.getElementById("result");

document.getElementById("predictionForm").addEventListener("submit", async (e) => {
    e.preventDefault();

    const model = document.getElementById("model").value;

    const payload = {
        model: model,
        fLength: parseFloat(document.getElementById("fLength").value),
        fWidth: parseFloat(document.getElementById("fWidth").value),
        fSize: parseFloat(document.getElementById("fSize").value),
        fConc: parseFloat(document.getElementById("fConc").value),
        fConc1: parseFloat(document.getElementById("fConc1").value),
        fAsym: parseFloat(document.getElementById("fAsym").value),
        fM3Long: parseFloat(document.getElementById("fM3Long").value),
        fM3Trans: parseFloat(document.getElementById("fM3Trans").value),
        fAlpha: parseFloat(document.getElementById("fAlpha").value),
        fDist: parseFloat(document.getElementById("fDist").value)
    };

    try {
        const response = await fetch("http://localhost:5000/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(payload)
        });

        const result = await response.json();
        document.getElementById("result").innerText =
            `Prediction: ${result.prediction} (Model: ${result.model_used})`;

    } catch (error) {
        document.getElementById("result").innerText = "Error connecting to backend";
    }
});
