document.getElementById("predictBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("fileInput");
  const outputBox = document.getElementById("outputBox");

  if (!fileInput.files.length) {
    outputBox.textContent = "⚠️ Please choose an image.";
    return;
  }

  const formData = new FormData();
  formData.append("image", fileInput.files[0]);

  outputBox.textContent = "⏳ Running prediction...";

  try {
    const res = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    outputBox.textContent = data.output || "No output received.";
  } catch (err) {
    outputBox.textContent = "❌ Error: " + err.message;
  }
});
