import React, { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [fileUrl, setFileUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setResult(null);
    if (selectedFile) {
      setFileUrl(URL.createObjectURL(selectedFile));
    } else {
      setFileUrl(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/predict/", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setResult({ error: "Prediction failed." });
    }
    setLoading(false);
  };

  return (
    <div style={{ minHeight: "100vh", background: "#f7f8fa", padding: 0, margin: 0 }}>
      <div style={{ maxWidth: 600, margin: "48px auto", background: "#fff", borderRadius: 16, boxShadow: "0 4px 24px rgba(0,0,0,0.08)", padding: 32 }}>
        <h1 style={{ textAlign: "center", color: "#2d3748", marginBottom: 24 }}>Diabetic Retinopathy Classifier</h1>
        <form onSubmit={handleSubmit} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
          <label htmlFor="file-upload" style={{
            background: "#edf2f7",
            border: "2px dashed #a0aec0",
            borderRadius: 8,
            padding: "32px 24px",
            cursor: "pointer",
            width: 320,
            textAlign: "center",
            color: file ? "#2b6cb0" : "#718096",
            fontWeight: 500,
            fontSize: 16
          }}>
            {file ? `Selected: ${file.name}` : "Click or drag an image here to upload"}
            <input
              id="file-upload"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              style={{ display: "none" }}
            />
          </label>
          <button
            type="submit"
            disabled={loading || !file}
            style={{
              background: loading ? "#a0aec0" : "#3182ce",
              color: "#fff",
              border: "none",
              borderRadius: 6,
              padding: "10px 32px",
              fontSize: 16,
              fontWeight: 600,
              cursor: loading ? "not-allowed" : "pointer",
              marginTop: 8
            }}
          >
            {loading ? "Predicting..." : "Predict"}
          </button>
        </form>

        {(fileUrl || result) && (
          <div style={{ display: "flex", justifyContent: "center", gap: 32, marginTop: 32, flexWrap: "wrap" }}>
            {fileUrl && (
              <div style={{ textAlign: "center" }}>
                <div style={{ fontWeight: 500, marginBottom: 8 }}>Input Image</div>
                <img
                  src={fileUrl}
                  alt="Input"
                  style={{ maxWidth: 220, maxHeight: 220, borderRadius: 12, border: "1px solid #e2e8f0", boxShadow: "0 2px 8px rgba(0,0,0,0.04)" }}
                />
              </div>
            )}
            {result && result.gradcam && (
              <div style={{ textAlign: "center" }}>
                <div style={{ fontWeight: 500, marginBottom: 8 }}>Grad-CAM Overlay</div>
                <img
                  src={`data:image/png;base64,${result.gradcam}`}
                  alt="Grad-CAM"
                  style={{ maxWidth: 220, maxHeight: 220, borderRadius: 12, border: "1px solid #e2e8f0", boxShadow: "0 2px 8px rgba(0,0,0,0.04)" }}
                />
              </div>
            )}
          </div>
        )}

        {result && (
          <div style={{ marginTop: 32, textAlign: "center" }}>
            {result.error ? (
              <span style={{ color: "#e53e3e", fontWeight: 500 }}>{result.error}</span>
            ) : (
              <div style={{
                display: "inline-block",
                background: "#f0fff4",
                border: "1px solid #c6f6d5",
                borderRadius: 8,
                padding: "18px 32px",
                marginTop: 8
              }}>
                <div style={{ fontSize: 18, fontWeight: 600, color: "#22543d" }}>Prediction: {result.class_name}</div>
                <div style={{ fontSize: 16, color: "#276749", marginTop: 4 }}>Confidence: {(result.confidence * 100).toFixed(2)}%</div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;