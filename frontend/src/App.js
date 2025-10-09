import React, { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [fileUrl, setFileUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [summary, setSummary] = useState("");

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setResult(null);
    setSummary("");
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
    setSummary("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("http://127.0.0.1:8000/predict/", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      setResult(data);

      // Fetch summary from /chat
      try {
        const chatRes = await fetch("http://127.0.0.1:8000/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            pred_class: data.class_id,
            confidence: data.confidence
          })
        });
        const chatData = await chatRes.json();
        setSummary(chatData.message);
      } catch (err) {
        setSummary("Could not fetch summary.");
      }
    } catch (err) {
      setResult({ error: "Prediction failed." });
      setSummary("");
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

        {summary && (
          <div style={{ marginTop: 24, background: "#f9fafb", padding: 16, borderRadius: 8 }}>
            <b>Summary & Advice:</b>
            <div style={{ marginTop: 8 }}>{summary}</div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;