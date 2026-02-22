import React, { useState, useRef, useEffect } from "react";
import { v4 as uuidv4 } from "uuid";

function App() {
  const [file, setFile] = useState(null);
  const [fileUrl, setFileUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [sessionId] = useState(uuidv4());
  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState(""); // State for chat input
  const [chatLoading, setChatLoading] = useState(false);
  const chatScrollRef = useRef(null);

  useEffect(() => {
    // scroll to bottom when messages change
    chatScrollRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, chatLoading]);

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

  const extractReplyFromResponse = (data) => {
    // Support multiple backend response formats
    if (!data) return null;
    if (typeof data === "string") return data;
    if (data.reply) return data.reply;
    if (data.response) return data.response;
    if (data.answer) return data.answer;
    // If backend returned something like { message: "..." }
    if (data.message) return data.message;
    // If backend returned a nested object with text
    if (data.data && typeof data.data === "string") return data.data;
    return JSON.stringify(data); // fallback: show whole payload
  };

  const handleChat = async () => {
  if (!chatInput.trim()) return;
  const userMessage = { role: "user", content: chatInput };
  const updatedMessages = [...messages, userMessage];

  setMessages(updatedMessages);
  setChatInput("");

    try {
      const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          question: chatInput, // ✅ send this instead of messages
        }),
      });

      const data = await res.json();
      const assistantMessage = { role: "assistant", content: data.reply };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      console.error("Chat error:", err);
    }
  };


  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleChat();
    }
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#f3f6fb",
        padding: 24,
        margin: 0,
        fontFamily: "Inter, Roboto, Arial, sans-serif",
      }}
    >
      <div
        style={{
          maxWidth: 1100,
          margin: "24px auto",
          display: "grid",
          gridTemplateColumns: "1fr 420px",
          gap: 24,
        }}
      >
        {/* Left column: Prediction + images */}
        <div
          style={{
            background: "#fff",
            borderRadius: 14,
            boxShadow: "0 6px 30px rgba(16,24,40,0.06)",
            padding: 28,
          }}
        >
          <h1
            style={{
              fontSize: 20,
              fontWeight: 700,
              color: "#0f172a",
              marginBottom: 18,
            }}
          >
            Diabetic Retinopathy Classifier
          </h1>

          <form
            onSubmit={handleSubmit}
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 12,
            }}
          >
            <label
              htmlFor="file-upload"
              style={{
                background: "#f8fafc",
                border: "2px dashed #e6eef8",
                borderRadius: 10,
                padding: "28px 18px",
                cursor: "pointer",
                textAlign: "center",
                color: file ? "#0ea5e9" : "#64748b",
                fontWeight: 600,
                display: "block",
              }}
            >
              {file ? `Selected: ${file.name}` : "Click or drag an image here to upload"}
              <input
                id="file-upload"
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                style={{ display: "none" }}
              />
            </label>

            <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
              <button
                type="submit"
                disabled={loading || !file}
                style={{
                  background: loading ? "#94a3b8" : "#0ea5e9",
                  color: "#fff",
                  border: "none",
                  borderRadius: 10,
                  padding: "10px 20px",
                  fontSize: 15,
                  fontWeight: 700,
                  cursor: loading ? "not-allowed" : "pointer",
                }}
              >
                {loading ? "Predicting..." : "Predict"}
              </button>

              {fileUrl && (
                <div
                  style={{
                    display: "flex",
                    gap: 12,
                    alignItems: "center",
                    background: "#f8fafc",
                    padding: 8,
                    borderRadius: 8,
                    border: "1px solid #e6eef8",
                  }}
                >
                  <img
                    src={fileUrl}
                    alt="preview"
                    style={{ width: 72, height: 72, objectFit: "cover", borderRadius: 6 }}
                  />
                  <div style={{ fontSize: 13, color: "#334155" }}>Preview</div>
                </div>
              )}
            </div>
          </form>

          {/* Prediction Results */}
          {result && (
            <div style={{ marginTop: 28 }}>
              {result.error ? (
                <div style={{ color: "#ef4444", fontWeight: 600 }}>{result.error}</div>
              ) : (
                <div
                  style={{
                    display: "grid",
                    gap: 12,
                    gridTemplateColumns: "1fr 1fr",
                    background: "#f8fffa",
                    border: "1px solid #d1fae5",
                    padding: 16,
                    borderRadius: 10,
                    alignItems: "start",
                  }}
                >
                  <div>
                    <div style={{ fontSize: 16, fontWeight: 700, color: "#065f46" }}>
                      Prediction: {result.diagnosis}
                    </div>
                    <div style={{ fontSize: 14, color: "#065f46", marginTop: 6 }}>
                      Confidence: {(result.confidence * 100).toFixed(2)}%
                    </div>
                  </div>

                  <div>
                    <div style={{ fontSize: 14, color: "#0f172a", marginBottom: 8, fontWeight: 600 }}>
                      Visual Explanations
                    </div>

                    <div style={{ display: "flex", gap: 8, flexDirection: "column" }}>
                      <div>
                        <div style={{ fontSize: 13, marginBottom: 6 }}>Grad-CAM</div>
                        <img
                          src={`data:image/png;base64,${result.gradcam}`}
                          alt="Grad-CAM"
                          style={{ width: "100%", maxHeight: 180, objectFit: "contain", borderRadius: 8 }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Right column: Chat */}
        <div
          style={{
            background: "#fff",
            borderRadius: 14,
            boxShadow: "0 6px 30px rgba(16,24,40,0.06)",
            padding: 18,
            display: "flex",
            flexDirection: "column",
            height: 640,
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div style={{ fontSize: 16, fontWeight: 800, color: "#0f172a" }}>Assistant</div>
              <div style={{ fontSize: 12, color: "#475569" }}>Session: {sessionId.slice(0, 8)}</div>
            </div>

            <div style={{ fontSize: 12, color: "#64748b" }}>
              {chatLoading ? "Assistant is typing..." : `${messages.length} messages`}
            </div>
          </div>

          <div
            style={{
              marginTop: 12,
              background: "#f8fafc",
              borderRadius: 10,
              padding: 12,
              flex: 1,
              overflowY: "auto",
              border: "1px solid #e6eef8",
            }}
          >
            {messages.length === 0 && (
              <div style={{ color: "#94a3b8", textAlign: "center", paddingTop: 36 }}>
                Ask the assistant about predictions, model explanations, or diabetic retinopathy.
              </div>
            )}

            {messages.map((msg, index) => {
              const isUser = msg.role === "user";
              return (
                <div
                  key={index}
                  style={{
                    display: "flex",
                    justifyContent: isUser ? "flex-end" : "flex-start",
                    marginBottom: 10,
                  }}
                >
                  <div
                    style={{
                      maxWidth: "78%",
                      background: isUser ? "#e0f2fe" : "#eef2ff",
                      color: "#0f172a",
                      padding: "10px 14px",
                      borderRadius: 12,
                      fontSize: 14,
                      lineHeight: 1.35,
                      boxShadow: "0 1px 2px rgba(2,6,23,0.04)",
                      whiteSpace: "pre-wrap",
                    }}
                  >
                    {typeof msg.content === "string" ? msg.content : "[Invalid message]"}
                  </div>
                </div>
              );
            })}
            <div ref={chatScrollRef} />
          </div>

          <div style={{ marginTop: 12, display: "flex", gap: 8 }}>
            <textarea
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Type your message... (Enter to send, Shift+Enter for newline)"
              style={{
                flex: 1,
                minHeight: 56,
                maxHeight: 150,
                padding: 12,
                borderRadius: 10,
                border: "1px solid #e6eef8",
                resize: "vertical",
                fontSize: 14,
                outline: "none",
              }}
              disabled={chatLoading}
            />

            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              <button
                onClick={handleChat}
                disabled={chatLoading || !chatInput.trim()}
                style={{
                  background: chatLoading ? "#94a3b8" : "#6366f1",
                  color: "#fff",
                  border: "none",
                  borderRadius: 10,
                  padding: "10px 16px",
                  fontWeight: 700,
                  cursor: chatLoading ? "not-allowed" : "pointer",
                }}
              >
                Send
              </button>

              <button
                onClick={() => {
                  setMessages([]);
                }}
                style={{
                  background: "#f1f5f9",
                  color: "#0f172a",
                  border: "1px solid #e6eef8",
                  borderRadius: 10,
                  padding: "8px 12px",
                  fontWeight: 700,
                  cursor: "pointer",
                }}
              >
                Clear
              </button>
            </div>
          </div>
        </div>
      </div>

      <div style={{ textAlign: "center", marginTop: 18, color: "#94a3b8", fontSize: 13 }}>
        Backend: <code>http://127.0.0.1:8000</code> • Chat endpoint: <code>/chat</code> • Session is stored client-side
      </div>
    </div>
  );
}

export default App;
