import React, { useState, useRef, useEffect } from "react";
import { Routes, Route, Navigate, useNavigate } from "react-router-dom";
import Login from "./Login";
import Signup from "./Signup";

const BACKEND_BASE = "http://127.0.0.1:8000";

function Dashboard({ username, onLogout }) {
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [fileUrl, setFileUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const [messages, setMessages] = useState([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const chatScrollRef = useRef(null);

  useEffect(() => {
    chatScrollRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, chatLoading]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files?.[0] || null;
    setFile(selectedFile);
    setResult(null);
    if (selectedFile) {
      try {
        const url = URL.createObjectURL(selectedFile);
        setFileUrl(url);
      } catch (err) {
        setFileUrl(null);
      }
    } else {
      setFileUrl(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    setResult(null);
    try {
      const fd = new FormData();
      fd.append("file", file, file.name);
      const res = await fetch(`${BACKEND_BASE}/predict/`, { method: "POST", body: fd });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setResult({ error: String(err) });
    } finally {
      setLoading(false);
    }
  };

  const handleChat = async () => {
    const question = chatInput.trim();
    if (!question) return;
    const newMessages = [...messages, { role: "user", content: question }];
    setMessages(newMessages);
    setChatInput("");
    setChatLoading(true);
    try {
      const token = localStorage.getItem("token");
      const res = await fetch(`${BACKEND_BASE}/chat/stream`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ question }),
      });

      // Add an empty assistant message that we'll fill incrementally
      setMessages((m) => [...m, { role: "assistant", content: "" }]);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = decoder.decode(value, { stream: true });
        // Parse SSE lines: each chunk is "data: ...\n\n"
        const lines = text.split("\n");
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const token = line.slice(6);
            if (token === "[DONE]") break;
            // Append token to the last (assistant) message
            setMessages((prev) => {
              const updated = [...prev];
              const last = updated[updated.length - 1];
              updated[updated.length - 1] = { ...last, content: last.content + token };
              return updated;
            });
          }
        }
      }

      // If the assistant message is still empty, show fallback
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last.role === "assistant" && !last.content) {
          const updated = [...prev];
          updated[updated.length - 1] = { ...last, content: "(no reply)" };
          return updated;
        }
        return prev;
      });
    } catch (err) {
      setMessages((m) => [...m, { role: "assistant", content: `Error: ${err.message}` }]);
    } finally {
      setChatLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleChat();
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("username");
    onLogout();
    navigate("/login");
  };

  return (
    <div style={{ minHeight: "100vh", background: "#f3f6fb", padding: 20, fontFamily: "Inter, Roboto, Arial, sans-serif" }}>
      {/* Navbar */}
      <div style={{ maxWidth: 1000, margin: "0 auto 12px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span style={{ fontSize: 22 }}>&#128065;</span>
          <span style={{ fontSize: 18, fontWeight: 800, color: "#4f46e5" }}>RetinaAI</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <span style={{ fontSize: 13, color: "#475569", fontWeight: 600 }}>{username}</span>
          <button onClick={handleLogout} style={{ padding: "6px 14px", borderRadius: 8, border: "1px solid #e2e8f0", background: "#fff", color: "#6366f1", fontWeight: 700, cursor: "pointer", fontSize: 13 }}>Logout</button>
        </div>
      </div>

      <div style={{ maxWidth: 1000, margin: "0 auto", display: "grid", gridTemplateColumns: "1fr 420px", gap: 20 }}>

        <div style={{ background: "#fff", padding: 20, borderRadius: 12, boxShadow: "0 6px 20px rgba(16,24,40,0.04)" }}>
          <h2 style={{ margin: 0, marginBottom: 12 }}>Predict</h2>
          <form onSubmit={handleSubmit} style={{ display: "flex", gap: 12, alignItems: "center" }}>
            <input type="file" accept="image/*" onChange={handleFileChange} />
            <button type="submit" disabled={loading || !file} style={{ padding: "8px 12px", borderRadius: 8, background: "#0ea5e9", color: "#fff", border: "none" }}>{loading ? "Predicting..." : "Predict"}</button>
          </form>

          {fileUrl && <div style={{ marginTop: 12 }}><img src={fileUrl} alt="preview" style={{ width: 200, borderRadius: 8 }} /></div>}

          {result && (
            <div style={{ marginTop: 12, background: "#f8fffa", border: "1px solid #d1fae5", padding: 12, borderRadius: 8 }}>
              {result.error ? (
                <div style={{ color: "#ef4444" }}>{result.error}</div>
              ) : (
                <div>
                  <div style={{ fontWeight: 700 }}>{result.diagnosis}</div>
                  <div style={{ color: "#334155" }}>Confidence: {(result.confidence * 100).toFixed(2)}%</div>
                  {result.gradcam && <img src={`data:image/png;base64,${result.gradcam}`} alt="gradcam" style={{ width: "100%", marginTop: 8, borderRadius: 6 }} />}
                </div>
              )}
            </div>
          )}
        </div>

        <div style={{ background: "#fff", padding: 18, borderRadius: 12, boxShadow: "0 6px 20px rgba(16,24,40,0.04)", display: "flex", flexDirection: "column", height: 560 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
            <div style={{ fontSize: 16, fontWeight: 800 }}>Assistant</div>
            <div style={{ fontSize: 12, color: "#64748b" }}>{messages.length} messages</div>
          </div>

          <div style={{ flex: 1, overflowY: "auto", padding: 8, border: "1px solid #eef2ff", borderRadius: 8 }}>
            {messages.length === 0 ? (
              <div style={{ color: "#94a3b8", textAlign: "center", paddingTop: 20 }}>Ask about predictions or diabetic retinopathy.</div>
            ) : (
              messages.map((m, i) => (
                <div key={i} style={{ marginBottom: 8, textAlign: m.role === "user" ? "right" : "left" }}>
                  <div style={{ display: "inline-block", background: m.role === "user" ? "#e0f2fe" : "#eef2ff", padding: "8px 12px", borderRadius: 10 }}>{typeof m.content === "string" ? m.content : "[Invalid message]"}</div>
                </div>
              ))
            )}
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
                onClick={() => setMessages([])}
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

    </div>
  );
}

function App() {
  const [username, setUsername] = useState(localStorage.getItem("username") || "");
  const isLoggedIn = !!username;

  return (
    <Routes>
      <Route path="/login" element={isLoggedIn ? <Navigate to="/" /> : <Login onLogin={setUsername} />} />
      <Route path="/signup" element={isLoggedIn ? <Navigate to="/" /> : <Signup onLogin={setUsername} />} />
      <Route path="/" element={isLoggedIn ? <Dashboard username={username} onLogout={() => setUsername("")} /> : <Navigate to="/login" />} />
    </Routes>
  );
}

export default App;
