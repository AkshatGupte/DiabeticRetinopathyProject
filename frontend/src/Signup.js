import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";

const BACKEND_BASE = process.env.REACT_APP_BACKEND_URL || "http://127.0.0.1:8000";

export default function Signup({ onLogin }) {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    if (!username || !email || !password || !confirm) {
      setError("Please fill in all fields");
      return;
    }
    if (password !== confirm) {
      setError("Passwords do not match");
      return;
    }
    if (password.length < 6) {
      setError("Password must be at least 6 characters");
      return;
    }
    setLoading(true);
    try {
      const res = await fetch(`${BACKEND_BASE}/auth/signup`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, email, password }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Signup failed");
      localStorage.setItem("token", data.access_token);
      localStorage.setItem("username", data.username);
      onLogin(data.username);
      navigate("/");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.wrapper}>
      <div style={styles.card}>
        <div style={styles.logoRow}>
          <span style={styles.logoIcon}>&#128065;</span>
          <span style={styles.logoText}>RetinaAI</span>
        </div>
        <h2 style={styles.title}>Create account</h2>
        <p style={styles.subtitle}>Get started with RetinaAI</p>

        {error && <div style={styles.error}>{error}</div>}

        <form onSubmit={handleSubmit} style={styles.form}>
          <label style={styles.label}>Username</label>
          <input
            style={styles.input}
            type="text"
            placeholder="Choose a username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            autoFocus
          />

          <label style={styles.label}>Email</label>
          <input
            style={styles.input}
            type="email"
            placeholder="you@example.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
          />

          <label style={styles.label}>Password</label>
          <input
            style={styles.input}
            type="password"
            placeholder="At least 6 characters"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />

          <label style={styles.label}>Confirm Password</label>
          <input
            style={styles.input}
            type="password"
            placeholder="Re-enter your password"
            value={confirm}
            onChange={(e) => setConfirm(e.target.value)}
          />

          <button type="submit" style={styles.button} disabled={loading}>
            {loading ? "Creating account..." : "Create Account"}
          </button>
        </form>

        <p style={styles.footer}>
          Already have an account?{" "}
          <Link to="/login" style={styles.link}>Sign in</Link>
        </p>
      </div>
    </div>
  );
}

const styles = {
  wrapper: {
    minHeight: "100vh",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    fontFamily: "Inter, Roboto, Arial, sans-serif",
    padding: 20,
  },
  card: {
    width: "100%",
    maxWidth: 420,
    background: "#fff",
    borderRadius: 16,
    padding: "40px 36px",
    boxShadow: "0 20px 60px rgba(0,0,0,0.15)",
  },
  logoRow: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    marginBottom: 24,
    justifyContent: "center",
  },
  logoIcon: { fontSize: 28 },
  logoText: { fontSize: 22, fontWeight: 800, color: "#4f46e5" },
  title: {
    margin: 0,
    fontSize: 24,
    fontWeight: 700,
    textAlign: "center",
    color: "#1e293b",
  },
  subtitle: {
    margin: "6px 0 24px",
    fontSize: 14,
    color: "#64748b",
    textAlign: "center",
  },
  form: { display: "flex", flexDirection: "column", gap: 12 },
  label: { fontSize: 13, fontWeight: 600, color: "#334155" },
  input: {
    padding: "12px 14px",
    borderRadius: 10,
    border: "1px solid #e2e8f0",
    fontSize: 14,
    outline: "none",
    transition: "border 0.2s",
  },
  button: {
    marginTop: 8,
    padding: "14px 0",
    borderRadius: 10,
    border: "none",
    background: "linear-gradient(135deg, #6366f1, #8b5cf6)",
    color: "#fff",
    fontSize: 15,
    fontWeight: 700,
    cursor: "pointer",
    transition: "opacity 0.2s",
  },
  error: {
    background: "#fef2f2",
    color: "#dc2626",
    padding: "10px 14px",
    borderRadius: 8,
    fontSize: 13,
    marginBottom: 8,
    border: "1px solid #fecaca",
  },
  footer: {
    textAlign: "center",
    marginTop: 20,
    fontSize: 13,
    color: "#64748b",
  },
  link: {
    color: "#6366f1",
    fontWeight: 600,
    textDecoration: "none",
  },
};
