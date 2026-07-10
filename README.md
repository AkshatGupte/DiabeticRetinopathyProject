# RetinaAI — Diabetic Retinopathy Detection

A full-stack web application that uses deep learning to classify diabetic retinopathy severity from retinal fundus images, with visual explanations (Grad-CAM) and an AI-powered medical assistant chatbot.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![React](https://img.shields.io/badge/React-18-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red)

## Features

- **Image Classification** — Upload a retinal image (JPG/PNG/DICOM) and get one of 5 diagnoses (No DR, Mild, Moderate, Severe, or Proliferative DR) with per-class probabilities
- **Grad-CAM Heatmaps** — Visual overlay showing which regions of the image the model focused on
- **AI Medical Assistant** — RAG-powered chatbot that answers questions about diabetic retinopathy using a medical PDF as context, with streaming responses
- **User Authentication** — JWT-based signup/login with bcrypt password hashing
- **Conversation Memory** — Per-user chat history persisted in the database (Postgres in production, SQLite locally), survives server restarts

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | React 18, React Router v7 |
| **Backend** | FastAPI, Uvicorn |
| **ML Model** | EfficientNet-B0 (PyTorch, timm) |
| **Explainability** | Grad-CAM (pytorch-grad-cam) |
| **Chat/RAG** | LangGraph, LangChain, HuggingFace Inference API, FAISS |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| **Database** | PostgreSQL (Supabase) |
| **Auth** | JWT (python-jose), bcrypt (passlib) |

## Project Structure

```
├── backend/
│   ├── main.py              # FastAPI server — all endpoints
│   ├── auth.py              # User model, JWT, password hashing
│   ├── model.py             # EfficientNet-B0 loading
│   ├── rag.py               # RAG chatbot (LangGraph + FAISS)
│   ├── conversion.py        # Image format handling (JPG/PNG/DICOM)
│   ├── lime_explainer.py    # LIME explainability (not yet wired to endpoint)
│   ├── requirements.txt
│   ├── Procfile             # For deployment (Render/Railway)
│   ├── faiss_index/         # Persisted vector index (auto-created on first boot)
│   └── models/
│       └── best_efficientnet.pth  # Trained model weights
├── data/                    # Training data + experiment scripts (not used by the app)
├── frontend/
│   ├── src/
│   │   ├── App.js           # Dashboard — predict + chat UI
│   │   ├── Login.js         # Login page
│   │   ├── Signup.js        # Signup page
│   │   └── index.js         # React entry point
│   └── package.json
└── rag/
    └── fact_sheet_22_diabetic_retinopathy_new.pdf  # Medical PDF for RAG
```

## API Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/auth/signup` | No | Create account, returns JWT |
| `POST` | `/auth/login` | No | Login, returns JWT |
| `GET` | `/auth/me` | Yes | Current user info |
| `POST` | `/predict/` | Yes | Upload retina image → diagnosis + Grad-CAM |
| `POST` | `/chat/stream` | Yes | Ask the AI assistant (SSE streaming) |
| `GET` | `/health` | No | Health check |

## Model Details

- **Architecture**: EfficientNet-B0 (pretrained on ImageNet, fine-tuned)
- **Classes**: 5 severity levels of diabetic retinopathy
- **Input**: 224×224 RGB images, normalized with ImageNet mean/std
- **Explainability**: Grad-CAM on the last convolutional layer
