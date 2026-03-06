# RetinaAI — Diabetic Retinopathy Detection

A full-stack web application that uses deep learning to classify diabetic retinopathy severity from retinal fundus images, with visual explanations (Grad-CAM) and an AI-powered medical assistant chatbot.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)
![React](https://img.shields.io/badge/React-18-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-red)

## Features

- **Image Classification** — Upload a retinal image (JPG/PNG/DICOM) and get one of 5 diagnoses: No DR, Mild, Moderate, Severe, or Proliferative DR
- **Grad-CAM Heatmaps** — Visual overlay showing which regions of the image the model focused on
- **AI Medical Assistant** — RAG-powered chatbot that answers questions about diabetic retinopathy using a medical PDF as context, with streaming responses
- **User Authentication** — JWT-based signup/login with bcrypt password hashing
- **Conversation Memory** — Per-user chat history maintained across messages

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
│   └── models/
│       └── best_efficientnet.pth  # Trained model weights
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
| `POST` | `/chat/stream` | Optional | Ask the AI assistant (SSE streaming) |
| `GET` | `/health` | No | Health check |

## Setup

### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL database (or [Supabase](https://supabase.com) free tier)
- HuggingFace API token

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `backend/.env` file (see `.env.example`):

```env
JWT_SECRET=your-random-secret-string
HUGGINGFACEHUB_API_TOKEN=hf_your_token
DATABASE_URL=postgresql://user:pass@host:port/dbname
```

Generate a JWT secret:
```bash
python -c "import secrets; print(secrets.token_urlsafe(64))"
```

Start the server:
```bash
uvicorn main:app --reload
```

### Frontend

```bash
cd frontend
npm install
```

Create a `frontend/.env` file (see `.env.example`):
```env
REACT_APP_BACKEND_URL=http://127.0.0.1:8000
```

Start the dev server:
```bash
npm start
```

## Deployment

| Component | Platform | Notes |
|-----------|----------|-------|
| Frontend | Vercel | Set `REACT_APP_BACKEND_URL` env var |
| Backend | Railway / Render | Set all env vars from `.env.example`, uses `Procfile` |
| Database | Supabase | Free PostgreSQL, use pooler connection string |

### Vercel (Frontend)
1. Connect your GitHub repo
2. Set root directory to `frontend`
3. Add env var: `REACT_APP_BACKEND_URL=https://your-backend-url.com`

### Railway/Render (Backend)
1. Connect your GitHub repo
2. Set root directory to `backend`
3. Add all env vars from `backend/.env.example`
4. Ensure model weights (`best_efficientnet.pth`) are included in the deploy

## Model Details

- **Architecture**: EfficientNet-B0 (pretrained on ImageNet, fine-tuned)
- **Classes**: 5 severity levels of diabetic retinopathy
- **Input**: 224×224 RGB images, normalized with ImageNet mean/std
- **Explainability**: Grad-CAM on the last convolutional layer

## License

This project is for educational and research purposes.