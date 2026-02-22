# Diabetic Retinopathy Project

Full-stack application for diabetic retinopathy analysis with a FastAPI backend (prediction, Grad-CAM, LIME, and RAG-based chat) and a React frontend.

## Prerequisites
- Python 3.10+ (backend)
- Node.js 18+ (frontend)
- Redis Stack (for chat history and search index)
- Pinecone API key and index named `retinopathy`
- Groq API key (for chat model)

Set required environment variables in a `.env` file at the project root or in the backend folder (loaded via `python-dotenv`). Typical variables:
- `PINECONE_API_KEY`
- `PINECONE_ENVIRONMENT` (if needed by your Pinecone client)
- `GROQ_API_KEY`

## Start Redis (required for chat history)
Redis Stack is required because chat history uses RediSearch commands like `FT._LIST`.

**Option A: Docker (recommended)**
```
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack:latest
```

**Option B: Local install**
Install Redis Stack from https://redis.io/docs/latest/operate/oss_and_stack/install/ and ensure it is running on `localhost:6379`.

## Start Backend (FastAPI)
From the project root:
```
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```
Backend runs at: http://127.0.0.1:8000

## Start Frontend (React)
From the project root:
```
cd frontend
npm install
npm run dev
```
Frontend runs at: http://127.0.0.1:5173 (or the port shown in the terminal).

## Notes
- The chat endpoint depends on Redis Stack. If Redis Stack is not running, chat will error.
- Ensure the Pinecone index `retinopathy` already exists with compatible embeddings.
- GPU is optional; the backend will use CPU if CUDA is unavailable.
