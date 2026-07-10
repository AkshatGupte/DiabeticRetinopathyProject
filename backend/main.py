import os
import json
import base64
import threading

import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from pytorch_grad_cam import GradCAM

from conversion import load_image_as_cv2_rgb
from langchain_core.messages import HumanMessage
from rag import app as rag_app
from model import load_retinopathy_model
from auth import (
    SignupRequest, LoginRequest, TokenResponse,
    hash_password, verify_password, create_access_token,
    get_db, get_current_user, UserDB,
)

load_dotenv()

app = FastAPI(title="RetinaAI API")

# CORS — set ALLOWED_ORIGINS env var in production (comma-separated).
# A wildcard is invalid when allow_credentials=True, so default to local dev origins.
_allowed_origins = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Max upload size (10 MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

device = "cuda" if torch.cuda.is_available() else "cpu"


@app.get("/health")
def health():
    return {"status": "ok"}

# ===================== AUTH ENDPOINTS =====================

@app.post("/auth/signup", response_model=TokenResponse)
def signup(body: SignupRequest, db: Session = Depends(get_db)):
    if db.query(UserDB).filter(UserDB.username == body.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    if db.query(UserDB).filter(UserDB.email == body.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = UserDB(
        username=body.username,
        email=body.email,
        hashed_password=hash_password(body.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": user.username})
    return TokenResponse(access_token=token, username=user.username)


@app.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(UserDB).filter(UserDB.username == body.username).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    token = create_access_token({"sub": user.username})
    return TokenResponse(access_token=token, username=user.username)


@app.get("/auth/me")
def get_me(current_user: UserDB = Depends(get_current_user)):
    return {"username": current_user.username, "email": current_user.email}

# ===================== MODEL SETUP =====================

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "models", "best_efficientnet.pth")
model = load_retinopathy_model(MODEL_PATH)
model.to(device)
model.eval()


label_map = {0: "No_DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferate_DR"}

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# One GradCAM instance for the app's lifetime — creating one per request
# re-registers hooks on the shared model and leaks them.
_target_layer = model.blocks[-1][-1].conv_pw
_cam = GradCAM(model=model, target_layers=[_target_layer])

# The shared model isn't safe for concurrent Grad-CAM backward passes.
_predict_lock = threading.Lock()


@app.post("/predict/")
def predict(
    file: UploadFile = File(...),
    current_user: UserDB = Depends(get_current_user),
):
    contents = file.file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 10 MB.")

    filename = file.filename or ""
    if "." not in filename:
        raise HTTPException(status_code=400, detail="File must have a .jpg, .png or .dcm extension")
    ext = filename.rsplit(".", 1)[-1].lower()

    try:
        img_cv = load_image_as_cv2_rgb(contents, ext)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    image = Image.fromarray(img_cv)
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    with _predict_lock:
        # Run model prediction
        with torch.no_grad():
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)

        pred_class = int(torch.argmax(probs))
        confidence = float(torch.max(probs))

        # Generate Grad-CAM heatmap (needs gradients — runs outside no_grad)
        heatmap = _cam(input_tensor=img_tensor)[0]

    # Convert heatmap to overlay
    img_np = np.array(image.resize((224, 224)))
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img_np, 0.6, heatmap_color, 0.4, 0)

    _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    gradcam_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "class_id": pred_class,
        "diagnosis": label_map[pred_class],
        "confidence": confidence,
        "probabilities": {
            label_map[i]: round(float(p), 4) for i, p in enumerate(probs[0].tolist())
        },
        "gradcam": gradcam_base64,
    }


# ===================== CHAT =====================

class ChatRequest(BaseModel):
    question: str


@app.post("/chat/stream")
def chat_stream_endpoint(
    body: ChatRequest,
    current_user: UserDB = Depends(get_current_user),
):
    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please provide a question")

    # Per-user memory thread — LangGraph loads past messages automatically
    config = {"configurable": {"thread_id": current_user.username}}
    inputs = {"messages": [HumanMessage(content=question)]}

    def event_generator():
        for message_chunk, metadata in rag_app.stream(
            inputs,
            config=config,
            stream_mode="messages",
        ):
            if (
                message_chunk.content
                and metadata.get("langgraph_node") == "chat node"
            ):
                yield f"data: {json.dumps(message_chunk.content)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
