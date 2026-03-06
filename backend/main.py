import os
import logging
import json
import base64

import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from sqlalchemy.orm import Session
from jose import JWTError, jwt
from pytorch_grad_cam import GradCAM

from conversion import load_image_as_cv2_rgb
from langchain_core.messages import HumanMessage
from rag import app as rag_app, llm_tool
from model import load_retinopathy_model
from auth import (
    SignupRequest, LoginRequest, TokenResponse,
    hash_password, verify_password, create_access_token,
    get_db, get_current_user, UserDB, oauth2_scheme,
    SECRET_KEY, ALGORITHM,
)

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RetinaAI API")

# CORS — use ALLOWED_ORIGINS env var in production (comma-separated)
_allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
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

MODEL_PATH = "models/best_efficientnet.pth"
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

@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),
    current_user: UserDB = Depends(get_current_user),
):
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 10 MB.")
    ext = file.filename.split(".")[-1].lower()

    img_cv = load_image_as_cv2_rgb(contents, ext)
    image = Image.fromarray(img_cv)
    img_tensor = preprocess(image).unsqueeze(0).to(device)

    
    # Run model prediction

    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.softmax(logits, dim=1)

    pred_class = int(torch.argmax(probs))
    confidence = float(torch.max(probs))

    # Generate Grad-CAM heatmap

    target_layer = model.blocks[-1][-1].conv_pw
    cam = GradCAM(model=model, target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=img_tensor)[0]
    heatmap = grayscale_cam

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
        "gradcam": gradcam_base64,
    }


#Not using this endpoint for now, but keeping it here for future use if needed
# @app.post("/chat")
# async def chat_endpoint(request: Request):
#     data = await request.json()
#     user_question = data.get("question")

#     if not user_question:
#         return {"reply": "please provide a question"}
    
#     # Get username from token for thread_id
#     thread_id = "default"
#     auth_header = request.headers.get("authorization", "")
#     if auth_header.startswith("Bearer "):
#         try:
#             payload = jwt.decode(auth_header[7:], SECRET_KEY, algorithms=[ALGORITHM])
#             thread_id = payload.get("sub", "default")
#         except JWTError:
#             pass

#     messages = [HumanMessage(content=user_question)]
#     config = {"configurable": {"thread_id": thread_id}}
#     result_state = rag_app.invoke({"messages": messages}, config=config)

#     msgs = result_state.get("messages")

#     if isinstance(msgs, list) and len(msgs) > 0:
#         last = msgs[-1]
#         reply_text = getattr(last, "content", str(last))
#     else:
#         reply_text = str(result_state)
        
#     return {"reply": str(reply_text)}


@app.post("/chat/stream")
async def chat_stream_endpoint(request: Request):
    data = await request.json()
    user_question = data.get("question")

    # Default thread
    thread_id = "default"

    # Extract user from JWT for memory thread
    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        try:
            payload = jwt.decode(auth_header[7:], SECRET_KEY, algorithms=[ALGORITHM])
            thread_id = payload.get("sub", "default")
        except JWTError:
            pass

    config = {"configurable": {"thread_id": thread_id}}

    # Only new user message — LangGraph will load past memory automatically
    inputs = {
        "messages": [HumanMessage(content=user_question)]
    }

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