from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import base64
import matplotlib.pyplot as plt
import cv2
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel
from dotenv import load_dotenv

app = FastAPI()

class ChatRequest(BaseModel):
    pred_class: int
    confidence: float

# Allow CORS for all origins (for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your trained model once at startup
model = tf.keras.models.load_model("../best_mobilenetv2.h5")
IMG_SIZE = (224, 224)

# Map numeric predictions to class names
label_map = {0: "No_DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferate_DR"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    preds = model.predict(img_array)
    pred_class = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))

    # Grad-CAM implementation
    def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    # Get Grad-CAM heatmap
    last_conv_layer_name = "Conv_1"  # For MobileNetV2
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_class)

    # Overlay heatmap on image
    img_for_overlay = np.array(image)
    heatmap_resized = cv2.resize(heatmap, (img_for_overlay.shape[1], img_for_overlay.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img_for_overlay, 0.6, heatmap_color, 0.4, 0)

    # Encode overlay image to base64
    _, buffer = cv2.imencode('.png', cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    gradcam_base64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "class_id": pred_class,
        "class_name": label_map[pred_class],
        "confidence": confidence,
        "gradcam": gradcam_base64
    }

prompt = PromptTemplate(
    template='''
    You are a helpful medical assistant. A user has uploaded an image of an eye and received a diagnosis.
    The diagnosis is {diagnosis} with a confidence of {confidence:.2f}.
    
    You should provide the user with information about the diagnosis, possible next steps, and any relevant advice.
    ''',
    input_variables=["diagnosis", "confidence"]
)

load_dotenv()
@app.post('/chat')
async def chat(request: ChatRequest):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.4)
    template = prompt.invoke({"diagnosis": label_map[request.pred_class], "confidence": request.confidence})
    response = llm.invoke(template)
    return {"message": str(response.content)}
