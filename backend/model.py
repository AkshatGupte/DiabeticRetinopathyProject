import torch
import torch.nn as nn
from timm import create_model


def load_retinopathy_model(model_path: str):
    """
    Returns a loaded EfficientNet model ready for inference.
    Prediction logic is handled in main.py.
    """
    model = create_model("efficientnet_b0", pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, 5)

    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model
