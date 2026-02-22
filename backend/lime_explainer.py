# lime.py
import io
import base64
import numpy as np
import cv2
from lime import lime_image
from torchvision import transforms
import torch
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# default image size used by your model
IMG_SIZE = (224, 224)

def _prepare_for_model(img_uint8):
    """
    img_uint8: HxWx3 uint8 RGB image (0..255)
    Returns model-ready batch array shape (1, H_model, W_model, 3) scaled 0..1
    """
    # resize to model input
    resized = cv2.resize(img_uint8, IMG_SIZE, interpolation=cv2.INTER_AREA)
    arr = resized.astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def predict_fn_for_lime(images, model):
    """
    PyTorch version of LIME prediction wrapper.
    DOES NOT require model passed from main.py
    Uses lime_model loaded above.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    batch_tensors = []

    for im in images:
        if im.dtype != np.uint8:
            im = (np.clip(im, 0, 1) * 255).astype(np.uint8)

        tensor = preprocess(im).unsqueeze(0)
        batch_tensors.append(tensor)

    batch = torch.cat(batch_tensors, dim=0).to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)

    return probs.cpu().numpy()

def segmentation_slic(image, n_segments=150, compactness=10):
    """
    image: HxWx3 uint8
    returns superpixel segmentation (H,W) for LIME
    """
    # slic expects float image in [0,1] or uint8; convert to float as recommended
    image_float = np.asarray(image).astype(np.float32) / 255.0
    segments = slic(image_float, n_segments=n_segments, compactness=compactness, start_label=0)
    return segments

def explain_image_with_lime(model, pil_image, top_labels=1, num_samples=1000, 
                            num_features=10, positive_only=True, segmentation_fn=None):
    """
    Run LIME on a single PIL.Image (RGB) and return:
      - pred_label (int)
      - pred_probs (np.array)
      - overlay_base64 (str)  <- PNG base64 encoding of LIME overlay (same HxW as original input)
    Notes:
      - num_samples controls speed/quality: 500..2000 typical. More → slower but more stable.
      - segmentation_fn: function(image)->segments, if None we use SLIC with defaults.
    """
    # convert PIL -> uint8 RGB
    img = pil_image.convert("RGB")
    img_np = np.array(img).astype(np.uint8)
    # prepare prediction wrapper for LIME
    classifier_fn = lambda imgs: predict_fn_for_lime(imgs, model)

    explainer = lime_image.LimeImageExplainer()

    if segmentation_fn is None:
        segmentation_fn = lambda x: segmentation_slic(x, n_segments=150, compactness=10)

    # Run explain_instance (can be slow)
    explanation = explainer.explain_instance(
        image=img_np,
        classifier_fn=classifier_fn,
        top_labels=top_labels,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=segmentation_fn
    )

    # model prediction for the (resized) image
    preds = predict_fn_for_lime([img_np], model)[0]
    pred_label = int(np.argmax(preds))

    # get image + mask where mask highlights top positive features for pred_label
    temp, mask = explanation.get_image_and_mask(
        label=pred_label,
        positive_only=positive_only,
        num_features=num_features,
        hide_rest=False
    )

    # temp is uint8 image with highlighted superpixels; combine with original to create overlay
    # mark_boundaries will draw superpixel borders; we also color positive regions
    # Create overlay: original image with red tint on selected superpixels
    overlay = img_np.copy().astype(np.float32) / 255.0
    # mask is same HxW with integers (superpixel ids) OR boolean; when hide_rest=False temp contains display
    # here: create colored overlay where mask>0
    if mask is not None:
        # mask values: 0/1 per pixel (LIME returns mask as int with superpixel ids highlighted)
        # we will highlight pixels where mask>0 (selected features)
        highlight = mask.astype(bool)
        color = np.array([1.0, 0.25, 0.25])  # red tint
        overlay[highlight] = overlay[highlight] * 0.5 + color * 0.5


    overlay_bounded = mark_boundaries((overlay).astype(np.float32), mask, color=(1,0,0))
    overlay_uint8 = (np.clip(overlay_bounded, 0, 1) * 255).astype(np.uint8)

    # encode overlay to base64 PNG
    _, buffer = cv2.imencode('.png', cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR))
    overlay_b64 = base64.b64encode(buffer).decode('utf-8')

    return {
        "pred_label": pred_label,
        "pred_probs": preds.tolist(),
        "lime_overlay_base64": overlay_b64
    }
