import io
import cv2
import numpy as np
import pydicom
import torch

IMG_SIZE = 224

def load_image_as_cv2_rgb(file_bytes: bytes, ext: str):
    """
    Loads JPG/PNG/DICOM into a numpy RGB image (H, W, 3).
    Matches exactly how training images were loaded.
    """
    ext = ext.lower()

    # -----------------------------
    # JPG / PNG
    # -----------------------------
    if ext in ["jpg", "jpeg", "png"]:
        file_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(file_arr, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            raise ValueError("Failed to decode JPG/PNG")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    # -----------------------------
    # DICOM
    # -----------------------------
    elif ext == "dcm":
        ds = pydicom.dcmread(io.BytesIO(file_bytes))
        arr = ds.pixel_array.astype(np.float32)

        # Normalize safely
        arr = arr / (arr.max() + 1e-8)
        arr = (arr * 255).astype(np.uint8)

        # Make RGB
        if len(arr.shape) == 2:
            arr = np.stack([arr] * 3, axis=-1)

        return arr

    else:
        raise ValueError("Unsupported file extension: " + ext)


