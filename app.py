from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import os
import joblib
from ultralytics import YOLO

# =====================================================
# APP SETUP
# =====================================================

app = FastAPI(title="Agri AI Backend")

# CORS (allow frontend like Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# ----------- CROP RECOMMENDATION MODEL ----------------
# =====================================================

# Absolute path to backend directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CROP_MODEL_PATH = os.path.join(
    BASE_DIR, "model", "crop_recommendation_model.joblib"
)


# Load crop model ONCE
crop_model = joblib.load(CROP_MODEL_PATH)

class CropInput(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    prediction = crop_model.predict(input_data)
    return prediction[0]


# =====================================================
# ----------- DISEASE DETECTION MODEL ------------------
# =====================================================

DISEASE_MODEL_PATH = os.path.join(
    BASE_DIR, "model", "disease.pt"
)

CONF_THRESHOLD = 0.25

# Load YOLO model ONCE
disease_model = YOLO(DISEASE_MODEL_PATH)


# =====================================================
# ------------------- ROUTES ---------------------------
# =====================================================

@app.get("/")
def home():
    return {"message": "Agri AI Backend is running"}


# -------- Crop Recommendation API --------
@app.post("/predict")
def predict_crop_api(data: CropInput):
    crop = predict_crop(
        data.N,
        data.P,
        data.K,
        data.temperature,
        data.humidity,
        data.ph,
        data.rainfall
    )
    return {"recommended_crop": crop}

# -------- Disease Detection API --------
@app.post("/disease")
async def detect_disease(file: UploadFile = File(...)):
    """
    Accepts an image file
    Returns annotated image as Base64 + detection results
    """

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    try:
        # Decode image
        np_img = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Run YOLO inference
        results = disease_model(image, conf=CONF_THRESHOLD, verbose=False)

        detections = []

        # Draw bounding boxes + collect results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = disease_model.names[cls]

                detections.append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "bbox": [x1, y1, x2, y2]
                })

                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = f"{label} {conf:.2f}"
                cv2.putText(
                    image,
                    text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # Encode annotated image
        _, buffer = cv2.imencode(".jpg", image)

        # Convert to Base64
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "image_base64": image_base64,
            "detections": detections
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# # -------- Disease Detection API --------
# @app.post("/disease")
# async def detect_disease(file: UploadFile = File(...)):
#     """
#     Accepts an image file
#     Returns annotated image as Base64
#     """

#     image_bytes = await file.read()

#     if not image_bytes:
#         raise HTTPException(status_code=400, detail="Empty image file")

#     try:
#         # Decode image
#         np_img = np.frombuffer(image_bytes, np.uint8)
#         image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

#         if image is None:
#             raise HTTPException(status_code=400, detail="Invalid image format")

#         # Run YOLO inference
#         results = disease_model(image, conf=CONF_THRESHOLD, verbose=False)

#         # Draw bounding boxes
#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 conf = float(box.conf[0])
#                 cls = int(box.cls[0])
#                 label = disease_model.names[cls]

#                 cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 text = f"{label} {conf:.2f}"
#                 cv2.putText(
#                     image,
#                     text,
#                     (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.6,
#                     (0, 255, 0),
#                     2
#                 )

#         # Encode annotated image
#         _, buffer = cv2.imencode(".jpg", image)

#         # Convert to Base64
#         image_base64 = base64.b64encode(buffer).decode("utf-8")

#         return {
#             "image_base64": image_base64
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
