# import joblib
# import numpy as np
# import os

# # Absolute path to project root
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # Model path
# MODEL_PATH = os.path.join(
#     BASE_DIR, "backend", "crop_recommendation_model.joblib"
# )

# # Load model ONCE
# model = joblib.load(MODEL_PATH)

# def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
#     """
#     Predict crop based on input parameters
#     """
#     input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
#     prediction = model.predict(input_data)
#     return prediction[0]


# # Optional: run directly for testing
# if __name__ == "__main__":
#     result = predict_crop(90, 42, 43, 20.8, 82.0, 6.5, 202.9)
#     print("Predicted Crop:", result)


import cv2
from ultralytics import YOLO

# --------- CONFIG ---------
MODEL_PATH = "/Users/mrunalimalode/Desktop/work/Mrunali/phase2/backend/model/disease.pt"       # path to your YOLOv11n model
IMAGE_PATH = "/Users/mrunalimalode/Desktop/work/Mrunali/phase2/backend/test_data/Early-Leaf-Spot-851-_jpg.rf.482076c67c6fedb569ea3ba643cf451a.jpg"         # path to image
CONF_THRESHOLD = 0.25
# --------------------------

# Load model
model = YOLO(MODEL_PATH)

# Run inference
results = model(
    source=IMAGE_PATH,
    conf=CONF_THRESHOLD,
    verbose=False
)

# Read image using OpenCV
image = cv2.imread(IMAGE_PATH)

# Draw predictions
for result in results:
    boxes = result.boxes

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put label
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

# Show result
cv2.imshow("YOLOv11 Inference", image)
cv2.waitKey(0)
cv2.destroyAllWindows()


