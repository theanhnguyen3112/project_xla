import os
import cv2
import torch
import pickle
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1

# =========================
# INIT MODELS
# =========================
model = YOLO("detection/weights/best.pt")
print("YOLOv8 model loaded successfully.")

resnet = InceptionResnetV1(pretrained='vggface2').eval()
print("FaceNet model loaded successfully.")

# =========================
# LOAD EMBEDDINGS
# =========================
def load_known_embeddings():
    try:
        with open("known_embeddings.pkl", "rb") as f:
            data = pickle.load(f)
        print("Known embeddings loaded successfully.")
    except:
        data = {}
        print("No embeddings found.")

    # CLEAN DATA
    cleaned = {}
    for name, emb_list in data.items():
        valid = []
        for emb in emb_list:
            if isinstance(emb, np.ndarray) and emb.shape == (512,):
                valid.append(emb / np.linalg.norm(emb))
        if len(valid) > 0:
            cleaned[name] = valid

    return cleaned

known_embeddings = load_known_embeddings()

# =========================
# EMBEDDING FUNCTION
# =========================
def get_embedding(face):
    if face is None or face.size == 0:
        return None

    face = cv2.resize(face, (160, 160))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    face_tensor = torch.tensor(face).permute(2, 0, 1).float() / 255.0
    face_tensor = face_tensor.unsqueeze(0)

    with torch.no_grad():
        emb = resnet(face_tensor).cpu().numpy().flatten()

    emb = emb / np.linalg.norm(emb)
    return emb

# =========================
# COSINE SIMILARITY
# =========================
def cosine_similarity(a, b):
    return np.dot(a, b)

# =========================
# COMPARE
# =========================
def recognize_face(embedding, known_embeddings, threshold=0.6):
    best_match = "Unknown"
    best_score = -1

    for name, emb_list in known_embeddings.items():
        for known_emb in emb_list:

            if known_emb is None:
                continue

            score = cosine_similarity(embedding, known_emb)

            if score > best_score:
                best_score = score
                best_match = name

    if best_score < threshold:
        best_match = "Unknown"

    print(f"Best score: {best_score:.3f} → {best_match}")
    return best_match

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            face = frame[y1:y2, x1:x2]

            embedding = get_embedding(face)

            if embedding is None:
                continue

            name = recognize_face(embedding, known_embeddings)

            # DRAW
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
