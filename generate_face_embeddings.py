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
model = YOLO("detection/weights/best.pt")  # YOLOv8 face detector
print("YOLOv8 model loaded successfully.")

resnet = InceptionResnetV1(pretrained='vggface2').eval()
print("FaceNet (InceptionResnetV1) loaded successfully.")

# =========================
# LOAD EXISTING EMBEDDINGS
# =========================
EMBEDDING_FILE = "known_embeddings.pkl"

if os.path.exists(EMBEDDING_FILE):
    with open(EMBEDDING_FILE, "rb") as f:
        known_embeddings = pickle.load(f)
    print("Loaded existing embeddings.")
else:
    known_embeddings = {}
    print("No existing embeddings found. Starting fresh.")

# =========================
# FUNCTION: EXTRACT EMBEDDING
# =========================
def get_embedding(face_img):
    if face_img is None or face_img.size == 0:
        return None

    # Resize to FaceNet input size
    face = cv2.resize(face_img, (160, 160))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Convert to tensor
    face_tensor = torch.tensor(face).permute(2, 0, 1).float() / 255.0
    face_tensor = face_tensor.unsqueeze(0)

    with torch.no_grad():
        embedding = resnet(face_tensor).cpu().numpy().flatten()

    # Normalize embedding
    embedding = embedding / np.linalg.norm(embedding)

    return embedding

# =========================
# MAIN FUNCTION
# =========================
def save_embeddings_from_directory(directory_path):
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        return

    for person_name in os.listdir(directory_path):
        person_path = os.path.join(directory_path, person_name)

        if not os.path.isdir(person_path):
            continue

        print(f"\nProcessing person: {person_name}")
        person_embeddings = []

        for filename in os.listdir(person_path):
            if not filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
                continue

            image_path = os.path.join(person_path, filename)
            img = cv2.imread(image_path)

            if img is None:
                print(f"Cannot read: {image_path}")
                continue

            print(f"  → Processing image: {filename}")

            results = model(img)

            if results[0].boxes is None:
                print("    No face detected.")
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()

            h, w = img.shape[:2]

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                # Clamp coordinates
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                face = img[y1:y2, x1:x2]

                embedding = get_embedding(face)

                if embedding is not None:
                    person_embeddings.append(embedding)

        # Save embeddings
        if len(person_embeddings) > 0:
            if person_name in known_embeddings:
                known_embeddings[person_name].extend(person_embeddings)
            else:
                known_embeddings[person_name] = person_embeddings

            print(f"Saved {len(person_embeddings)} embeddings for {person_name}")
        else:
            print(f"No valid embeddings for {person_name}")

    # Save to file
    with open(EMBEDDING_FILE, "wb") as f:
        pickle.dump(known_embeddings, f)

    print("\nAll embeddings saved successfully!")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    dataset_path = "/home/anhnt298/Downloads/datasets_fr"  # CHANGE THIS
    save_embeddings_from_directory(dataset_path)
