import os
import cv2
import torch
import pickle
import numpy as np
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1


class FaceEmbeddingManager:
    def __init__(self,
                 yolo_model_path="weights/best.pt",
                 embedding_file="database/known_embeddings.pkl"):

        lib_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(lib_dir)

        if not os.path.isabs(yolo_model_path):
            yolo_model_path = os.path.join(lib_dir, yolo_model_path)

        if not os.path.isabs(embedding_file):
            embedding_file = os.path.join(project_dir, embedding_file)

        self.model = YOLO(yolo_model_path)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

        self.embedding_file = embedding_file
        os.makedirs(os.path.dirname(self.embedding_file), exist_ok=True)

        if os.path.exists(self.embedding_file):
            with open(self.embedding_file, "rb") as f:
                self.known_embeddings = pickle.load(f)
        else:
            self.known_embeddings = {}

    def get_embedding(self, face_img):
        if face_img is None or face_img.size == 0:
            return None

        face = cv2.resize(face_img, (160, 160))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        face_tensor = torch.tensor(face).permute(2, 0, 1).float() / 255.0
        face_tensor = face_tensor.unsqueeze(0)

        with torch.no_grad():
            embedding = self.resnet(face_tensor).cpu().numpy().flatten()

        embedding = embedding / np.linalg.norm(embedding)
        return embedding

    def process_images(self, image_list):
        embeddings = []

        for img in image_list:
            results = self.model(img)

            if results[0].boxes is None:
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            h, w = img.shape[:2]

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                face = img[y1:y2, x1:x2]
                emb = self.get_embedding(face)

                if emb is not None:
                    embeddings.append(emb)

        return embeddings

    def save_person(self, person_name, embeddings):
        if len(embeddings) == 0:
            return False

        if person_name not in self.known_embeddings:
            self.known_embeddings[person_name] = []

        self.known_embeddings[person_name].extend(embeddings)

        with open(self.embedding_file, "wb") as f:
            pickle.dump(self.known_embeddings, f)

        return True