import os
import cv2
import shutil
import sys
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

# Allow running this file directly: `python main.py`
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
MY_PROJECT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from my_project.libs.face_embedding_lib import FaceEmbeddingManager


class FaceCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Capture & Recognition")

        self.cap = cv2.VideoCapture(0)

        embedding_file = os.path.join(MY_PROJECT_DIR, "database", "known_embeddings.pkl")
        self.manager = FaceEmbeddingManager(embedding_file=embedding_file)

        # Capture tab state
        self.name_var = tk.StringVar()
        self.capture_count = 0
        self.max_capture = 8
        self.captured_images = []
        self.lock_name = False
        self.capture_status_var = tk.StringVar(value=f"Captured: 0/{self.max_capture}")

        # Detect tab state
        self.recognition_var = tk.StringVar(value="Detected: Unknown")
        self.score_var = tk.StringVar(value="Score: -")
        self.recognition_threshold = 0.6

        # UI with 2 tabs
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True)

        self.capture_tab = ttk.Frame(self.notebook)
        self.detect_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.capture_tab, text="Capture Face ID")
        self.notebook.add(self.detect_tab, text="Detect Person")

        self._build_capture_tab()
        self._build_detect_tab()

        self.current_frame = None
        self.update_frame()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_capture_tab(self):
        self.capture_label = tk.Label(self.capture_tab)
        self.capture_label.pack(padx=10, pady=10)

        self.entry = tk.Entry(self.capture_tab, textvariable=self.name_var, font=("Arial", 14))
        self.entry.pack(pady=4)

        self.btn_capture = tk.Button(self.capture_tab, text="Capture", command=self.capture_image)
        self.btn_capture.pack(pady=4)

        self.capture_status = tk.Label(
            self.capture_tab,
            textvariable=self.capture_status_var,
            font=("Arial", 12)
        )
        self.capture_status.pack(pady=4)

    def _build_detect_tab(self):
        self.detect_label = tk.Label(self.detect_tab)
        self.detect_label.pack(padx=10, pady=10)

        self.detect_result = tk.Label(
            self.detect_tab,
            textvariable=self.recognition_var,
            font=("Arial", 14, "bold")
        )
        self.detect_result.pack(pady=2)

        self.detect_score = tk.Label(
            self.detect_tab,
            textvariable=self.score_var,
            font=("Arial", 12)
        )
        self.detect_score.pack(pady=2)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(20, self.update_frame)
            return

        self.current_frame = frame.copy()
        self._update_capture_preview(frame)
        self._update_detect_preview(frame)

        self.root.after(15, self.update_frame)

    def _update_capture_preview(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.capture_label.imgtk = imgtk
        self.capture_label.configure(image=imgtk)

    def _update_detect_preview(self, frame):
        annotated = frame.copy()
        detected_name = "Unknown"
        best_score = -1.0

        results = self.manager.model(frame)
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            h, w = frame.shape[:2]

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                face = frame[y1:y2, x1:x2]
                emb = self.manager.get_embedding(face)
                name, score = self.recognize_face(emb)

                if score > best_score:
                    best_score = score
                    detected_name = name

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    name,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

        self.recognition_var.set(f"Detected: {detected_name}")
        if best_score < 0:
            self.score_var.set("Score: -")
        else:
            self.score_var.set(f"Score: {best_score:.3f}")

        rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        self.detect_label.imgtk = imgtk
        self.detect_label.configure(image=imgtk)

    def capture_image(self):
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No camera frame available yet!")
            return

        name = self.name_var.get().strip()

        if not name:
            messagebox.showwarning("Warning", "Enter name first!")
            return

        if not self.lock_name:
            self.lock_name = True
            self.entry.config(state='disabled')

        if self.capture_count >= self.max_capture:
            return

        self.captured_images.append(self.current_frame.copy())
        self.capture_count += 1
        self.capture_status_var.set(f"Captured: {self.capture_count}/{self.max_capture}")

        print(f"Captured {self.capture_count}/8")

        if self.capture_count == self.max_capture:
            self.ask_save()

    def ask_save(self):
        result = messagebox.askyesno("Save", "Save this person?")

        if result:
            self.save_data()
        else:
            self.reset()

    def save_data(self):
        name = self.name_var.get().strip()
        dataset_root = os.path.join(os.getcwd(), "dataset")
        dataset_dir = os.path.join(dataset_root, name)

        os.makedirs(dataset_dir, exist_ok=True)

        # Save images
        for i, img in enumerate(self.captured_images):
            path = os.path.join(dataset_dir, f"img{i+1}.jpg")
            cv2.imwrite(path, img)

        # Generate embeddings
        embeddings = self.manager.process_images(self.captured_images)

        success = self.manager.save_person(name, embeddings)
        if success:
            # Keep in-memory embeddings fresh for tab 2 recognition immediately.
            self.manager.known_embeddings = self.manager._clean_embeddings(self.manager.known_embeddings)

        # Remove full dataset folder (it will be recreated next time)
        if os.path.exists(dataset_root):
            shutil.rmtree(dataset_root)

        if success:
            messagebox.showinfo("Done", f"{name} saved successfully!")
        else:
            messagebox.showerror("Error", "No valid face detected!")

        self.reset()

    def reset(self):
        self.capture_count = 0
        self.captured_images = []
        self.lock_name = False
        self.capture_status_var.set(f"Captured: 0/{self.max_capture}")
        self.entry.config(state='normal')
        self.name_var.set("")

    def recognize_face(self, embedding):
        if embedding is None:
            return "Unknown", -1.0

        best_match = "Unknown"
        best_score = -1.0

        for name, emb_list in self.manager.known_embeddings.items():
            for known_emb in emb_list:
                score = float(embedding.dot(known_emb))
                if score > best_score:
                    best_score = score
                    best_match = name

        if best_score < self.recognition_threshold:
            return "Unknown", best_score

        return best_match, best_score

    def on_close(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceCaptureApp(root)
    root.mainloop()