import os
import cv2
import shutil
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

from libs.face_embedding_lib import FaceEmbeddingManager


class FaceCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Capture System")

        self.cap = cv2.VideoCapture(0)

        self.manager = FaceEmbeddingManager()

        self.name_var = tk.StringVar()
        self.capture_count = 0
        self.max_capture = 8
        self.captured_images = []
        self.lock_name = False

        # UI
        self.label = tk.Label(root)
        self.label.pack()

        self.entry = tk.Entry(root, textvariable=self.name_var, font=("Arial", 14))
        self.entry.pack()

        self.btn_capture = tk.Button(root, text="Capture", command=self.capture_image)
        self.btn_capture.pack()

        self.update_frame()

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = frame.copy()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def capture_image(self):
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
        dataset_dir = os.path.join(os.getcwd(), "dataset", name)

        os.makedirs(dataset_dir, exist_ok=True)

        # Save images
        for i, img in enumerate(self.captured_images):
            path = os.path.join(dataset_dir, f"img{i+1}.jpg")
            cv2.imwrite(path, img)

        # Generate embeddings
        embeddings = self.manager.process_images(self.captured_images)

        success = self.manager.save_person(name, embeddings)

        # Remove dataset folder
        shutil.rmtree(dataset_dir)

        if success:
            messagebox.showinfo("Done", f"{name} saved successfully!")
        else:
            messagebox.showerror("Error", "No valid face detected!")

        self.reset()

    def reset(self):
        self.capture_count = 0
        self.captured_images = []
        self.lock_name = False
        self.entry.config(state='normal')
        self.name_var.set("")


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceCaptureApp(root)
    root.mainloop()