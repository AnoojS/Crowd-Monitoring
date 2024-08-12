import os
import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO

class WeaponDetection:
    def __init__(self, root):
        self.root = root
        self.root.title("Weapon Detection")

        self.capture_button = tk.Button(root, text="Capture Photo", command=self.capture_photo)
        self.capture_button.pack(pady=10)

        self.upload_button = tk.Button(root, text="Upload Photo", command=self.upload_photo)
        self.upload_button.pack(pady=10)

        self.process_button = tk.Button(root, text="Process Image", command=self.process_image)
        self.process_button.pack(pady=10)

        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=10)

        self.confidence_label = tk.Label(root, text="")
        self.confidence_label.pack(pady=10)

        self.canvas = tk.Canvas(root, width=400, height=400)
        self.canvas.pack()

        self.camera = cv2.VideoCapture(1)

    def capture_photo(self):
        ret, frame = self.camera.read()
        if ret:
            cv2.imwrite("captured_photo.png", frame)
            self.display_image("captured_photo.png")

    def upload_photo(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            uploaded_filename = "captured_photo.png"
            self.save_uploaded_photo(file_path, uploaded_filename)
            self.display_image(file_path)

    def save_uploaded_photo(self, source_path, destination_filename):
        with open(source_path, 'rb') as source_file:
            with open(destination_filename, 'wb') as destination_file:
                destination_file.write(source_file.read())

    def process_image(self):

        model = YOLO('yolov8x-cls.pt')

        # Predict with the model
        results = model.predict('captured_photo.png',save=True)

        top1 = results[0].probs.top1
        top1conf = float(results[0].probs.top1conf)
        top1name = results[0].names[results[0].probs.top1]
        
        weapons = [764,763,413]
        if(top1 in weapons):
            self.result_label.config(text= top1name + " Found")
            self.confidence_label.config(text = "Confidence:" + str(top1conf))
        else:
            self.result_label.config(text= "None Found")

        if os.path.exists("captured_photo.png"):
            os.remove("captured_photo.png")

    def display_image(self, image_path):
        image = Image.open(image_path)
        photo = ImageTk.PhotoImage(image)

        self.canvas.config(width=image.width, height=image.height)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.canvas.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = WeaponDetection(root)
    root.mainloop()