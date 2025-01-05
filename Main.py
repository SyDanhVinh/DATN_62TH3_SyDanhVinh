import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageTk
import cv2

# ====== Load Model ======
model_path = 'D:/DATN/DATN_SyDanhVinh/model.h5'
if os.path.exists(model_path):
    model = load_model(model_path)
    print("Model loaded successfully.")
else:
    print("Model not found. Please train and save the model first.")
    exit()

# ====== Skin Detection Function ======
def is_skin_image(img_path):
    try:
        # Load the image
        img = cv2.imread(img_path)
        if img is None:
            return False

        # Convert to HSV color space
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define skin color range in HSV
        lower_skin = np.array([0, 40, 60], dtype=np.uint8)  # Adjusted lower bound
        upper_skin = np.array([50, 150, 255], dtype=np.uint8)  # Adjusted upper bound

        # Create a mask for skin tones
        skin_mask = cv2.inRange(hsv_img, lower_skin, upper_skin)

        # Calculate the percentage of skin pixels
        skin_percentage = np.sum(skin_mask > 0) / (skin_mask.size) * 100

        # If more than 10% of the pixels match skin tones, classify as skin
        return skin_percentage > 10
    except Exception as e:
        print("Error in is_skin_image:", e)
        return False

# ====== Predict Image Function ======
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(112, 112))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    return "Ác tính" if prediction[0][0] > 0.5 else "Lành tính"

# ====== GUI Application ======
def open_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        display_image(file_path)

        # Check if the image is of human skin
        if is_skin_image(file_path):
            result = predict_image(file_path)
            result_label.config(text=f"Dự đoán: {result}")
        else:
            result_label.config(text="Ảnh không phải là da người.")

def display_image(img_path):
    img = Image.open(img_path)
    img = img.resize((400, 400))  # Resize for display
    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

# ====== Setup Tkinter Window ======
root = tk.Tk()
root.title("Dự đoán ung thư da")

# Get screen dimensions
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Window size and position (centered)
window_width = 600
window_height = 500
x_cordinate = int((screen_width / 2) - (window_width / 2))
y_cordinate = int((screen_height / 2) - (window_height / 2))
root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

# Add padding
root.config(padx=20, pady=20)

# GUI Components
image_label = Label(root)
image_label.pack()

result_label = Label(root, text="Dự đoán: ", font=("Arial", 14))
result_label.pack()

upload_button = Button(root, text="Tải ảnh", command=open_file)
upload_button.pack()

# Run the application
root.mainloop()