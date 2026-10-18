import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image

try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite

# 1. เปิดหน้าต่างเลือกไฟล์ภาพ
root = tk.Tk()
root.withdraw()
print("กรุณาเลือกไฟล์รูปภาพที่ต้องการทดสอบในหน้าต่าง Popup...")
image_path = filedialog.askopenfilename(
    title="เลือกรูปภาพใบพืช",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.JPG *.PNG")]
)

if not image_path:
    print("คุณไม่ได้เลือกไฟล์ ยกเลิกการทดสอบ")
    exit()

print(f"กำลังทดสอบไฟล์: {os.path.basename(image_path)}")

# 2. โหลดโมเดลและ labels
interp = tflite.Interpreter(model_path="plant_model.tflite")
interp.allocate_tensors()

with open("labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

# 3. Preprocessing
img = Image.open(image_path).convert("RGB")
img_resized = img.resize((224, 224))
img_np = np.array(img_resized, dtype=np.float32) / 255.0
inp = np.expand_dims(img_np, axis=0)

# 4. Predict
p_in = interp.get_input_details()[0]['index']
p_out = interp.get_output_details()[0]['index']

interp.set_tensor(p_in, inp)
interp.invoke()
output = interp.get_tensor(p_out)[0]

# 5. แสดงผลคะแนนทั้ง 26 คลาส
print("\n" + "="*60)
print(f"--- ผลคะแนนทั้ง 26 Index สำหรับภาพ: {os.path.basename(image_path)} ---")
print("="*60)

# เรียงลำดับจากคะแนนมากไปน้อย
top_indices = np.argsort(output)[::-1]

for idx in top_indices:
    score = output[idx] * 100.0
    label_name = labels[idx] if idx < len(labels) else "Unknown"
    print(f"Index {idx:2d} | Score: {score:6.2f}% | Label ปัจจุบัน: {label_name}")