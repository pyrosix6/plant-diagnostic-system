import os
import numpy as np
from PIL import Image
import ai_edge_litert.interpreter as tflite

# 1. โหลดโมเดลและ Labels
interp = tflite.Interpreter(model_path="plant_model.tflite")
interp.allocate_tensors()

with open("labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

# 2. ค้นหาไฟล์ภาพ .jpg หรือ .png ในโฟลเดอร์ให้อัตโนมัติ
img_files = [f for f in os.listdir(".") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not img_files:
    print("ไม่พบไฟล์รูปภาพในโฟลเดอร์โปรเจกต์เลยครับ")
    exit()

test_img_path = img_files[0]
print(f"กำลังทดสอบด้วยไฟล์ภาพ: {test_img_path}")

# 3. โหลดและปรับขนาดภาพ
img = Image.open(test_img_path).convert("RGB")
img_resized = img.resize((224, 224))
img_np = np.array(img_resized, dtype=np.float32)

p_in = interp.get_input_details()[0]['index']
p_out = interp.get_output_details()[0]['index']

# 4. ทดสอบ Normalization 3 รูปแบบ
methods = {
    "Scale [0, 1] (/255.0)": img_np / 255.0,
    "Scale [-1, 1] ((x/127.5)-1)": (img_np / 127.5) - 1.0,
    "Raw Pixel [0, 255]": img_np
}

print("\n--- ผลการทดสอบแต่ละ Preprocessing ---")
for name, inp_data in methods.items():
    inp = np.expand_dims(inp_data, axis=0)
    interp.set_tensor(p_in, inp)
    interp.invoke()
    output = interp.get_tensor(p_out)[0]
    
    # กรณีโมเดลไม่ได้ทำ Softmax ให้ทำเพิ่มเติม
    if np.max(output) > 1.0 or np.min(output) < 0.0:
        exp_preds = np.exp(output - np.max(output))
        output = exp_preds / np.sum(exp_preds)
        
    top_idx = np.argmax(output)
    conf = output[top_idx] * 100.0
    label_name = labels[top_idx] if top_idx < len(labels) else f"Index_{top_idx}"
    print(f"[{name}] -> ทายได้: {label_name} | ความเชื่อมั่น: {conf:.2f}%")