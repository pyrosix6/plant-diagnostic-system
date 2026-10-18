import ai_edge_litert.interpreter as tflite
import numpy as np

# 1. โหลดโมเดล
interp = tflite.Interpreter(model_path="plant_model.tflite")
interp.allocate_tensors()

# 2. อ่านไฟล์ labels ปัจจุบัน
with open("labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f if line.strip()]

print(f"จำนวนคลาสใน labels.txt: {len(labels)} คลาส")
print("--- ลำดับ Labels ปัจจุบัน (10 คลาสแรก) ---")
for idx, label in enumerate(labels[:10]):
    print(f"Index {idx}: {label}")