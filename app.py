import base64
import io
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import requests
import streamlit as st

# 1. โหลด TFLite Interpreter
try:
  import ai_edge_litert.interpreter as tflite
except ImportError:
  try:
    import tflite_runtime.interpreter as tflite
  except ImportError:
    import tensorflow.lite as tflite

# ตกแต่ง UI ด้วย Custom CSS
st.markdown(
    """
    <style>
    .main { background-color: #0f172a; }
    .stApp { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
    .card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .card-success { border-left: 5px solid #10b981; }
    .card-warning { border-left: 5px solid #f59e0b; }
    .card-danger { border-left: 5px solid #ef4444; }
    .card-title { font-size: 0.9rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; margin-bottom: 6px; }
    .card-value { font-size: 1.1rem; font-weight: 500; color: #f8fafc; }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# State สลับภาษาและบันทึกประวัติ
if "lang" not in st.session_state:
  st.session_state.lang = "TH"

if "detection_history" not in st.session_state:
  st.session_state.detection_history = []


# ฟังก์ชันดึงสภาพอากาศจาก Open-Meteo API
def get_weather_spray_advice(lat, lon):
  url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,precipitation,rain,wind_speed_10m&hourly=precipitation_probability&forecast_days=1&timezone=auto"
  try:
    response = requests.get(url, timeout=5)
    data = response.json()

    current = data.get("current", {})
    hourly = data.get("hourly", {})

    temp = current.get("temperature_2m", 0)
    humidity = current.get("relative_humidity_2m", 0)
    wind_speed = current.get("wind_speed_10m", 0)
    rain_prob = (
        hourly.get("precipitation_probability", [0])[0]
        if hourly.get("precipitation_probability")
        else 0
    )

    reasons = []
    is_suitable = True

    if wind_speed > 15:
      is_suitable = False
      reasons.append(
          f"ความเร็วลมสูงเกินไป ({wind_speed} km/h) อาจทำให้ยาฟุ้งกระจาย"
      )

    if rain_prob > 30:
      is_suitable = False
      reasons.append(
          f"โอกาสฝนตกสูง ({rain_prob}%) สารเคมีอาจถูกชะล้าง"
      )

    if humidity < 40:
      reasons.append(
          f"อากาศแห้งเกินไป ({humidity}%) ตัวยาอาจระเหยเร็วเกินไป"
      )
    elif humidity > 85:
      reasons.append(
          f"ความชื้นสูงเกินไป ({humidity}%) ละอองยาอาจไม่เกาะใบ"
      )

    return {
        "status": "SUCCESS",
        "is_suitable": is_suitable,
        "temp": temp,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "rain_prob": rain_prob,
        "reasons": reasons,
    }
  except Exception as e:
    return {"status": "ERROR", "message": str(e)}


# Dictionary ข้อความสองภาษา
T = {
    "TH": {
        "title": "ระบบตรวจวินิจฉัยโรคพืชและคำนวณสัดส่วนยา",
        "sidebar_control": "แผงควบคุม (Control Panel)",
        "select_plant": "1. เลือกชนิดพืช",
        "input_mode": "เลือกช่องทางรับภาพใบพืช",
        "mode_upload": "อัปโหลดรูปภาพ",
        "mode_camera": "ถ่ายภาพด้วยกล้องสด",
        "upload_img": "เลือกรูปภาพใบพืช",
        "camera_img": "ส่องกล้องถ่ายภาพใบพืช",
        "gps_label": "2. ระบุพิกัด GPS แปลงเกษตร (ถ้ามี)",
        "weather_header": "สภาพอากาศสำหรับการฉีดพ่น",
        "weather_ok": "สภาวะเหมาะสมในการฉีดพ่นยา",
        "weather_not_ok": "ไม่แนะนำให้ฉีดพ่นยาในขณะนี้",
        "temp_label": "อุณหภูมิ",
        "wind_label": "ความเร็วลม",
        "humidity_label": "ความชื้น",
        "rain_label": "โอกาสฝน",
        "caution_label": "ข้อควรระวัง:",
        "btn_analyze": "เริ่มวิเคราะห์โรคพืช",
        "calc_header": "เครื่องคำนวณสัดส่วนยาสเปรย์พ่น",
        "select_chem": "เลือกชนิดตัวยาหลัก:",
        "mix_chem": "เลือกชนิดตัวยาที่จะผสมร่วม (Optional):",
        "chem_advice": "คำแนะนำการใช้ยา:",
        "area_rai": "ขนาดพื้นที่ (ไร่)",
        "area_ngan": "ขนาดพื้นที่ (งาน)",
        "dose_label": "ปริมาณยาต่อน้ำ 20 ลิตร",
        "water_label": "ปริมาณน้ำพ่นต่อไร่ (ลิตร)",
        "calc_result": "ปริมาณยารวมที่ต้องใช้:",
        "water_result": "ปริมาณน้ำที่ต้องเตรียมผสม:",
        "history": "ประวัติการวิเคราะห์",
        "stat_header": "สถิติการตรวจพบโรค",
        "result_header": "ผลการตรวจวินิจฉัยโรค",
        "top3_title": "กราฟความน่าจะเป็น Top 3 (Confidence Score)",
        "export_pdf": "ส่งออกรายงาน (PDF / Print)",
        "btn_print": "พิมพ์รายงานฉบับเต็ม / บันทึกเป็น PDF",
        "lang_select": "เลือกภาษา / Language",
        "theme_info": "สลับ Dark/Light Mode ได้ที่เมนู Settings",
        "med_title": "กลุ่มตัวยาแนะนำ",
        "action_title": "ขั้นตอนการจัดการ",
        "safety_title": "ข้อมูลความปลอดภัยและการเกษตร (Safety & PHI)",
        "phi_label": "ระยะเว้นการเก็บเกี่ยว (PHI)",
        "ppe_label": "อุปกรณ์ป้องกันที่ต้องสวมใส่ (PPE)",
        "btn_back_home": "กลับสู่หน้าหลัก",
    },
    "EN": {
        "title": "Plant Disease Diagnostic & Spray Dosage System",
        "sidebar_control": "Control Panel",
        "select_plant": "1. Select Plant Type",
        "input_mode": "Select Image Input Source",
        "mode_upload": "Upload Image",
        "mode_camera": "Live Camera",
        "upload_img": "Upload Leaf Image",
        "camera_img": "Capture Leaf Image with Camera",
        "gps_label": "2. Field GPS Location (Optional)",
        "weather_header": "Weather Condition for Spraying",
        "weather_ok": "Suitable conditions for spraying",
        "weather_not_ok": "Not recommended for spraying at this time",
        "temp_label": "Temperature",
        "wind_label": "Wind Speed",
        "humidity_label": "Humidity",
        "rain_label": "Rain Chance",
        "caution_label": "Caution:",
        "btn_analyze": "Start Analysis",
        "calc_header": "Spray Dosage Calculator",
        "select_chem": "Select Primary Chemical:",
        "mix_chem": "Select Secondary Chemical to Mix (Optional):",
        "chem_advice": "Chemical Advice:",
        "area_rai": "Area (Rai)",
        "area_ngan": "Area (Ngan)",
        "dose_label": "Dosage per 20L Water",
        "water_label": "Water per Rai (Liters)",
        "calc_result": "Total Required Chemical:",
        "water_result": "Total Water Required:",
        "history": "Analysis History",
        "stat_header": "Disease Statistics",
        "result_header": "Diagnostic Result",
        "top3_title": "Top 3 Confidence Scores",
        "export_pdf": "Export Report (PDF / Print)",
        "btn_print": "Print Full Report / Save as PDF",
        "lang_select": "Language / เลือกภาษา",
        "theme_info": "Switch Dark/Light Mode in Settings",
        "med_title": "Recommended Treatment",
        "action_title": "Management Protocol",
        "safety_title": "Agriculture & Safety Info (PHI & PPE)",
        "phi_label": "Pre-Harvest Interval (PHI)",
        "ppe_label": "Required Protective Equipment (PPE)",
        "btn_back_home": "Back to Main Menu",
    },
}

# ฐานข้อมูลตัวยาทางการเกษตร + PHI, PPE
CHEMICAL_DB = {
    "กำหนดเอง (Custom Input)": {
        "dose": 30.0,
        "water_rai": 80.0,
        "unit": "กรัม/มล.",
        "type": "custom",
        "phi": "ระบุตามฉลาก",
        "ppe": "หน้ากาก, ถุงมือยาง",
        "note_th": "ระบุอัตราส่วนตามฉลากข้างขวดด้วยตนเอง",
        "note_en": "Manually enter dosage from product label",
    },
    "ไม่มีการผสมเพิ่ม (None)": {
        "dose": 0.0,
        "water_rai": 80.0,
        "unit": "กรัม",
        "type": "none",
        "phi": "-",
        "ppe": "-",
        "note_th": "-",
        "note_en": "-",
    },
    "Mancozeb (แมนโคเซ็บ)": {
        "dose": 40.0,
        "water_rai": 80.0,
        "unit": "กรัม",
        "type": "contact",
        "phi": "7-14 วัน",
        "ppe": "หน้ากากกันสารเคมี, ถุงมือยาง, แว่นตากันสารเคมี",
        "note_th": (
            "ยาสัมผัส ใช้ 30-50 กรัม/น้ำ 20 ลิตร (ฉีดพ่นให้ทั่วทั้งใบ)"
        ),
        "note_en": "Contact fungicide. 30-50g per 20L water.",
    },
    "Chlorothalonil (คลอโรทาโลนิล)": {
        "dose": 30.0,
        "water_rai": 80.0,
        "unit": "กรัม",
        "type": "contact",
        "phi": "14 วัน",
        "ppe": "หน้ากาก, ถุงมือยาง, เสื้อแขนยาว",
        "note_th": "ยาสัมผัส ป้องกันราน้ำฝน ใช้ 30 กรัม/น้ำ 20 ลิตร",
        "note_en": "Contact fungicide. 30g per 20L water.",
    },
    "Carbendazim (คาร์เบนดาซิม)": {
        "dose": 20.0,
        "water_rai": 60.0,
        "unit": "กรัม",
        "type": "systemic",
        "phi": "14-21 วัน",
        "ppe": "หน้ากากกันสารเคมี, ถุงมือยาง",
        "note_th": (
            "ยาดูดซึม ป้องกันราแป้ง/ใบจุด ใช้ 20 กรัม/น้ำ 20 ลิตร"
        ),
        "note_en": "Systemic fungicide. 20g per 20L water.",
    },
    "Azoxystrobin (อะซอกซีสโตรบิน)": {
        "dose": 15.0,
        "water_rai": 60.0,
        "unit": "มล.",
        "type": "systemic",
        "phi": "3-7 วัน",
        "ppe": "หน้ากาก, ถุงมือยาง, แว่นตากันสารเคมี",
        "note_th": (
            "ยาดูดซึมเข้มข้นสูง ใช้เพียง 10-15 มล./น้ำ 20 ลิตร"
        ),
        "note_en": "Concentrated systemic. 10-15 ml per 20L water.",
    },
    "Copper Hydroxide (คอปเปอร์)": {
        "dose": 30.0,
        "water_rai": 80.0,
        "unit": "กรัม",
        "type": "copper",
        "phi": "7 วัน",
        "ppe": (
            "หน้ากาก, ถุงมือยาง, แว่นตากันสารเคมี, รองเท้าบูท"
        ),
        "note_th": (
            "กำจัดแคงเกอร์/แบคทีเรีย ใช้ 30 กรัม/น้ำ 20 ลิตร"
            " (เลี่ยงผสมยาที่เป็นกรด)"
        ),
        "note_en": "Bactericide/Fungicide. 30g per 20L water.",
    },
}

lang = st.session_state.lang

# รายชื่อคลาสพืช
plant_classes = {
    "Banana": ["banana_cordana", "banana_pestalotiopsis", "banana_sigatoka"],
    "Corn": [
        "corn_gray_leaf_spot",
        "corn_healthy",
        "corn_northern_blight",
        "corn_rust",
    ],
    "Grape": [
        "grape_black_rot",
        "grape_esca",
        "grape_healthy",
        "grape_leaf_blight",
    ],
    "Pepper": ["pepper_bacterial_spot", "pepper_healthy"],
    "Potato": ["potato_early_blight", "potato_healthy", "potato_late_blight"],
    "Rice": [
        "rice_blast",
        "rice_brown_spot",
        "rice_healthy",
        "rice_leaf_blight",
        "rice_tungro",
    ],
    "Tomato": [
        "tomato_bacterial_spot",
        "tomato_early_blight",
        "tomato_late_blight",
        "tomato_leaf_mold",
        "tomato_septoria",
    ],
}

# คำแนะนำแบบครบทุกโรค
advice_data = {
    "banana_sigatoka": {
        "medicine": {
            "TH": (
                "Mancozeb (แมนโคเซ็บ) หรือ Propiconazole (โพรพิโคนาโซล)"
            ),
            "EN": "Mancozeb or Propiconazole",
        },
        "action": {
            "TH": (
                "ตัดใบที่เป็นโรคเผาทำลายทันที ฉีดพ่น Mancozeb 30-50 กรัม/น้ำ 20"
                " ลิตร ทุก 7-10 วัน และปรับระยะต้นให้โปร่ง"
            ),
            "EN": (
                "Prune and burn infected leaves immediately. Spray Mancozeb"
                " 30-50g per 20L water every 7-10 days."
            ),
        },
    },
    "banana_cordana": {
        "medicine": {
            "TH": "Chlorothalonil (คลอโรทาโลนิล) หรือ Difenoconazole",
            "EN": "Chlorothalonil or Difenoconazole",
        },
        "action": {
            "TH": (
                "ตัดแต่งใบใกล้วงโคนออก พ่น Chlorothalonil 30 กรัม/น้ำ 20 ลิตร"
                " ให้ทั่วทั้งบนใบและใต้ใบ"
            ),
            "EN": (
                "Prune lower leaves. Apply Chlorothalonil 30g per 20L water on"
                " both upper and lower leaf surfaces."
            ),
        },
    },
    "banana_pestalotiopsis": {
        "medicine": {
            "TH": "Carbendazim (คาร์เบนดาซิม)",
            "EN": "Carbendazim",
        },
        "action": {
            "TH": (
                "ตัดส่วนเนื้อเยื่อใบที่แห้งตายออก พ่น Carbendazim 20 กรัม/น้ำ 20"
                " ลิตร ทุก 10 วัน"
            ),
            "EN": (
                "Remove dried leaf tissues. Apply Carbendazim 20g per 20L water"
                " every 10 days."
            ),
        },
    },
    "rice_healthy": {
        "medicine": {
            "TH": "ไม่จำเป็นต้องใช้ยาเคมี",
            "EN": "No chemical treatment required",
        },
        "action": {
            "TH": (
                "คุมระดับน้ำในนา 5-10 ซม."
                " หลีกเลี่ยงการใส่ปุ๋ยยูเรียเกินความจำเป็น"
            ),
            "EN": (
                "Maintain water level at 5-10 cm. Avoid excessive nitrogen"
                " fertilizer usage."
            ),
        },
    },
    "rice_blast": {
        "medicine": {
            "TH": "Tricyclazole (ไตรไซคลาโซล)",
            "EN": "Tricyclazole",
        },
        "action": {
            "TH": (
                "งดใส่ปุ๋ยไนโตรเจนทันที พ่น Tricyclazole 10-15 กรัม/น้ำ 20 ลิตร"
            ),
            "EN": (
                "Halt nitrogen fertilization immediately. Spray Tricyclazole"
                " 10-15g per 20L water."
            ),
        },
    },
    "rice_brown_spot": {
        "medicine": {"TH": "Mancozeb (แมนโคเซ็บ)", "EN": "Mancozeb"},
        "action": {
            "TH": (
                "ใส่ปุ๋ยโพแทสเซียม (0-0-60) เพิ่มความแข็งแรง พ่น Mancozeb 30"
                " กรัม/น้ำ 20 ลิตร"
            ),
            "EN": (
                "Apply potassium fertilizer (0-0-60). Spray Mancozeb 30g per 20L"
                " water."
            ),
        },
    },
    "rice_leaf_blight": {
        "medicine": {
            "TH": "Copper Hydroxide (คอปเปอร์ ไฮดรอกไซด์)",
            "EN": "Copper Hydroxide",
        },
        "action": {
            "TH": (
                "ห้ามระบายน้ำข้ามแปลงจากจุดที่ระบาด พ่นสารกลุ่มคอปเปอร์ 30"
                " กรัม/น้ำ 20 ลิตร"
            ),
            "EN": (
                "Do not drain water to other fields. Spray Copper Hydroxide 30g"
                " per 20L water."
            ),
        },
    },
    "rice_tungro": {
        "medicine": {
            "TH": "Imidacloprid (อิมิดาโคลพริด)",
            "EN": "Imidacloprid",
        },
        "action": {
            "TH": (
                "ถอนต้นที่เป็นโรคเผาทิ้ง พ่น Imidacloprid"
                " กำจัดเพลี้ยจักจั่นสีเขียว"
            ),
            "EN": (
                "Uproot infected plants and destroy them. Spray Imidacloprid to"
                " control leafhoppers."
            ),
        },
    },
    "corn_healthy": {
        "medicine": {
            "TH": "ไม่จำเป็นต้องใช้ยาเคมี",
            "EN": "No chemical treatment required",
        },
        "action": {
            "TH": "พูนโคนต้น และตรวจดูหนอนกระทู้ตามฤดูกาล",
            "EN": (
                "Keep soil well-drained and inspect regularly for fall"
                " armyworms."
            ),
        },
    },
    "corn_rust": {
        "medicine": {
            "TH": "Azoxystrobin + Difenoconazole",
            "EN": "Azoxystrobin + Difenoconazole",
        },
        "action": {
            "TH": "พ่นสาร Azoxystrobin อัตรา 10-15 ซีซี/น้ำ 20 ลิตร",
            "EN": (
                "Apply Azoxystrobin at 10-15 ml per 20L water immediately upon"
                " pustule detection."
            ),
        },
    },
    "corn_gray_leaf_spot": {
        "medicine": {
            "TH": "Carbendazim (คาร์เบนดาซิม)",
            "EN": "Carbendazim",
        },
        "action": {
            "TH": (
                "ตัดแต่งใบช่วงล่างออก พ่น Carbendazim 20 กรัม/น้ำ 20 ลิตร ทุก 7"
                " วัน"
            ),
            "EN": (
                "Prune lower foliage. Spray Carbendazim 20g per 20L water every"
                " 7 days."
            ),
        },
    },
    "corn_northern_blight": {
        "medicine": {"TH": "Mancozeb (แมนโคเซ็บ)", "EN": "Mancozeb"},
        "action": {
            "TH": (
                "พ่น Mancozeb 30 กรัม/น้ำ 20 ลิตร และเผาทำลายตอซังหลังเก็บเกี่ยว"
            ),
            "EN": (
                "Apply Mancozeb 30g per 20L water. Burn crop residues after"
                " harvesting."
            ),
        },
    },
    "tomato_healthy": {
        "medicine": {
            "TH": "ไม่จำเป็นต้องใช้ยาเคมี",
            "EN": "No chemical treatment required",
        },
        "action": {
            "TH": "ให้น้ำโคนต้น ทำค้างยกใบสูงพ้นพื้นดิน",
            "EN": (
                "Water at soil level and utilize trellises to elevate foliage"
                " above ground."
            ),
        },
    },
    "tomato_early_blight": {
        "medicine": {
            "TH": "Chlorothalonil (คลอโรทาโลนิล)",
            "EN": "Chlorothalonil",
        },
        "action": {
            "TH": (
                "ปลิดใบโคนต้นที่เป็นแผลทิ้ง พ่น Chlorothalonil 30 กรัม/น้ำ 20"
                " ลิตร"
            ),
            "EN": (
                "Remove lower infected leaves. Apply Chlorothalonil 30g per 20L"
                " water."
            ),
        },
    },
    "tomato_late_blight": {
        "medicine": {"TH": "Metalaxyl (เมทาแลกซิล)", "EN": "Metalaxyl"},
        "action": {
            "TH": (
                "พ่น Metalaxyl 30 กรัม/น้ำ 20 ลิตร ทันที งดการให้น้ำแบบฝอย"
            ),
            "EN": (
                "Apply Metalaxyl 30g per 20L water immediately. Avoid overhead"
                " irrigation."
            ),
        },
    },
    "tomato_leaf_mold": {
        "medicine": {"TH": "Copper Oxychloride", "EN": "Copper Oxychloride"},
        "action": {
            "TH": (
                "ตัดแต่งทรงพุ่ม พ่น Copper Oxychloride 40 กรัม/น้ำ 20 ลิตร"
                " เน้นใต้ใบ"
            ),
            "EN": (
                "Prune canopy for airflow. Spray Copper Oxychloride 40g per 20L"
                " water under leaves."
            ),
        },
    },
    "tomato_septoria": {
        "medicine": {"TH": "Mancozeb", "EN": "Mancozeb"},
        "action": {
            "TH": (
                "เก็บเศษใบตกค้างบนพื้นดินออก และพ่น Mancozeb 30 กรัม/น้ำ 20"
                " ลิตร"
            ),
            "EN": "Clear fallen leaf debris. Spray Mancozeb 30g per 20L water.",
        },
    },
    "tomato_bacterial_spot": {
        "medicine": {
            "TH": "Copper Hydroxide + Kasugamycin",
            "EN": "Copper Hydroxide + Kasugamycin",
        },
        "action": {
            "TH": (
                "พ่นสารกลุ่มทองแดงผสมคาสูกาไมซิน"
                " เลี่ยงการทำงานตอนใบพืชเปียก"
            ),
            "EN": (
                "Apply Copper Hydroxide combined with Kasugamycin. Avoid"
                " handling wet foliage."
            ),
        },
    },
    "potato_healthy": {
        "medicine": {
            "TH": "ไม่จำเป็นต้องใช้ยาเคมี",
            "EN": "No chemical treatment required",
        },
        "action": {
            "TH": "พูนโคนดินสม่ำเสมอ คุมความชื้นดินให้พอเหมาะ",
            "EN": (
                "Mound soil around stems regularly and maintain appropriate"
                " soil moisture."
            ),
        },
    },
    "potato_early_blight": {
        "medicine": {"TH": "Chlorothalonil", "EN": "Chlorothalonil"},
        "action": {
            "TH": "พ่น Chlorothalonil 30 กรัม/น้ำ 20 ลิตร ทุก 7 วัน",
            "EN": "Spray Chlorothalonil 30g per 20L water every 7 days.",
        },
    },
    "potato_late_blight": {
        "medicine": {
            "TH": "Cymoxanil + Mancozeb",
            "EN": "Cymoxanil + Mancozeb",
        },
        "action": {
            "TH": "พ่น Cymoxanil + Mancozeb 30 กรัม/น้ำ 20 ลิตร",
            "EN": "Apply Cymoxanil + Mancozeb 30g per 20L water.",
        },
    },
    "grape_healthy": {
        "medicine": {
            "TH": "ไม่จำเป็นต้องใช้ยาเคมี",
            "EN": "No chemical treatment required",
        },
        "action": {
            "TH": "จัดทรงพุ่มให้เปิดรับแสงแดดทั่วถึง",
            "EN": "Prune vine canopy to ensure adequate sunlight exposure.",
        },
    },
    "grape_black_rot": {
        "medicine": {
            "TH": "Myclobutanil (ไมโคลบิวทานิล)",
            "EN": "Myclobutanil",
        },
        "action": {
            "TH": "เก็บผลเน่าแห้งเผาทิ้ง พ่น Myclobutanil ป้องกันช่วงติดผล",
            "EN": (
                "Remove mummified berries. Spray Myclobutanil during fruit set"
                " period."
            ),
        },
    },
    "grape_esca": {
        "medicine": {
            "TH": "ปูนแดง + Fosetyl-Al",
            "EN": "Copper Paste + Fosetyl-Al",
        },
        "action": {
            "TH": (
                "ตัดกิ่งแห้งทารอยแผลด้วยปูนแดง ราดโคนด้วย Fosetyl-Al"
            ),
            "EN": (
                "Prune dead wood, apply protective paste to wounds, and drench"
                " roots with Fosetyl-Al."
            ),
        },
    },
    "grape_leaf_blight": {
        "medicine": {"TH": "Carbendazim", "EN": "Carbendazim"},
        "action": {
            "TH": (
                "พ่น Carbendazim 20 กรัม/น้ำ 20 ลิตร"
                " งดรดน้ำขึ้นค้างตอนเย็น"
            ),
            "EN": (
                "Spray Carbendazim 20g per 20L water. Avoid late evening"
                " irrigation."
            ),
        },
    },
    "pepper_healthy": {
        "medicine": {
            "TH": "ไม่จำเป็นต้องใช้ยาเคมี",
            "EN": "No chemical treatment required",
        },
        "action": {
            "TH": "ตรวจดูเพลี้ยไฟสม่ำเสมอ",
            "EN": "Inspect regularly for thrips and manage moisture levels.",
        },
    },
    "pepper_bacterial_spot": {
        "medicine": {"TH": "Copper Hydroxide", "EN": "Copper Hydroxide"},
        "action": {
            "TH": (
                "พ่น Copper Hydroxide 30 กรัม/น้ำ 20 ลิตร"
                " ทุก 5 วันในช่วงฝนชื้น"
            ),
            "EN": (
                "Spray Copper Hydroxide 30g per 20L water every 5 days during"
                " humid periods."
            ),
        },
    },
}


# ฟังก์ชันตรวจจับและทำ Heatmap ด้วย OpenCV
def generate_disease_heatmap_and_boxes(pil_image):
  img_np = np.array(pil_image)
  img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
  hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
  h, w, _ = img_bgr.shape

  lower_green = np.array([25, 40, 40])
  upper_green = np.array([85, 255, 255])
  leaf_mask = cv2.inRange(hsv, lower_green, upper_green)

  kernel_leaf = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
  leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel_leaf)

  contours_leaf, _ = cv2.findContours(
      leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )
  main_leaf_contour_mask = np.zeros((h, w), dtype=np.uint8)

  if contours_leaf:
    max_cnt = max(contours_leaf, key=cv2.contourArea)
    if cv2.contourArea(max_cnt) > (h * w * 0.15):
      cv2.drawContours(main_leaf_contour_mask, [max_cnt], -1, 255, -1)
      main_leaf_contour_mask = cv2.dilate(
          main_leaf_contour_mask, kernel_leaf, iterations=2
      )
    else:
      main_leaf_contour_mask[:, :] = 255
  else:
    main_leaf_contour_mask[:, :] = 255

  lower_yellow = np.array([12, 80, 100])
  upper_yellow = np.array([28, 255, 255])
  mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

  lower_brown = np.array([2, 30, 40])
  upper_brown = np.array([18, 200, 200])
  mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)

  disease_raw = cv2.bitwise_or(mask_yellow, mask_brown)
  disease_mask = cv2.bitwise_and(disease_raw, main_leaf_contour_mask)

  kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
  disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_OPEN, kernel_clean)
  disease_mask = cv2.morphologyEx(disease_mask, cv2.MORPH_CLOSE, kernel_clean)

  heatmap = cv2.applyColorMap(disease_mask, cv2.COLORMAP_JET)
  heatmap_overlay = cv2.addWeighted(img_bgr, 0.7, heatmap, 0.3, 0)

  contours_disease, _ = cv2.findContours(
      disease_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
  )
  boxed_img = img_bgr.copy()

  box_count = 0
  img_area = h * w

  for cnt in contours_disease:
    area = cv2.contourArea(cnt)
    if 400 < area < (img_area * 0.40):
      x, y, bw, bh = cv2.boundingRect(cnt)
      cv2.rectangle(boxed_img, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
      box_count += 1

  return (
      cv2.cvtColor(boxed_img, cv2.COLOR_BGR2RGB),
      cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB),
      box_count,
  )


def get_healthy_image(plant_name):
  base_path = os.path.join("assets", "healthy", plant_name)
  for ext in [".JPG", ".jpg", ".png", ".jpeg"]:
    full_path = base_path + ext
    if os.path.exists(full_path):
      return Image.open(full_path).convert("RGB")
  return None


def image_to_base64(pil_img):
  buffered = io.BytesIO()
  pil_img.save(buffered, format="JPEG")
  return base64.b64encode(buffered.getvalue()).decode()


labels_file = "labels_backup.txt"
if os.path.exists(labels_file):
  with open(labels_file, "r", encoding="utf-8") as f:
    all_labels = [line.strip() for line in f if line.strip()]
else:
  all_labels = []


@st.cache_resource
def load_plant_model():
  model_path = "plant_model_backup.tflite"
  if not os.path.exists(model_path):
    return None
  interpreter = tflite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()
  return interpreter


plant_interp = load_plant_model()

# UI Sidebar
with st.sidebar:
  # ปุ่มกลับสู่หน้าหลัก
  if st.button(T[lang]["btn_back_home"], use_container_width=True):
    st.session_state.current_page = "main"
    st.rerun()

  st.markdown("---")
  st.header(T[lang]["sidebar_control"])

  selected_lang = st.radio(
      T[lang]["lang_select"],
      ["TH", "EN"],
      index=0 if lang == "TH" else 1,
      horizontal=True,
  )
  if selected_lang != st.session_state.lang:
    st.session_state.lang = selected_lang
    st.rerun()

  st.caption(T[lang]["theme_info"])
  st.markdown("---")

  selected_plant = st.selectbox(
      T[lang]["select_plant"], list(plant_classes.keys())
  )

  gps_coords = st.text_input(
      T[lang]["gps_label"],
      value="16.4322, 102.8236",
      help="เช่น 16.4322, 102.8236",
  )

  # ส่วนเช็กสภาพอากาศจาก API
  if gps_coords:
    try:
      lat, lon = [float(coord.strip()) for coord in gps_coords.split(",")]
      weather = get_weather_spray_advice(lat, lon)

      if weather["status"] == "SUCCESS":
        st.markdown(f"#### {T[lang]['weather_header']}")
        if weather["is_suitable"]:
          st.success(f"**{T[lang]['weather_ok']}**")
        else:
          st.error(f"**{T[lang]['weather_not_ok']}**")

        col_w1, col_w2 = st.columns(2)
        col_w1.metric(T[lang]["temp_label"], f"{weather['temp']} °C")
        col_w1.metric(T[lang]["wind_label"], f"{weather['wind_speed']} km/h")
        col_w2.metric(T[lang]["humidity_label"], f"{weather['humidity']} %")
        col_w2.metric(T[lang]["rain_label"], f"{weather['rain_prob']} %")

        if weather["reasons"]:
          st.caption(f"**{T[lang]['caution_label']}**")
          for reason in weather["reasons"]:
            st.caption(f"- {reason}")
    except ValueError:
      st.caption(
          "ระบุพิกัด GPS ในรูปแบบ lat, lon (เช่น 16.4322, 102.8236)"
      )

  st.markdown("---")
  run_btn = st.button(
      T[lang]["btn_analyze"], type="primary", use_container_width=True
  )

  # เครื่องคำนวณสัดส่วนยา + เช็ก Tank-Mix Compatibility
  st.markdown("---")
  st.subheader(T[lang]["calc_header"])

  selected_chem = st.selectbox(
      T[lang]["select_chem"], list(CHEMICAL_DB.keys())
  )
  mix_chem = st.selectbox(
      T[lang]["mix_chem"],
      [k for k in CHEMICAL_DB.keys() if k != "กำหนดเอง (Custom Input)"],
  )

  chem_info = CHEMICAL_DB[selected_chem]
  mix_info = CHEMICAL_DB[mix_chem]

  if mix_chem != "ไม่มีการผสมเพิ่ม (None)":
    if (
        chem_info["type"] == "copper" and mix_info["type"] == "copper"
    ) or (chem_info["type"] == "copper" and "คอปเปอร์" in mix_chem):
      st.error(
          "คำเตือนความปลอดภัย (Tank-Mix Warning):"
          " ไม่ควรผสมสารกลุ่มคอปเปอร์ร่วมกัน หรือผสมกับสารที่เป็นกรดจัด!"
      )
    else:
      st.warning(
          "ข้อควรระวัง: เมื่อผสมยา 2 ชนิด ควรละลายยาผงในน้ำก่อน"
          " แล้วจึงเติมยาแบบน้ำตามลงไป"
      )

  note_text = chem_info["note_th"] if lang == "TH" else chem_info["note_en"]
  st.caption(f"{T[lang]['chem_advice']} {note_text}")

  col_a, col_b = st.columns(2)
  with col_a:
    rai = st.number_input(
        T[lang]["area_rai"], min_value=0.0, value=1.0, step=0.5
    )
  with col_b:
    ngan = st.number_input(
        T[lang]["area_ngan"], min_value=0.0, max_value=4.0, value=0.0, step=1.0
    )

  dose_per_20l = st.number_input(
      f"{T[lang]['dose_label']} ({chem_info['unit']})",
      min_value=1.0,
      value=float(chem_info["dose"]),
      step=5.0,
  )
  water_per_rai = st.number_input(
      T[lang]["water_label"],
      min_value=10.0,
      value=float(chem_info["water_rai"]),
      step=10.0,
  )

  total_rai = rai + (ngan / 4.0)
  total_water = total_rai * water_per_rai
  total_chemical = (total_water / 20.0) * dose_per_20l

  st.success(
      f"{T[lang]['calc_result']} {total_chemical:.1f} {chem_info['unit']}"
  )
  st.info(f"{T[lang]['water_result']} {total_water:.1f} Liters")

  st.markdown("---")
  st.subheader(T[lang]["history"])
  history_placeholder = st.empty()


# =========================================================
# Main Layout: แบ่งคอลัมน์ซ้าย (รับภาพ/กล้องสด) & ขวา (ผลตรวจ)
# =========================================================
st.title(T[lang]["title"])

col_left, col_right = st.columns([1.1, 0.9])

with col_left:
  st.subheader("Uploaded & Processed Image")

  # ส่วนสลับช่องทางรับภาพ (อัปโหลด vs กล้องสด)
  input_mode = st.radio(
      T[lang]["input_mode"],
      [T[lang]["mode_upload"], T[lang]["mode_camera"]],
      horizontal=True,
  )

  uploaded_file = None
  camera_file = None

  if input_mode == T[lang]["mode_upload"]:
    uploaded_file = st.file_uploader(
        T[lang]["upload_img"], type=["jpg", "jpeg", "png"]
    )
  else:
    camera_file = st.camera_input(
        T[lang]["camera_img"], key="main_cam", help="ส่องใบพืชแล้วกดถ่ายภาพ"
    )

  active_image_file = (
      uploaded_file
      if input_mode == T[lang]["mode_upload"]
      else camera_file
  )

  # แสดงผลภาพและ Heatmap เมื่อมีรูปภาพเข้ามา
  if active_image_file is not None:
    image = Image.open(active_image_file).convert("RGB")
    boxed_img, heatmap_img, box_count = generate_disease_heatmap_and_boxes(
        image
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Bounding Box", "Heatmap", "Compare", "Original"]
    )

    with tab1:
      st.image(
          boxed_img,
          caption=f"Detected ~{box_count} disease spots",
          use_container_width=True,
      )
    with tab2:
      st.image(
          heatmap_img, caption="Heatmap intensity", use_container_width=True
      )
    with tab3:
      healthy_img = get_healthy_image(selected_plant)
      if healthy_img is not None:
        slider_val = st.slider(
            "Slider (Left: Active Image | Right: Healthy)", 0, 100, 50
        )
        w, h = image.size
        healthy_resized = healthy_img.resize((w, h))
        img_uploaded_np = np.array(image)
        img_healthy_np = np.array(healthy_resized)
        split_idx = int(w * (slider_val / 100.0))

        blended_np = np.zeros_like(img_uploaded_np)
        blended_np[:, :split_idx] = img_uploaded_np[:, :split_idx]
        blended_np[:, split_idx:] = img_healthy_np[:, split_idx:]
        if 0 < split_idx < w:
          cv2.line(
              blended_np, (split_idx, 0), (split_idx, h), (255, 255, 255), 3
          )

        st.image(
            blended_np,
            caption=f"Comparing with healthy {selected_plant}",
            use_container_width=True,
        )
      else:
        st.warning(
            "No healthy reference image found in"
            f" assets/healthy/{selected_plant}"
        )
    with tab4:
      st.image(
          image, caption="Original active image", use_container_width=True
      )
  else:
    if input_mode == T[lang]["mode_upload"]:
      st.info("กรุณาอัปโหลดรูปภาพใบพืช หรือสลับไปโหมดกล้องสด")
    else:
      st.info("กรุณากดเปิดกล้องและถ่ายภาพใบพืชผ่านช่องรับภาพด้านบน")


# ฝั่งขวา (ผลการตรวจวิเคราะห์ + Cross-Check Validation + PHI/PPE)
with col_right:
  st.subheader(T[lang]["result_header"])

  if plant_interp is None:
    st.error("Error: plant_model_backup.tflite not found.")
  elif len(all_labels) == 0:
    st.error("Error: labels_backup.txt not found.")
  elif active_image_file is None:
    st.write("Waiting for image upload or camera capture...")
  elif run_btn:
    with st.spinner("Analyzing..."):
      img_resized = image.resize((224, 224))
      img_np = np.array(img_resized, dtype=np.float32) / 255.0
      inp = np.expand_dims(img_np, axis=0)

      input_details = plant_interp.get_input_details()
      output_details = plant_interp.get_output_details()

      plant_interp.set_tensor(input_details[0]["index"], inp)
      plant_interp.invoke()
      raw_preds = plant_interp.get_tensor(output_details[0]["index"])[0]

      # ตรวจสอบพืชข้ามคลาส (Cross-Check Validation)
      best_overall_idx = np.argmax(raw_preds)
      best_overall_label = all_labels[best_overall_idx]

      detected_plant_category = None
      for cat, labels_in_cat in plant_classes.items():
        if best_overall_label in labels_in_cat:
          detected_plant_category = cat
          break

      if detected_plant_category and detected_plant_category != selected_plant:
        st.markdown(
            f"""
            <div class="card card-danger">
                <div class="card-title" style="color: #ef4444;">เตือนภัย: ชนิดพืชไม่ตรงกัน (Plant Mismatch)</div>
                <div class="card-value" style="font-size: 1.1rem; font-weight: 600; color: #f8fafc;">
                    คุณเลือกพืชเป็น <span style="color: #f59e0b;">{selected_plant}</span> แต่ภาพที่วิเคราะห์ได้คือ <span style="color: #ef4444; font-weight: bold;">{detected_plant_category.upper()}</span> ({best_overall_label.replace("_", " ").title()})
                </div>
                <div style="margin-top: 8px; font-size: 0.9rem; color: #cbd5e1;">
                    กรุณาเปลี่ยน "1. เลือกชนิดพืช" ในแผงควบคุมด้านซ้ายให้ตรงกับภาพที่ถ่าย เพื่อความถูกต้องในการแนะนำตัวยา
                </div>
            </div>
        """,
            unsafe_allow_html=True,
        )

      allowed_labels = plant_classes[selected_plant]
      scores = {}
      for label in allowed_labels:
        if label in all_labels:
          idx = all_labels.index(label)
          scores[label] = raw_preds[idx]

      if scores:
        best_disease = max(scores, key=scores.get)
        confidence = scores[best_disease] * 100.0
        display_title = best_disease.replace("_", " ").upper()

        disease_info = advice_data.get(
            best_disease.lower(),
            {
                "medicine": {
                    "TH": "สารป้องกันกำจัดเชื้อราทั่วไป",
                    "EN": "Broad-spectrum Fungicide",
                },
                "action": {
                    "TH": (
                        "ตัดแต่งส่วนที่เป็นโรคทิ้ง และพ่นสารเคมีตามอาการ"
                    ),
                    "EN": (
                        "Prune infected parts and treat with appropriate"
                        " fungicides."
                    ),
                },
            },
        )

        med_text = disease_info["medicine"].get(
            lang, disease_info["medicine"]["EN"]
        )
        action_text = disease_info["action"].get(
            lang, disease_info["action"]["EN"]
        )

        card_class = (
            "card-success"
            if "healthy" in best_disease.lower()
            else "card-warning"
        )

        # Card ผลการตรวจ
        st.markdown(
            f"""
                    <div class="card {card_class}">
                        <div class="card-title">Diagnostic Result</div>
                        <div class="card-value" style="font-size: 1.4rem; font-weight: 700;">{display_title}</div>
                        <div style="color: #a7f3d0; margin-bottom: 12px;">Confidence: {confidence:.2f}% | GPS: {gps_coords}</div>
                        <div style="margin-top: 8px;"><b>{T[lang]['med_title']}:</b> {med_text}</div>
                        <div style="margin-top: 6px; color: #cbd5e1;"><b>{T[lang]['action_title']}:</b> {action_text}</div>
                    </div>
                """,
            unsafe_allow_html=True,
        )

        # Card ข้อมูลความปลอดภัย PHI & PPE
        st.markdown(
            f"""
                    <div class="card" style="border-left: 5px solid #3b82f6;">
                        <div class="card-title" style="color: #60a5fa;">{T[lang]['safety_title']}</div>
                        <div style="margin-top: 8px; color: #f8fafc;"><b>{T[lang]['phi_label']}:</b> <span style="color: #f59e0b; font-weight: bold;">{chem_info['phi']}</span></div>
                        <div style="margin-top: 6px; color: #f8fafc;"><b>{T[lang]['ppe_label']}:</b> {chem_info['ppe']}</div>
                    </div>
                """,
            unsafe_allow_html=True,
        )

        st.markdown(f"*{T[lang]['top3_title']}*")
        sorted_scores = sorted(
            scores.items(), key=lambda x: x[1], reverse=True
        )[:3]
        df_top3 = pd.DataFrame({
            "Disease": [
                item[0].replace("_", " ").title() for item in sorted_scores
            ],
            "Confidence (%)": [
                round(item[1] * 100, 2) for item in sorted_scores
            ],
        }).set_index("Disease")

        st.bar_chart(df_top3)

        st.markdown("---")
        st.markdown(f"*{T[lang]['export_pdf']}*")

        img_b64 = image_to_base64(image)

        report_html = f"""
                <script>
                function printReport() {{
                    var printWindow = window.open('', '', 'height=850,width=850');
                    printWindow.document.write('<html><head><title>Plant Diagnostic Report</title>');
                    printWindow.document.write('<style>');
                    printWindow.document.write('body {{ font-family: Arial, sans-serif; padding: 24px; line-height: 1.6; color: #1e293b; }}');
                    printWindow.document.write('h2 {{ color: #059669; border-bottom: 2px solid #059669; padding-bottom: 8px; margin-bottom: 20px; }}');
                    printWindow.document.write('.img-container {{ text-align: center; margin: 15px 0; }}');
                    printWindow.document.write('.img-container img {{ max-width: 300px; height: auto; border-radius: 8px; border: 1px solid #cbd5e1; }}');
                    printWindow.document.write('.box {{ background: #f8fafc; padding: 12px; border-left: 4px solid #059669; margin-bottom: 10px; border-radius: 4px; }}');
                    printWindow.document.write('.safety-box {{ background: #eff6ff; padding: 12px; border-left: 4px solid #3b82f6; margin-bottom: 10px; border-radius: 4px; }}');
                    printWindow.document.write('</style></head><body>');
                    printWindow.document.write('<h2>Plant Diagnostic Report</h2>');
                    printWindow.document.write('<div class="img-container"><img src="data:image/jpeg;base64,{img_b64}" /></div>');
                    printWindow.document.write('<p><b>Plant Type:</b> {selected_plant} | <b>GPS Location:</b> {gps_coords}</p>');
                    printWindow.document.write('<p><b>Diagnostic Result:</b> <span style="color:#dc2626; font-weight:bold;">{display_title}</span> ({confidence:.2f}%)</p>');
                    printWindow.document.write('<p><b>Field Area:</b> {total_rai:.2f} Rai</p>');
                    printWindow.document.write('<p><b>Chemical Dosage:</b> {total_chemical:.1f} {chem_info["unit"]} in {total_water:.1f}L Water</p>');
                    printWindow.document.write('<div class="box"><b>{T[lang]['med_title']}:</b> {med_text}</div>');
                    printWindow.document.write('<div class="box"><b>{T[lang]['action_title']}:</b> {action_text}</div>');
                    printWindow.document.write('<div class="safety-box"><b>{T[lang]['phi_label']}:</b> {chem_info["phi"]}<br/><b>{T[lang]['ppe_label']}:</b> {chem_info["ppe"]}</div>');
                    printWindow.document.write('<hr style="margin-top:20px; border:none; border-top:1px solid #e2e8f0;"/><p style="font-size: 12px; color: #64748b;">Generated by Plant Disease Diagnostic System</p>');
                    printWindow.document.write('</body></html>');
                    printWindow.document.close();
                    printWindow.print();
                }}
                </script>
                <button onclick="printReport()" style="padding: 12px 20px; background-color: #059669; color: white; border: none; border-radius: 8px; cursor: pointer; width: 100%; font-size: 15px; font-weight: 600;">
                    {T[lang]['btn_print']}
                </button>
                """
        st.components.v1.html(report_html, height=60)

        new_entry = {
            "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
            "Plant": selected_plant,
            "Result": best_disease.replace("_", " ").title(),
            "GPS": gps_coords,
            "Confidence": f"{confidence:.2f}%",
        }
        st.session_state.detection_history.insert(0, new_entry)

# ตารางประวัติและสถิติ
if st.session_state.detection_history:
  df_hist = pd.DataFrame(st.session_state.detection_history)
  history_placeholder.dataframe(
      df_hist.head(5), use_container_width=True, hide_index=True
  )

  with st.sidebar:
    st.markdown("---")
    st.subheader(T[lang]["stat_header"])
    disease_counts = df_hist["Result"].value_counts()
    st.bar_chart(disease_counts)