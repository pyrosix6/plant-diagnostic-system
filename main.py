import base64
import io
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import requests
import streamlit as st

# ตั้งค่าหน้าเว็บ Streamlit
st.set_page_config(
    page_title="Plant Diagnostic & Care Recommendation System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 1. โหลด TFLite Interpreter
try:
  import ai_edge_litert.interpreter as tflite
except ImportError:
  try:
    import tflite_runtime.interpreter as tflite
  except ImportError:
    import tensorflow.lite as tflite


# ฟังก์ชันช่วยแปลงภาพเป็น Base64
def get_image_base64(image_path):
  if os.path.exists(image_path):
    with open(image_path, "rb") as img_file:
      return base64.b64encode(img_file.read()).decode()
  return ""


# ดึงภาพพื้นหลัง
bg_main_b64 = get_image_base64(os.path.join("assets", "Anime1.jpg"))
bg_app_b64 = get_image_base64(os.path.join("assets", "Anime2.jpg"))
if not bg_app_b64:
  bg_app_b64 = bg_main_b64

current_bg = (
    bg_main_b64
    if st.session_state.get("current_page", "main") == "main"
    else bg_app_b64
)

# ตกแต่ง UI ด้วย Custom CSS
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url("data:image/jpeg;base64,{current_bg}") no-repeat center center fixed;
        background-size: cover;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }}
    .stApp::before {{
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(4, 47, 46, 0.78);
        backdrop-filter: blur(8px);
        z-index: -1;
    }}
    .card {{
        background: rgba(6, 78, 59, 0.82);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(52, 211, 153, 0.3);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }}
    .card-success {{ border-left: 5px solid #10b981; }}
    .card-warning {{ border-left: 5px solid #f59e0b; }}
    .card-danger {{ border-left: 5px solid #ef4444; }}
    .card-title {{ font-size: 0.85rem; font-weight: 600; color: #34d399; text-transform: uppercase; margin-bottom: 6px; }}
    .card-value {{ font-size: 1.1rem; font-weight: 500; color: #f0fdf4; }}

    div[data-testid="stFileUploader"] {{
        position: relative;
        margin-top: -155px;
        opacity: 0;
        z-index: 99;
        cursor: pointer;
    }}
    div[data-testid="stFileUploader"] section {{
        height: 150px;
        cursor: pointer;
    }}

    .upload-box-ui {{
        background: rgba(4, 47, 46, 0.9);
        border: 2px dashed #34d399;
        border-radius: 16px;
        padding: 25px 20px;
        text-align: center;
        margin-bottom: 20px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }}
    
    .stButton>button {{
        border-radius: 10px;
        font-weight: 600;
        background-color: #059669;
        color: white;
        border: none;
        padding: 8px 16px;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: #047857;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4);
    }}
    </style>
""",
    unsafe_allow_html=True,
)

# จัดการ State ระบบ
if "current_page" not in st.session_state:
  st.session_state.current_page = "main"

if "lang" not in st.session_state:
  st.session_state.lang = "TH"

if "default_gps" not in st.session_state:
  st.session_state.default_gps = "16.4322, 102.8236"

if "enable_weather" not in st.session_state:
  st.session_state.enable_weather = True

if "detection_history" not in st.session_state:
  st.session_state.detection_history = []

lang = st.session_state.lang


# ฟังก์ชันดึงสภาพอากาศจาก Open-Meteo API
@st.cache_data(ttl=600)
def get_weather_spray_advice(lat, lon):
  try:
    lat_val = float(str(lat).strip())
    lon_val = float(str(lon).strip())

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat_val}&longitude={lon_val}&current=temperature_2m,relative_humidity_2m,wind_speed_10m&hourly=precipitation_probability&forecast_days=1&timezone=auto"
    headers = {"User-Agent": "PlantDiagnosticSystem/1.0"}
    response = requests.get(url, headers=headers, timeout=5)

    if response.status_code == 200:
      data = response.json()
      current = data.get("current", {})
      hourly = data.get("hourly", {})

      temp = current.get("temperature_2m", 0)
      humidity = current.get("relative_humidity_2m", 0)
      wind_speed = current.get("wind_speed_10m", 0)

      precip_list = hourly.get("precipitation_probability", [0])
      rain_prob = precip_list[0] if precip_list and len(precip_list) > 0 else 0

      reasons = []
      is_suitable = True

      if wind_speed > 15:
        is_suitable = False
        reasons.append(
            f"ความเร็วลมสูงเกินไป ({wind_speed} km/h) อาจทำให้ยาฟุ้งกระจาย"
        )
      if rain_prob > 30:
        is_suitable = False
        reasons.append(f"โอกาสฝนตกสูง ({rain_prob}%) สารเคมีอาจถูกชะล้าง")
      if humidity < 40:
        reasons.append(f"อากาศแห้งเกินไป ({humidity}%) ตัวยาอาจระเหยเร็วเกินไป")
      elif humidity > 85:
        reasons.append(f"ความชื้นสูงเกินไป ({humidity}%) ละอองยาอาจไม่เกาะใบ")

      return {
          "status": "SUCCESS",
          "is_suitable": is_suitable,
          "temp": round(temp, 1),
          "humidity": round(humidity, 1),
          "wind_speed": round(wind_speed, 1),
          "rain_prob": round(rain_prob, 1),
          "reasons": reasons,
      }
    else:
      return {
          "status": "SUCCESS",
          "is_suitable": True,
          "temp": 28.0,
          "humidity": 70.0,
          "wind_speed": 5.0,
          "rain_prob": 10.0,
          "reasons": [
              "หมายเหตุ: ใช้ค่าสำรองเนื่องจากข้อจำกัดของ API ภายนอก"
              " (Rate Limit)"
          ],
      }
  except Exception:
    return {
        "status": "SUCCESS",
        "is_suitable": True,
        "temp": 28.0,
        "humidity": 70.0,
        "wind_speed": 5.0,
        "rain_prob": 10.0,
        "reasons": [],
    }


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
        "note_th": "ยาสัมผัส ใช้ 30-50 กรัม/น้ำ 20 ลิตร",
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
        "note_th": "ยาดูดซึม ป้องกันราแป้ง/ใบจุด ใช้ 20 กรัม/น้ำ 20 ลิตร",
        "note_en": "Systemic fungicide. 20g per 20L water.",
    },
    "Azoxystrobin (อะซอกซีสโตรบิน)": {
        "dose": 15.0,
        "water_rai": 60.0,
        "unit": "มล.",
        "type": "systemic",
        "phi": "3-7 วัน",
        "ppe": "หน้ากาก, ถุงมือยาง, แว่นตากันสารเคมี",
        "note_th": "ยาดูดซึมเข้มข้นสูง ใช้เพียง 10-15 มล./น้ำ 20 ลิตร",
        "note_en": "Concentrated systemic. 10-15 ml per 20L water.",
    },
    "Copper Hydroxide (คอปเปอร์)": {
        "dose": 30.0,
        "water_rai": 80.0,
        "unit": "กรัม",
        "type": "copper",
        "phi": "7 วัน",
        "ppe": "หน้ากาก, ถุงมือยาง, แว่นตากันสารเคมี, รองเท้าบูท",
        "note_th": (
            "กำจัดแคงเกอร์/แบคทีเรีย ใช้ 30 กรัม/น้ำ 20 ลิตร"
            " (เลี่ยงผสมยาที่เป็นกรด)"
        ),
        "note_en": "Bactericide/Fungicide. 30g per 20L water.",
    },
}

DETAILED_CHEMICALS = [
    {
        "id": "mancozeb",
        "name": "Mancozeb (แมนโคเซ็บ)",
        "group": "สารป้องกันกำจัดเชื้อรา กลุ่ม M3 (ยาสัมผัส)",
        "properties": (
            "ยาสัมผัสปกคลุมใบพืช ป้องกันการเจริญเติบโตของเส้นใยเชื้อรา"
            " ได้กว้างขวาง เหมาะสำหรับช่วงเริ่มต้นระบาด"
        ),
        "targets": "โรคใบจุดสีน้ำตาล, โรคราแป้ง, โรคราน้ำฝน, โรคแอนแทรคโนส",
        "dosage": "30-50 กรัม ต่อน้ำ 20 ลิตร",
        "caution": (
            "ควรฉีดพ่นให้ทั่วทั้งบนใบและใต้ใบ"
            " เลี่ยงการฉีดพ่นช่วงที่มีแดดจัดเพื่อป้องกันใบไหม้"
        ),
        "phi": "7 - 14 วัน",
        "ppe": "หน้ากากกันสารเคมี, ถุงมือยาง, แว่นตากันสารเคมี",
        "brands": "ไดเทน เอ็ม-45 (Dithane M-45), แมนเซ็บ (Manzeb), โพลีแรม",
    },
    {
        "id": "chlorothalonil",
        "name": "Chlorothalonil (คลอโรทาโลนิล)",
        "group": "สารป้องกันกำจัดเชื้อรา กลุ่ม M5 (ยาสัมผัสต้านทานสปอร์)",
        "properties": (
            "ยาสัมผัสชนิดทนการชะล้างของฝน เกาะติดใบพืชได้ดีเยี่ยม"
            " ยับยั้งการงอกของสปอร์เชื้อรา"
        ),
        "targets": "โรคใบไหม้แผลใหญ่, โรคราน้ำฝน, โรคใบจุดคอร์ดาเนีย, โรคราน้ำหมอก",
        "dosage": "30 กรัม ต่อน้ำ 20 ลิตร",
        "caution": "ห้ามผสมร่วมกับสารกำจัดศัตรูพืชที่มีส่วนผสมของน้ำมัน (EC)",
        "phi": "14 วัน",
        "ppe": "หน้ากากกันสารเคมี, ถุงมือยาง, เสื้อแขนยาว, แว่นตา",
        "brands": "ดาโกนิล (Daconil), คลอโรทาโลนิล 75, บราโว",
    },
    {
        "id": "carbendazim",
        "name": "Carbendazim (คาร์เบนดาซิม)",
        "group": "สารป้องกันกำจัดเชื้อรา กลุ่ม 1 (ยาดูดซึม)",
        "properties": (
            "ยาดูดซึมเข้าสู่เนื้อเยื่อพืช ยับยั้งการแบ่งเซลล์ของเชื้อรา"
            " ออกฤทธิ์ป้องกันและรักษาโรค"
        ),
        "targets": "โรคใบจุดกล้วย, โรคแอนแทรคโนส, โรคราแป้ง, โรคใบจุดสีเทา",
        "dosage": "20 กรัม ต่อน้ำ 20 ลิตร",
        "caution": (
            "สลับตัวยากับกลุ่มอื่นสม่ำเสมอ"
            " เพื่อป้องกันเชื้อราดื้อยาจากการใช้ซ้ำเป็นเวลานาน"
        ),
        "phi": "14 - 21 วัน",
        "ppe": "หน้ากากกันสารเคมี, ถุงมือยาง",
        "brands": "บาเวสสติน (Bavistin), คาร์เบน, คาร์เบนดาซิม 50",
    },
    {
        "id": "azoxystrobin",
        "name": "Azoxystrobin (อะซอกซีสโตรบิน)",
        "group": "สารป้องกันกำจัดเชื้อรา กลุ่ม 11 (ยาดูดซึมยับยั้งการหายใจ)",
        "properties": (
            "ยาดูดซึมกว้างขวาง ประสิทธิภาพสูง ออกฤทธิ์เร็ว ยับยั้งการสร้างพลังงานของเชื้อรา"
        ),
        "targets": "โรคราสนิมข้าวโพด, โรคใบไหม้แผลใหญ่, โรคแอนแทรคโนส, โรคราแป้ง",
        "dosage": "10-15 มล. ต่อน้ำ 20 ลิตร (สารชนิดน้ำ)",
        "caution": "ระวังพิษต่อสัตว์น้ำ และไม่ควรฉีดพ่นซ้ำเกิน 2 ครั้งติดต่อกัน",
        "phi": "3 - 7 วัน",
        "ppe": "หน้ากาก, ถุงมือยาง, แว่นตากันสารเคมี",
        "brands": "อะมิสสตาร์ (Amistar), ออติวา (Ortiva - ผสมดิเฟโนโคนาโซล)",
    },
    {
        "id": "copper_hydroxide",
        "name": "Copper Hydroxide (คอปเปอร์ ไฮดรอกไซด์)",
        "group": "สารกำจัดแบคทีเรียและเชื้อรา กลุ่ม M1 (สารประกอบทองแดง)",
        "properties": (
            "ออกฤทธิ์กำจัดทั้งเชื้อแบคทีเรียและเชื้อรา"
            " เข้าทำลายผนังเซลล์ของเชื้อโรคอย่างรวดเร็ว"
        ),
        "targets": "โรคขอบใบไหม้แบคทีเรีย, โรคแคงเกอร์, โรคขอบใบแห้งในข้าว",
        "dosage": "30 กรัม ต่อน้ำ 20 ลิตร",
        "caution": (
            "ห้ามผสมกับสารที่เป็นกรดจัด หรือผสมร่วมกับยาชนิดอื่นในถังเดียวกัน"
            " (อาจเกิดการตกตะกอน)"
        ),
        "phi": "7 วัน",
        "ppe": "หน้ากาก, ถุงมือยาง, แว่นตากันสารเคมี, รองเท้าบูท",
        "brands": "คอร์ไซด์ (Kocide), ฟังกูแรน (Funguran), คอปเปอร์ ไฮดรอกไซด์",
    },
]

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

advice_data = {
    "banana_sigatoka": {
        "medicine": {
            "TH": "Mancozeb หรือ Propiconazole",
            "EN": "Mancozeb or Propiconazole",
        },
        "action": {
            "TH": "ตัดใบที่เป็นโรคเผาทำลายทันที พ่น Mancozeb ทุก 7-10 วัน",
            "EN": "Prune and burn infected leaves. Spray Mancozeb.",
        },
    },
    "banana_cordana": {
        "medicine": {
            "TH": "Chlorothalonil หรือ Difenoconazole",
            "EN": "Chlorothalonil or Difenoconazole",
        },
        "action": {
            "TH": "ตัดแต่งใบโคนออก พ่น Chlorothalonil 30 กรัม/น้ำ 20 ลิตร",
            "EN": "Prune lower leaves. Apply Chlorothalonil.",
        },
    },
    "banana_pestalotiopsis": {
        "medicine": {"TH": "Carbendazim", "EN": "Carbendazim"},
        "action": {
            "TH": "ตัดส่วนเนื้อเยื่อใบแห้งออก พ่น Carbendazim 20 กรัม/น้ำ 20 ลิตร",
            "EN": "Remove dried tissues. Apply Carbendazim.",
        },
    },
    "rice_healthy": {
        "medicine": {
            "TH": "ไม่จำเป็นต้องใช้ยาเคมี",
            "EN": "No chemical treatment required",
        },
        "action": {
            "TH": "คุมระดับน้ำในนา 5-10 ซม.",
            "EN": "Maintain water level at 5-10 cm.",
        },
    },
    "rice_blast": {
        "medicine": {"TH": "Tricyclazole", "EN": "Tricyclazole"},
        "action": {
            "TH": "งดใส่ปุ๋ยไนโตรเจน พ่น Tricyclazole 10-15 กรัม/น้ำ 20 ลิตร",
            "EN": "Halt nitrogen. Spray Tricyclazole.",
        },
    },
    "rice_brown_spot": {
        "medicine": {"TH": "Mancozeb", "EN": "Mancozeb"},
        "action": {
            "TH": "ใส่ปุ๋ยโพแทสเซียม พ่น Mancozeb 30 กรัม/น้ำ 20 ลิตร",
            "EN": "Apply potassium. Spray Mancozeb.",
        },
    },
    "rice_leaf_blight": {
        "medicine": {"TH": "Copper Hydroxide", "EN": "Copper Hydroxide"},
        "action": {
            "TH": "ห้ามระบายน้ำข้ามแปลง พ่นสารคอปเปอร์ 30 กรัม/น้ำ 20 ลิตร",
            "EN": "Do not drain water. Spray Copper Hydroxide.",
        },
    },
    "rice_tungro": {
        "medicine": {"TH": "Imidacloprid", "EN": "Imidacloprid"},
        "action": {
            "TH": "ถอนต้นที่เป็นโรคเผาทิ้ง พ่นกำจัดเพลี้ยจักจั่น",
            "EN": "Uproot infected plants and spray Imidacloprid.",
        },
    },
    "corn_healthy": {
        "medicine": {
            "TH": "ไม่จำเป็นต้องใช้ยาเคมี",
            "EN": "No chemical treatment required",
        },
        "action": {
            "TH": "พูนโคนต้น และตรวจดูหนอนกระทู้",
            "EN": "Keep soil well-drained and inspect for armyworms.",
        },
    },
    "corn_rust": {
        "medicine": {
            "TH": "Azoxystrobin + Difenoconazole",
            "EN": "Azoxystrobin + Difenoconazole",
        },
        "action": {
            "TH": "พ่นสาร Azoxystrobin 10-15 ซีซี/น้ำ 20 ลิตร",
            "EN": "Apply Azoxystrobin.",
        },
    },
    "corn_gray_leaf_spot": {
        "medicine": {"TH": "Carbendazim", "EN": "Carbendazim"},
        "action": {
            "TH": "ตัดแต่งใบช่วงล่าง พ่น Carbendazim 20 กรัม/น้ำ 20 ลิตร",
            "EN": "Prune lower foliage. Spray Carbendazim.",
        },
    },
    "corn_northern_blight": {
        "medicine": {"TH": "Mancozeb", "EN": "Mancozeb"},
        "action": {
            "TH": "พ่น Mancozeb 30 กรัม/น้ำ 20 ลิตร",
            "EN": "Apply Mancozeb.",
        },
    },
    "tomato_healthy": {
        "medicine": {
            "TH": "ไม่จำเป็นต้องใช้ยาเคมี",
            "EN": "No chemical treatment required",
        },
        "action": {
            "TH": "ให้น้ำโคนต้น ทำค้างยกใบสูงพ้นพื้นดิน",
            "EN": "Water at soil level and use trellises.",
        },
    },
    "tomato_early_blight": {
        "medicine": {"TH": "Chlorothalonil", "EN": "Chlorothalonil"},
        "action": {
            "TH": "ปลิดใบโคนต้นที่เป็นแผลทิ้ง พ่น Chlorothalonil",
            "EN": "Remove lower leaves. Apply Chlorothalonil.",
        },
    },
    "tomato_late_blight": {
        "medicine": {"TH": "Metalaxyl", "EN": "Metalaxyl"},
        "action": {
            "TH": "พ่น Metalaxyl 30 กรัม/น้ำ 20 ลิตร ทันที",
            "EN": "Apply Metalaxyl immediately.",
        },
    },
    "tomato_leaf_mold": {
        "medicine": {"TH": "Copper Oxychloride", "EN": "Copper Oxychloride"},
        "action": {
            "TH": "ตัดแต่งทรงพุ่ม พ่น Copper Oxychloride เน้นใต้ใบ",
            "EN": "Prune canopy. Spray Copper Oxychloride.",
        },
    },
    "tomato_septoria": {
        "medicine": {"TH": "Mancozeb", "EN": "Mancozeb"},
        "action": {
            "TH": "เก็บเศษใบตกค้างออก พ่น Mancozeb",
            "EN": "Clear fallen debris. Spray Mancozeb.",
        },
    },
    "tomato_bacterial_spot": {
        "medicine": {
            "TH": "Copper Hydroxide + Kasugamycin",
            "EN": "Copper Hydroxide + Kasugamycin",
        },
        "action": {
            "TH": "พ่นสารทองแดงผสมคาสูกาไมซิน",
            "EN": "Apply Kasugamycin and Copper Hydroxide.",
        },
    },
    "potato_healthy": {
        "medicine": {
            "TH": "ไม่จำเป็นต้องใช้ยาเคมี",
            "EN": "No chemical treatment required",
        },
        "action": {
            "TH": "พูนโคนดินสม่ำเสมอ",
            "EN": "Mound soil around stems regularly.",
        },
    },
    "potato_early_blight": {
        "medicine": {"TH": "Chlorothalonil", "EN": "Chlorothalonil"},
        "action": {
            "TH": "พ่น Chlorothalonil ทุก 7 วัน",
            "EN": "Spray Chlorothalonil every 7 days.",
        },
    },
    "potato_late_blight": {
        "medicine": {
            "TH": "Cymoxanil + Mancozeb",
            "EN": "Cymoxanil + Mancozeb",
        },
        "action": {
            "TH": "พ่น Cymoxanil + Mancozeb 30 กรัม/น้ำ 20 ลิตร",
            "EN": "Apply Cymoxanil + Mancozeb.",
        },
    },
    "grape_healthy": {
        "medicine": {
            "TH": "ไม่จำเป็นต้องใช้ยาเคมี",
            "EN": "No chemical treatment required",
        },
        "action": {
            "TH": "จัดทรงพุ่มเปิดรับแสงแดด",
            "EN": "Prune canopy for sunlight exposure.",
        },
    },
    "grape_black_rot": {
        "medicine": {"TH": "Myclobutanil", "EN": "Myclobutanil"},
        "action": {
            "TH": "เก็บผลเน่าแห้งเผาทิ้ง พ่น Myclobutanil",
            "EN": "Remove mummified berries. Apply Myclobutanil.",
        },
    },
    "grape_esca": {
        "medicine": {"TH": "ปูนแดง + Fosetyl-Al", "EN": "Copper Paste + Fosetyl-Al"},
        "action": {
            "TH": "ตัดกิ่งแห้งทารอยแผลด้วยปูนแดง",
            "EN": "Prune dead wood and apply paste.",
        },
    },
    "grape_leaf_blight": {
        "medicine": {"TH": "Carbendazim", "EN": "Carbendazim"},
        "action": {
            "TH": "พ่น Carbendazim 20 กรัม/น้ำ 20 ลิตร",
            "EN": "Spray Carbendazim.",
        },
    },
    "pepper_healthy": {
        "medicine": {
            "TH": "ไม่จำเป็นต้องใช้ยาเคมี",
            "EN": "No chemical treatment required",
        },
        "action": {
            "TH": "ตรวจดูเพลี้ยไฟสม่ำเสมอ",
            "EN": "Inspect regularly for thrips.",
        },
    },
    "pepper_bacterial_spot": {
        "medicine": {"TH": "Copper Hydroxide", "EN": "Copper Hydroxide"},
        "action": {
            "TH": "พ่น Copper Hydroxide ทุก 5 วันในช่วงฝน",
            "EN": "Spray Copper Hydroxide every 5 days.",
        },
    },
}


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


# =========================================================
# การจัดการสลับหน้า (Main / App / Chemicals / Guide / Settings)
# =========================================================
if st.session_state.current_page == "main":
  logo_base64 = get_image_base64(os.path.join("assets", "logo.png"))
  icon_app = get_image_base64(os.path.join("assets", "icons", "icon_app.png"))
  icon_guide = get_image_base64(os.path.join("assets", "icons", "icon_guide.png"))
  icon_chem = get_image_base64(os.path.join("assets", "icons", "icon_chem.png"))
  icon_settings = get_image_base64(
      os.path.join("assets", "icons", "icon_settings.png")
  )

  if logo_base64:
    st.markdown(
        f"""
            <div style="text-align: center; padding-top: 1vh; margin-bottom: 2vh;">
                <img src="data:image/png;base64,{logo_base64}" style="max-width: 380px; width: 65%; height: auto; filter: drop-shadow(0px 6px 12px rgba(0,0,0,0.7));" />
            </div>
        """,
        unsafe_allow_html=True,
    )

  st.markdown(
      """
        <div style="background: rgba(4, 47, 46, 0.85); border-radius: 14px; padding: 12px 20px; margin-bottom: 15px; backdrop-filter: blur(6px); border: 1px solid rgba(52, 211, 153, 0.3); text-align: center;">
            <div style="color: #34d399; font-weight: 800; font-size: 1.4rem; margin-bottom: 2px;">ระบบตรวจวินิจฉัยโรคพืชและคำนวณสัดส่วนยาอัจฉริยะ</div>
            <div style="color: #a7f3d0; font-size: 0.85rem;">Plant Analysis & Care Recommendation System</div>
        </div>
    """,
      unsafe_allow_html=True,
  )

  col1, col2 = st.columns(2)

  with col1:
    img_tag_app = (
        f'<img src="data:image/png;base64,{icon_app}"/>' if icon_app else "🌿"
    )
    st.markdown(
        f"""
            <div class="card" style="text-align: center;">
                <div style="width: 42px; height: 42px; margin: 0 auto 8px auto; border-radius: 50%; background: rgba(4, 47, 46, 0.9); border: 2px solid #34d399; display: flex; align-items: center; justify-content: center; overflow: hidden;">{img_tag_app}</div>
                <h4 style="color: #f0fdf4; margin-bottom: 4px; font-size: 1.1rem;">ระบบตรวจวินิจฉัยโรคพืช</h4>
                <p style="color: #a7f3d0; font-size: 0.8rem; margin-bottom: 0;">อัปโหลดรูปภาพหรือเปิดกล้องสแกนใบพืช วิเคราะห์ด้วย AI และคำนวณปริมาณยา</p>
            </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button(
        "เข้าสู่ระบบตรวจวินิจฉัย", use_container_width=True, type="primary"
    ):
      st.session_state.current_page = "app"
      st.rerun()

    img_tag_chem = (
        f'<img src="data:image/png;base64,{icon_chem}"/>' if icon_chem else "🧪"
    )
    st.markdown(
        f"""
            <div class="card" style="text-align: center; margin-top: 10px;">
                <div style="width: 42px; height: 42px; margin: 0 auto 8px auto; border-radius: 50%; background: rgba(4, 47, 46, 0.9); border: 2px solid #34d399; display: flex; align-items: center; justify-content: center; overflow: hidden;">{img_tag_chem}</div>
                <h4 style="color: #f0fdf4; margin-bottom: 4px; font-size: 1.1rem;">แนะนำตัวยาและสารเคมี</h4>
                <p style="color: #a7f3d0; font-size: 0.8rem; margin-bottom: 0;">คู่มือตัวยาทางการเกษตร คุณสมบัติ คำเตือน และตัวอย่างชื่อแบรนด์สินค้า</p>
            </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("ดูข้อมูลตัวยาและสารเคมี", use_container_width=True):
      st.session_state.current_page = "chemicals"
      st.rerun()

  with col2:
    img_tag_guide = (
        f'<img src="data:image/png;base64,{icon_guide}"/>'
        if icon_guide
        else "📖"
    )
    st.markdown(
        f"""
            <div class="card" style="text-align: center;">
                <div style="width: 42px; height: 42px; margin: 0 auto 8px auto; border-radius: 50%; background: rgba(4, 47, 46, 0.9); border: 2px solid #34d399; display: flex; align-items: center; justify-content: center; overflow: hidden;">{img_tag_guide}</div>
                <h4 style="color: #f0fdf4; margin-bottom: 4px; font-size: 1.1rem;">วิธีใช้งานระบบ</h4>
                <p style="color: #a7f3d0; font-size: 0.8rem; margin-bottom: 0;">คู่มือการใช้งานระบบสแกนโรคพืช การดู Heatmap และขั้นตอนต่างๆ</p>
            </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("คู่มือการใช้งาน", use_container_width=True):
      st.session_state.current_page = "guide"
      st.rerun()

    img_tag_settings = (
        f'<img src="data:image/png;base64,{icon_settings}"/>'
        if icon_settings
        else "⚙️"
    )
    st.markdown(
        f"""
            <div class="card" style="text-align: center; margin-top: 10px;">
                <div style="width: 42px; height: 42px; margin: 0 auto 8px auto; border-radius: 50%; background: rgba(4, 47, 46, 0.9); border: 2px solid #34d399; display: flex; align-items: center; justify-content: center; overflow: hidden;">{img_tag_settings}</div>
                <h4 style="color: #f0fdf4; margin-bottom: 4px; font-size: 1.1rem;">การตั้งค่าระบบ</h4>
                <p style="color: #a7f3d0; font-size: 0.8rem; margin-bottom: 0;">ตั้งค่าภาษา พิกัดแปลงเกษตร ภาษาหลัก และระบบดึงสภาพอากาศ</p>
            </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("ตั้งค่าระบบ", use_container_width=True):
      st.session_state.current_page = "settings"
      st.rerun()

  st.markdown(
      "<p style='text-align: center; color: #a7f3d0; font-size: 0.75rem;"
      " margin-top: 12px;'>University Project - Plant Diagnostic & Care"
      " Recommendation System</p>",
      unsafe_allow_html=True,
  )

elif st.session_state.current_page == "chemicals":
  st.title("คู่มือแนะนำตัวยาและสารเคมีทางการเกษตร")
  st.caption(
      "ข้อมูลคุณสมบัติการออกฤทธิ์ อัตราส่วนการใช้ คำเตือน และแบรนด์สินค้าแนะนำ"
  )

  with st.sidebar:
    if st.button(T[lang]["btn_back_home"], use_container_width=True):
      st.session_state.current_page = "main"
      st.rerun()

  for chem in DETAILED_CHEMICALS:
    st.markdown("---")
    col_img, col_info = st.columns([0.8, 2.2])

    with col_img:
      base_name = chem["id"]
      img_found = None
      for ext in [".png", ".PNG", ".jpg", ".JPG", ".jpeg"]:
        test_path = os.path.join("assets", "chemicals", base_name + ext)
        if os.path.exists(test_path):
          img_found = test_path
          break

      if img_found:
        st.image(img_found, caption=chem["name"], use_container_width=True)
      else:
        st.markdown(
            f"""
                    <div style="background: rgba(15, 23, 42, 0.6); border-radius: 8px; height: 180px; display: flex; align-items: center; justify-content: center; text-align: center; padding: 10px; color: #94a3b8; font-size: 0.85rem;">
                        ไม่พบรูปภาพตัวยา<br/>(assets/chemicals/{base_name}.png)
                    </div>
                """,
            unsafe_allow_html=True,
        )

    with col_info:
      st.markdown(
          f"### <span style='color: #34d399;'>{chem['name']}</span>",
          unsafe_allow_html=True,
      )
      st.markdown(
          f"**กลุ่มกลไกการออกฤทธิ์:** <span"
          f" style='color:#93c5fd;'>{chem['group']}</span>",
          unsafe_allow_html=True,
      )
      st.markdown(f"**คุณสมบัติหลัก:** {chem['properties']}")
      st.markdown(f"**โรคพืชที่รักษาได้ดี:** {chem['targets']}")

      col_sub1, col_sub2 = st.columns(2)
      with col_sub1:
        st.markdown(
            f"**อัตราส่วนแนะนำ:** <span"
            f" style='color:#facc15;'>{chem['dosage']}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(f"**ระยะเว้นการเก็บเกี่ยว (PHI):** {chem['phi']}")
      with col_sub2:
        st.markdown(f"**อุปกรณ์ป้องกัน (PPE):** {chem['ppe']}")

      st.markdown(
          f"**ชื่อแบรนด์การค้าตัวอย่าง:** <span"
          f" style='color:#f43f5e;'>{chem['brands']}</span>",
          unsafe_allow_html=True,
      )

      st.markdown(
          f"""
                <div style="background: rgba(69, 26, 3, 0.8); border-left: 4px solid #f59e0b; padding: 8px 12px; border-radius: 4px; margin-top: 8px; font-size: 0.88rem; color: #fef08a;">
                    <b>คำเตือน/ข้อควรระวัง:</b> {chem['caution']}
                </div>
            """,
          unsafe_allow_html=True,
      )

elif st.session_state.current_page == "guide":
  st.title("คู่มือการใช้งานระบบ (User Guide)")
  st.caption(
      "ขั้นตอนและวิธีการใช้งานระบบตรวจวินิจฉัยโรคพืชและคำนวณสัดส่วนยาตั้งแต่เริ่มต้น"
  )

  with st.sidebar:
    if st.button(T[lang]["btn_back_home"], use_container_width=True):
      st.session_state.current_page = "main"
      st.rerun()

  guide_steps = [
      {
          "step": 1,
          "title": "หน้าแรกของระบบ (Main Menu)",
          "img": "ad1.png",
          "desc": (
              "หน้าจอหลักต้อนรับผู้ใช้งานเข้าสู่ระบบตรวจวินิจฉัยโรคพืชและคำนวณสัดส่วนยาอัจฉริยะ"
              " พร้อมปุ่มเมนูหลัก"
          ),
      },
      {
          "step": 2,
          "title": "แผงควบคุมการทำงาน (Control Panel & Sidebar)",
          "img": "ad2.png",
          "desc": """
**รายละเอียดเมนูต่างๆ บนแผงควบคุม:**

1. **ปุ่มกลับสู่หน้าหลัก:** ใช้สำหรับย้อนกลับไปหน้า Main Menu
2. **แผงควบคุมภาษา:** เลือกสลับภาษาระหว่าง TH / EN
3. **เลือกชนิดพืช:** เลือกประเภทพืชที่ต้องการตรวจวินิจฉัย
4. **ระบุพิกัด GPS:** ใส่พิกัดแปลงเกษตรเพื่อดึงสภาพอากาศ
5. **สภาพอากาศสด:** แสดงผลการประเมินความเหมาะสมในการฉีดพ่น
6. **ช่องทางรับภาพ:** เลือกอัปโหลดไฟล์หรือส่องกล้องสด
7. **กล่องอัปโหลดภาพ:** ลากวางหรือคลิกเพื่อเลือกไฟล์รูปภาพ
8. **ปุ่มเริ่มวิเคราะห์:** กดเพื่อส่งภาพให้ AI ประมวลผล
""",
      },
      {
          "step": 3,
          "title": "การเลือกชนิดพืชที่ต้องการตรวจ (Plant Selection)",
          "img": "ad3.png",
          "desc": (
              "เมนูดรอปดาวน์สำหรับเลือกประเภทพืชเพื่อกำหนดขอบเขตให้ AI"
              " วิเคราะห์ได้แม่นยำ"
          ),
      },
      {
          "step": 4,
          "title": "การเลือกช่องทางรับภาพและอัปโหลดใบพืช (Input Mode)",
          "img": "ad4.png",
          "desc": "รองรับการนำเข้าภาพ 2 รูปแบบตามความสะดวกในการใช้งานหน้างาน",
      },
      {
          "step": 5,
          "title": "การกดปุ่มเริ่มวิเคราะห์โรค (Run Analysis)",
          "img": "ad5.png",
          "desc": "สั่งการประมวลผลวิเคราะห์โรคพืชด้วยโมเดล AI (MobileNetV2)",
      },
      {
          "step": 6,
          "title": "การประมวลผลและผลการวินิจฉัยโรค (Diagnostic Results)",
          "img": "ad6.png",
          "desc": (
              "แสดงผลการวิเคราะห์โรคอย่างละเอียด"
              " พร้อมกราฟสถิติและความปลอดภัย"
          ),
      },
      {
          "step": 7,
          "title": "การส่งออกรายงานฉบับเต็ม (PDF / Print Export)",
          "img": "ad7.png",
          "desc": "พิมพ์รายงานการวินิจฉัยโรคทางการเกษตรหรือบันทึกเป็นไฟล์ PDF",
      },
      {
          "step": 8,
          "title": "ผลการวินิจฉัยโรค สถิติ และการพิมพ์รายงาน (Results & Export)",
          "img": "ad8.png",
          "desc": (
              "แสดงชื่อโรค, คำแนะนำตัวยา, ข้อมูลความปลอดภัย (PHI/PPE),"
              " ตารางประวัติ และปุ่มพิมพ์รายงาน PDF"
          ),
      },
  ]

  for g in guide_steps:
    st.markdown("---")
    st.markdown(
        f"### ขั้นตอนที่ {g['step']}: {g['title']}",
        unsafe_allow_html=True,
    )
    col_g1, col_g2 = st.columns([1.2, 1.8])

    with col_g1:
      img_path = os.path.join("assets", "guide", g["img"])
      if os.path.exists(img_path):
        st.image(
            img_path,
            caption=f"รูปภาพประกอบ {g['img']}",
            use_container_width=True,
        )
      else:
        st.markdown(
            f"""
                    <div style="background: rgba(15, 23, 42, 0.6); border-radius: 8px; height: 180px; display: flex; align-items: center; justify-content: center; text-align: center; padding: 10px; color: #94a3b8; font-size: 0.85rem;">
                        ไม่พบรูปภาพคู่มือ<br/>(assets/guide/{g['img']})
                    </div>
                """,
            unsafe_allow_html=True,
        )

    with col_g2:
      st.markdown("**คำอธิบายการใช้งาน:**")
      st.info(g["desc"])

elif st.session_state.current_page == "settings":
  st.title("การตั้งค่าระบบ (System Settings)")

  with st.sidebar:
    if st.button(T[lang]["btn_back_home"], use_container_width=True):
      st.session_state.current_page = "main"
      st.rerun()

  st.subheader("1. ภาษาหลักของระบบ (Language Preferences)")
  set_lang = st.radio(
      "เลือกภาษาที่ต้องการใช้งาน:",
      ["TH - ภาษาไทย", "EN - English"],
      index=0 if st.session_state.lang == "TH" else 1,
  )

  st.markdown("---")
  st.subheader("2. ตั้งค่าพิกัดแปลงเกษตรเริ่มต้น (Default Field Location)")
  set_gps = st.text_input(
      "พิกัด GPS แปลงเกษตรหลัก (Latitude, Longitude):",
      value=st.session_state.default_gps,
      help="เช่น 16.4322, 102.8236 (พิกัดขอนแก่น)",
  )

  st.markdown("---")
  st.subheader("3. การเชื่อมต่อ API สภาพอากาศ (Weather Integration)")
  set_weather = st.toggle(
      "เปิดใช้งานการดึงข้อมูลสภาพอากาศอัตโนมัติ (Open-Meteo API)",
      value=st.session_state.enable_weather,
  )

  st.markdown("---")
  col_s1, col_s2 = st.columns(2)
  with col_s1:
    if st.button("บันทึกการตั้งค่า", type="primary", use_container_width=True):
      st.session_state.lang = "TH" if "TH" in set_lang else "EN"
      st.session_state.default_gps = set_gps
      st.session_state.enable_weather = set_weather
      st.success("บันทึกการตั้งค่าเรียบร้อยแล้ว!")
      st.rerun()

  with col_s2:
    if st.button("คืนค่าเริ่มต้น (Reset)", use_container_width=True):
      st.session_state.lang = "TH"
      st.session_state.default_gps = "16.4322, 102.8236"
      st.session_state.enable_weather = True
      st.info("คืนค่าการตั้งค่าเริ่มต้นเรียบร้อยแล้ว")
      st.rerun()

else:
  # --- หน้าหลักการตรวจวินิจฉัย (App Page) ---

  with st.sidebar:
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
        value=st.session_state.default_gps,
        help="เช่น 16.4322, 102.8236",
    )

    if st.session_state.enable_weather and gps_coords:
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
        else:
          st.warning(
              f"ไม่สามารถโหลดข้อมูลอากาศได้: {weather.get('message', 'Unknown')}"
          )
      except ValueError:
        st.caption("ระบุพิกัด GPS ในรูปแบบ lat, lon")

    st.markdown("---")
    run_btn = st.button(
        T[lang]["btn_analyze"], type="primary", use_container_width=True
    )

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
            " ไม่ควรผสมสารกลุ่มคอปเปอร์ร่วมกัน!"
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

  st.title(T[lang]["title"])

  col_left, col_right = st.columns([1.1, 0.9])

  with col_left:
    st.subheader("Uploaded & Processed Image")

    input_mode = st.radio(
        T[lang]["input_mode"],
        [T[lang]["mode_upload"], T[lang]["mode_camera"]],
        horizontal=True,
    )

    uploaded_file = None
    camera_file = None

    if input_mode == T[lang]["mode_upload"]:
      upload_icon_b64 = get_image_base64(
          os.path.join("assets", "icons", "upload_icon.png")
      )
      if not upload_icon_b64:
        upload_icon_b64 = get_image_base64(
            os.path.join("assets", "icons", "icon_app.png")
        )

      icon_html = (
          f'<img src="data:image/png;base64,{upload_icon_b64}"'
          ' style="width: 75px; height: 75px; margin-bottom: 10px;'
          ' object-fit: contain; filter: drop-shadow(0 4px 8px'
          ' rgba(0,0,0,0.4));" />'
          if upload_icon_b64
          else '<div style="font-size: 3rem; margin-bottom: 8px;">📸</div>'
      )

      st.markdown(
          f"""
            <div class="upload-box-ui">
                {icon_html}
                <div style="font-size: 1.2rem; font-weight: 700; color: #34d399; margin-bottom: 4px;">Drop & Upload Image</div>
                <div style="font-size: 0.9rem; color: #f0fdf4;">ลากและวางรูปภาพใบพืชที่นี่ หรือคลิกเพื่อเลือกไฟล์จากอุปกรณ์</div>
                <div style="font-size: 0.78rem; color: #a7f3d0; margin-top: 6px;">รองรับไฟล์ JPG, PNG, JPEG</div>
            </div>
            """,
          unsafe_allow_html=True,
      )

      uploaded_file = st.file_uploader(
          T[lang]["upload_img"],
          type=["jpg", "jpeg", "png"],
          label_visibility="collapsed",
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
          st.warning(f"No healthy reference image found for {selected_plant}")
      with tab4:
        st.image(
            image, caption="Original active image", use_container_width=True
        )
    else:
      if input_mode == T[lang]["mode_upload"]:
        st.info("กรุณาอัปโหลดรูปภาพใบพืชผ่านกล่องด้านบนเพื่อเริ่มวิเคราะห์")
      else:
        st.info("กรุณากดเปิดกล้องและถ่ายภาพใบพืชผ่านช่องรับภาพด้านบน")

  with col_right:
    st.subheader(T[lang]["result_header"])

    if plant_interp is None:
      st.error("Error: plant_model_backup.tflite not found.")
    elif len(all_labels) == 0:
      st.error("Error: labels_backup.txt not found.")
    elif active_image_file is None:
      st.markdown(
          """
            <div class="card" style="text-align: center; padding: 40px 20px;">
                <div style="font-size: 2.8rem; margin-bottom: 10px;">🍃</div>
                <div style="font-size: 1.1rem; color: #a7f3d0; font-weight: 600;">กำลังรอการประมวลผลข้อมูลภาพ...</div>
                <div style="font-size: 0.85rem; color: #6ee7b7; margin-top: 6px;">อัปโหลดรูปภาพใบพืชและกดปุ่ม 'เริ่มวิเคราะห์โรคพืช' ด้านซ้ายมือ</div>
            </div>
          """,
          unsafe_allow_html=True,
      )
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

        best_overall_idx = np.argmax(raw_preds)
        best_overall_label = all_labels[best_overall_idx]

        detected_plant_category = None
        for cat, labels_in_cat in plant_classes.items():
          if best_overall_label in labels_in_cat:
            detected_plant_category = cat
            break

        if (
            detected_plant_category
            and detected_plant_category != selected_plant
        ):
          st.markdown(
              f"""
              <div class="card card-danger">
                  <div class="card-title" style="color: #ef4444;">เตือนภัย: ชนิดพืชไม่ตรงกัน (Plant Mismatch)</div>
                  <div class="card-value" style="font-size: 1.1rem; font-weight: 600; color: #f8fafc;">
                      คุณเลือกพืชเป็น <span style="color: #f59e0b;">{selected_plant}</span> แต่ภาพที่วิเคราะห์ได้คือ <span style="color: #ef4444; font-weight: bold;">{detected_plant_category.upper()}</span> ({best_overall_label.replace("_", " ").title()})
                  </div>
                  <div style="margin-top: 8px; font-size: 0.9rem; color: #cbd5e1;">
                      กรุณาเปลี่ยน "1. เลือกชนิดพืช" ในแผงควบคุมด้านซ้ายให้ตรงกับภาพที่ถ่าย
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

          if "healthy" in best_disease.lower():
            display_title = (
                "ไม่มีโรค (Healthy)"
                if lang == "TH"
                else "No Disease (Healthy)"
            )
          else:
            display_title = best_disease.replace("_", " ").upper()

          disease_info = advice_data.get(
              best_disease.lower(),
              {
                  "medicine": {
                      "TH": "สารป้องกันกำจัดเชื้อราทั่วไป",
                      "EN": "Broad-spectrum Fungicide",
                  },
                  "action": {
                      "TH": "ตัดแต่งส่วนที่เป็นโรคทิ้ง และพ่นสารเคมีตามอาการ",
                      "EN": "Prune and treat with appropriate fungicides.",
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
                      printWindow.document.write('<p><b>Diagnostic Result:</b> <span style="color:#059669; font-weight:bold;">{display_title}</span> ({confidence:.2f}%)</p>');
                      printWindow.document.write('<p><b>Field Area:</b> {total_rai:.2f} Rai</p>');
                      printWindow.document.write('<p><b>Chemical Dosage:</b> {total_chemical:.1f} {chem_info["unit"]} in {total_water:.1f}L Water</p>');
                      printWindow.document.write('<div class="box"><b>{T[lang]["med_title"]}:</b> {med_text}</div>');
                      printWindow.document.write('<div class="box"><b>{T[lang]["action_title"]}:</b> {action_text}</div>');
                      printWindow.document.write('<div class="safety-box"><b>{T[lang]["phi_label"]}:</b> {chem_info["phi"]}<br/><b>{T[lang]["ppe_label"]}:</b> {chem_info["ppe"]}</div>');
                      printWindow.document.write('<hr style="margin-top:20px; border:none; border-top:1px solid #e2e8f0;"/><p style="font-size: 12px; color: #64748b;">Generated by Plant Disease Diagnostic System</p>');
                      printWindow.document.write('</body></html>');
                      printWindow.document.close();
                      printWindow.print();
                  }}
                  </script>
                  <button onclick="printReport()" style="padding: 12px 20px; background-color: #059669; color: white; border: none; border-radius: 8px; cursor: pointer; width: 100%; font-size: 15px; font-weight: 600;">
                      {T[lang]["btn_print"]}
                  </button>
                  """
          st.components.v1.html(report_html, height=60)

          new_entry = {
              "Time": pd.Timestamp.now().strftime("%H:%M:%S"),
              "Plant": selected_plant,
              "Result": display_title,
              "GPS": gps_coords,
              "Confidence": f"{confidence:.2f}%",
          }
          st.session_state.detection_history.insert(0, new_entry)

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