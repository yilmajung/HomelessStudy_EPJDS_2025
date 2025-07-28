# import os
# import csv
# import requests
# from PIL import Image
# from io import BytesIO
# from ultralytics import YOLO
# from tqdm import tqdm

# # === Configuration ===
# MODEL_PATH = '/HomelessStudy_EPJDS_2025/yolo/weights/best.pt'
# INPUT_CSV = 'dallas_mapillary_image_data.csv'
# OUTPUT_CSV = 'yolo_predictions.csv'
# FAILED_LOG = 'failed_urls.txt'
# SAVE_EVERY = 1000                        # Save after every 1000 images

import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from tqdm import tqdm

# Configuration
INPUT_CSV = 'dallas_mapillary_image_data.csv'
OUTPUT_CSV = 'dallas_image_urls_with_preds.csv'
MODEL_PATH = 'yolo/weights/best.pt'
URL_COL = 'url'

# Load YOLO model
model = YOLO(MODEL_PATH)

# Load CSV
df = pd.read_csv(INPUT_CSV)
df['yolo_class'] = ''
df['yolo_confidence'] = ''
df['yolo_bbox'] = ''

# Helper function
def fetch_image(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('RGB')
    except Exception:
        return None

# Main processing loop
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
    url = row[URL_COL]
    img = fetch_image(url)
    
    if img is None:
        df.at[idx, 'yolo_class'] = 'ERROR'
        continue

    try:
        results = model(img)[0]
        classes, confs, boxes = [], [], []

        for box in results.boxes:
            cls = model.names[int(box.cls)]
            conf = float(box.conf)
            bbox = [round(coord.item(), 2) for coord in box.xyxy[0]]

            classes.append(cls)
            confs.append(f"{conf:.3f}")
            boxes.append(str(bbox))

        df.at[idx, 'yolo_class'] = ';'.join(classes)
        df.at[idx, 'yolo_confidence'] = ';'.join(confs)
        df.at[idx, 'yolo_bbox'] = ';'.join(boxes)

    except Exception as e:
        df.at[idx, 'yolo_class'] = 'ERROR'

# Save updated DataFrame
df.to_csv(OUTPUT_CSV, index=False)
print("Done. Saved predictions to:", OUTPUT_CSV)
