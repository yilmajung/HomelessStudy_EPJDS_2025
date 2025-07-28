import os
import csv
import requests
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from tqdm import tqdm

# === Configuration ===
MODEL_PATH = '/Users/wooyongjung/WJ_Projects/HomelessStudy_EPJDS_2025/runs/classify/train3/weights/best.pt'
URL_LIST_FILE = 'image_urls.txt'         # one URL per line
OUTPUT_CSV = 'yolo_predictions.csv'
FAILED_LOG = 'failed_urls.txt'
SAVE_EVERY = 1000                        # Save after every 1000 images

# === Load YOLO model ===
model = YOLO(MODEL_PATH)

# === Load URLs ===
with open(URL_LIST_FILE, 'r') as f:
    urls = [line.strip() for line in f if line.strip()]

# === Output setup ===
results_buffer = []
headers_written = os.path.exists(OUTPUT_CSV)

# === Helper functions ===
def fetch_image(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert('RGB')
    except Exception:
        return None

def save_results_to_csv(results, append=True):
    mode = 'a' if append else 'w'
    with open(OUTPUT_CSV, mode, newline='') as f:
        writer = csv.writer(f)
        if not headers_written:
            writer.writerow(['url', 'class', 'confidence', 'bbox'])  # modify as needed
        for r in results:
            writer.writerow(r)

# === Main processing loop ===
with open(FAILED_LOG, 'a') as fail_log:
    for i, url in enumerate(tqdm(urls, desc="Processing URLs")):
        img = fetch_image(url)
        if img is None:
            fail_log.write(url + '\n')
            continue

        try:
            results = model(img)[0]
            for box in results.boxes:
                cls = model.names[int(box.cls)]
                conf = float(box.conf)
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                results_buffer.append([url, cls, conf, bbox])
        except Exception as e:
            fail_log.write(f"{url}\n")
            continue

        if len(results_buffer) >= SAVE_EVERY:
            save_results_to_csv(results_buffer)
            results_buffer.clear()

    # Save any remaining results
    if results_buffer:
        save_results_to_csv(results_buffer)

print("✅ Processing complete.")
