import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torch
import os

# Configuration
MODEL_PATH = 'yolo/weights/best.pt'
INPUT_CSV = 'filtered_dallas_mapillary_image_data_part1.csv'
OUTPUT_CSV = 'dallas_image_urls_with_preds_part1.csv'
INTERMEDIATE_CSV = 'dallas_intermediate_preds_part1.csv'
URL_COL = 'url'

NUM_WORKERS = 20
BATCH_SIZE = 32
SAVE_EVERY = 1000

# Load YOLO classifier on GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH).to(device)

# Load and prepare CSV
df = pd.read_csv('dallas_intermediate_preds_part1.csv')
df['prediction'] = df['prediction'].fillna('')
df['confidence'] = df['confidence'].fillna('')

# Identify rows already processed
df_to_process = df[(df['prediction'] == '') | (df['prediction'] == 'ERROR')].copy()

# Image fetcher
def fetch_image(idx_url):
    idx, url = idx_url
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content)).convert("RGB")
        return (idx, img)
    except Exception:
        return (idx, None)

# Classifier batch prediction
def predict_batch(batch_data):
    indices, images = zip(*batch_data)
    results = model(list(images), verbose=False)
    output = []
    for i, r in enumerate(results):
        pred_class = int(r.probs.top1)
        pred_conf = float(r.probs.top1conf)
        label = 'yes' if pred_class == 1 else 'no'
        output.append((indices[i], label, f"{pred_conf:.3f}"))
    return output

# Save intermediate CSV
def save_intermediate():
    df.to_csv(INTERMEDIATE_CSV, index=False)
    print(f"Intermediate results saved to {INTERMEDIATE_CSV}")

# Main Loop
batch = []
processed_since_save = 0

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = [executor.submit(fetch_image, (idx, url)) for idx, url in df_to_process[URL_COL].items()]

    for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching + Classifying"):
        idx, img = future.result()
        if img is None:
            df.at[idx, 'prediction'] = 'ERROR'
            continue

        batch.append((idx, img))

        if len(batch) >= BATCH_SIZE:
            preds = predict_batch(batch)
            for idx_p, label, conf in preds:
                df.at[idx_p, 'prediction'] = label
                df.at[idx_p, 'confidence'] = conf
                processed_since_save += 1
            batch.clear()

        if processed_since_save >= SAVE_EVERY:
            save_intermediate()
            processed_since_save = 0

    # Final batch
    if batch:
        preds = predict_batch(batch)
        for idx_p, label, conf in preds:
            df.at[idx_p, 'prediction'] = label
            df.at[idx_p, 'confidence'] = conf
    save_intermediate()

# Save final result
df.to_csv(OUTPUT_CSV, index=False)
print(f"All predictions saved to {OUTPUT_CSV}")