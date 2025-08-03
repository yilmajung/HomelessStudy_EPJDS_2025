import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torch
import os
import sys

# Parse chunk ID from CLI argument
chunk_id = int(sys.argv[1])

# Configuration
MODEL_PATH = 'yolo/weights/best.pt'
INPUT_CSV = f'chunk_{chunk_id}.csv'
INTERMEDIATE_CSV = f'intermediate_chunk_{chunk_id}.csv'
OUTPUT_CSV = f'output_chunk_{chunk_id}.csv'
URL_COL = 'url'

NUM_WORKERS = 10
BATCH_SIZE = 32
SAVE_EVERY = 2000

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH).to(device)

# Load data
if os.path.exists(INTERMEDIATE_CSV):
    print(f"Resuming from {INTERMEDIATE_CSV}")
    df = pd.read_csv(INTERMEDIATE_CSV)
else:
    df = pd.read_csv(INPUT_CSV)
    df['prediction'] = ''
    df['confidence'] = ''

# Identify rows that need to be processed
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

# Prediction batcher
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

# Save intermediate results
def save_intermediate():
    df.to_csv(INTERMEDIATE_CSV, index=False)
    print(f"Saved intermediate to {INTERMEDIATE_CSV}")

# Main loop
batch = []
processed_since_save = 0

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = [executor.submit(fetch_image, (idx, url)) for idx, url in df_to_process[URL_COL].items()]

    for future in tqdm(as_completed(futures), total=len(futures), desc=f"Chunk {chunk_id}"):
        idx, img = future.result()
        if img is None:
            df.at[idx, 'prediction'] = 'ERROR'
            continue

        batch.append((idx, img))

        if len(batch) >= BATCH_SIZE:
            try:
                preds = predict_batch(batch)
                for idx_p, label, conf in preds:
                    df.at[idx_p, 'prediction'] = label
                    df.at[idx_p, 'confidence'] = conf
                    processed_since_save += 1
                batch.clear()
            except Exception as e:
                print(f"⚠️ Batch prediction failed: {e}")
                batch.clear()

        if processed_since_save >= SAVE_EVERY:
            save_intermediate()
            processed_since_save = 0

# === Final batch ===
if batch:
    try:
        preds = predict_batch(batch)
        for idx_p, label, conf in preds:
            df.at[idx_p, 'prediction'] = label
            df.at[idx_p, 'confidence'] = conf
    except Exception as e:
        print(f"Final batch failed: {e}")

save_intermediate()

# Save final result
df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved final output to {OUTPUT_CSV}")
