import os
import sys
import time
import signal
import threading
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import torch

# Parse chunk ID from CLI
chunk_id = int(sys.argv[1])  # e.g., python run_chunk_safe.py 3

# Config
MODEL_PATH = 'yolo/weights/best.pt'
INPUT_CSV = f'data_chunks/chunk_{chunk_id}.csv'
INTERMEDIATE_CSV = f'output/intermediate_chunk_{chunk_id}.csv'
OUTPUT_CSV = f'output/output_chunk_{chunk_id}.csv'
URL_COL = 'url'

NUM_WORKERS = int(os.getenv('NUM_WORKERS', 10))
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 32))
SAVE_EVERY = int(os.getenv('SAVE_EVERY', 10000))  # processed rows since last save

REQUEST_TIMEOUT = 12
MAX_RETRIES = 3
BACKOFF_BASE = 1.5  # exponential backoff factor

os.makedirs(os.path.dirname(INTERMEDIATE_CSV), exist_ok=True)

# Atomic save helpers
def atomic_save_csv(df: pd.DataFrame, path: str):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)  # atomic on POSIX

def rotate_and_save(df: pd.DataFrame, path: str):
    # Timestamped backup + latest pointer
    ts = time.strftime("%Y%m%d_%H%M%S")
    bkp = f"{path}.{ts}"
    atomic_save_csv(df, bkp)
    atomic_save_csv(df, path)

# Graceful shutdown flag
shutdown_flag = threading.Event()

def _handle_signal(signum, frame):
    print(f"\n Received signal {signum}. Saving intermediate and exiting…")
    shutdown_flag.set()
signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(MODEL_PATH).to(device)

# Load/Resume data
if os.path.exists(INTERMEDIATE_CSV):
    print(f"Resuming from {INTERMEDIATE_CSV}")
    df = pd.read_csv(INTERMEDIATE_CSV)
else:
    df = pd.read_csv(INPUT_CSV)
    if 'prediction' not in df.columns: df['prediction'] = ''
    if 'confidence' not in df.columns: df['confidence'] = ''

total_rows = len(df)
already_done = ((df['prediction'] != '') & (df['prediction'] != 'ERROR') & (~df['prediction'].isna())).sum()
df_to_process = df[(df['prediction'].isna()) | (df['prediction'] == '') | (df['prediction'] == 'ERROR')].copy()
todo_rows = len(df_to_process)

print(f"Chunk {chunk_id}: total={total_rows:,} | done={already_done:,} | to_process={todo_rows:,}")
assert URL_COL in df.columns, f"Missing column '{URL_COL}' in {INPUT_CSV}"

# Requests Session (connection pooling)
_session_local = threading.local()
def get_session():
    if not hasattr(_session_local, "session"):
        s = requests.Session()
        s.headers.update({"User-Agent": "yolo-batch-classifier/1.0"})
        _session_local.session = s
    return _session_local.session

# Image fetcher with retries/backoff
def fetch_image(idx_url):
    idx, url = idx_url
    sess = get_session()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = sess.get(url, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            return (idx, img)
        except Exception:
            if attempt == MAX_RETRIES:
                return (idx, None)
            time.sleep(BACKOFF_BASE ** attempt)

# Batch prediction
def predict_batch(batch_data):
    indices, images = zip(*batch_data)
    # verbose=False suppresses per-image prints
    results = model(list(images), verbose=False)
    out = []
    for i, r in enumerate(results):
        pred_class = int(r.probs.top1)
        pred_conf = float(r.probs.top1conf)
        label = 'yes' if pred_class == 1 else 'no'
        out.append((indices[i], label, f"{pred_conf:.3f}"))
    return out

# Save helpers
processed_since_save = 0
save_lock = threading.Lock()

def save_intermediate(tag=""):
    with save_lock:
        rotate_and_save(df, INTERMEDIATE_CSV)
        filled = ((df['prediction'] != '') & (~df['prediction'].isna())).sum()
        print(f"Saved intermediate{(' '+tag) if tag else ''}: "
              f"{INTERMEDIATE_CSV} | rows={len(df):,} | filled={filled:,}")

# Main
batch = []

try:
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = [executor.submit(fetch_image, (idx, url))
                   for idx, url in df_to_process[URL_COL].items()]

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Chunk {chunk_id}"):
            if shutdown_flag.is_set():
                save_intermediate("(shutdown)")
                break

            idx, img = future.result()
            if img is None:
                # Keep row so final length matches input; mark ERROR for resume
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
                    print(f"Batch prediction failed: {e}")
                    batch.clear()

                if processed_since_save >= SAVE_EVERY:
                    save_intermediate()
                    processed_since_save = 0

    # Final batch (if we didn’t exit early)
    if not shutdown_flag.is_set() and batch:
        try:
            preds = predict_batch(batch)
            for idx_p, label, conf in preds:
                df.at[idx_p, 'prediction'] = label
                df.at[idx_p, 'confidence'] = conf
        except Exception as e:
            print(f"Final batch failed: {e}")

    save_intermediate("(final)")

    # Final output save (atomic) + integrity check
    rotate_and_save(df, OUTPUT_CSV)
    print(f"Saved final output to {OUTPUT_CSV} | rows={len(df):,}")

    # Optional on-disk verification (catch truncation)
    _ver = pd.read_csv(OUTPUT_CSV, nrows=5)  # light read just to ensure readable
    if len(df) != total_rows:
        print(f"WARNING: in-memory rows {len(df)} != input rows {total_rows}")
    assert len(df) == total_rows, f"Row count changed in memory: {len(df)} vs {total_rows}"

except Exception as e:
    # Save something if we crash unexpectedly
    print(f"Exception: {e}. Saving emergency intermediate…")
    try:
        save_intermediate("(exception)")
    except Exception as ee:
        print(f"Failed to save intermediate: {ee}")
    raise
