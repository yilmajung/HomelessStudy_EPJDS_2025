import requests
import os
import csv
import mercantile
from datetime import datetime
from vt2geojson.tools import vt_bytes_to_geojson
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# CONFIG
load_dotenv()
ACCESS_TOKEN = os.getenv("MAPILLARY_ACCESS_TOKEN", "MLY|MY_TOKEN_HERE")
WEST, SOUTH, EAST, NORTH = -77.1198, 38.7916, -76.9094, 38.9955
START_DATE = datetime(2016, 1, 1)
END_DATE   = datetime(2024, 5, 31, 23, 59, 59)
start_ms = int(START_DATE.timestamp() * 1000)
end_ms   = int(END_DATE.timestamp()   * 1000)

OUTFILE = "dc_mapillary_image_data.csv"  

def fetch_tile_geojson(x, y, z):
    url = (
        f"https://tiles.mapillary.com/maps/vtp/mly1_public/2/"
        f"{z}/{x}/{y}?access_token={ACCESS_TOKEN}"
    )
    r = requests.get(url); r.raise_for_status()
    return vt_bytes_to_geojson(r.content, x, y, z, layer="image")

def fetch_image_url(image_id):
    # include your retry/backoff logic here if you like
    endpoint = f"https://graph.mapillary.com/{image_id}"
    params = {"fields": "thumb_2048_url", "access_token": ACCESS_TOKEN}
    r = requests.get(endpoint, params=params); r.raise_for_status()
    return r.json().get("thumb_2048_url")

def process_tile(tile):
    recs = []
    geojson = fetch_tile_geojson(tile.x, tile.y, tile.z)
    for feat in geojson["features"]:
        lon, lat = feat["geometry"]["coordinates"]
        if not (WEST <= lon <= EAST and SOUTH <= lat <= NORTH):
            continue
        cap_at = feat["properties"].get("captured_at", 0)
        if cap_at < start_ms or cap_at > end_ms:
            continue
        img_id = feat["properties"].get("id")
        if not img_id:
            continue
        img_url = fetch_image_url(img_id)
        if img_url:
            recs.append((img_id, cap_at, lon, lat, img_url))
    return recs

# RETRY‐ONLY WORKFLOW
def retry_missing_tiles(missing_coords, max_workers=8):
    tiles = [mercantile.Tile(x, y, 14) for x, y in missing_coords]
    all_new = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for recs in tqdm(executor.map(process_tile, tiles), total=len(tiles), desc="Retrying missing tiles", unit="tile"):
            all_new.extend(recs)
    return all_new

if __name__ == "__main__":
    
    MISSING_COORDS = [
    #    (4684,6269),
    #     (4683,6267),
        (4687,6271),
        (4683,6264),
        (4685,6262),
        (4685,6268),
        (4685,6266),
        (4687,6264),
        (4686,6266),
        (4688,6266),
        (4688,6269),
        (4687,6266),
        (4689,6270),
        (4689,6269),
        (4685,6265),
        (4690,6264),
        (4688,6265),
        (4690,6266),
        (4688,6267),
        (4690,6267),
        (4684,6266),
        (4685,6267),
        (4688,6270),
        (4689,6265),
        (4687,6267),
        (4686,6263),
        (4690,6269),
        (4687,6268),
        (4688,6268),
        (4686,6268)
    ]

    new_records = retry_missing_tiles(MISSING_COORDS, max_workers=8)
    print(f"Fetched {len(new_records)} records from {len(MISSING_COORDS)} tiles")

    with open(OUTFILE, "a", newline="") as f:
        writer = csv.writer(f)
        for row in new_records:
            writer.writerow(row)

    print("Appended missing records. All done.")
