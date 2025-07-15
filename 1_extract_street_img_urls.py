import os
import requests
import mercantile
from datetime import datetime
from vt2geojson.tools import vt_bytes_to_geojson
from dotenv import load_dotenv
from tqdm import tqdm
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()
# Mapillary API access token
ACCESS_TOKEN = os.getenv("MAPILLARY_ACCESS_TOKEN", "MLY|MY_TOKEN")

# City bounding box coordinates (west, south, east, north)
# Washington D.C. bounding box 
WEST, SOUTH, EAST, NORTH = -77.1198, 38.7916, -76.9094, 38.9955

# San Francisco bounding box
# WEST, SOUTH, EAST, NORTH = -122.5149, 37.7080, -122.3569, 37.8324

# Date range (inclusive)
START_DATE = datetime(2016, 1, 1)
END_DATE   = datetime(2024, 5, 31, 23, 59, 59)

# Convert to milliseconds since epoch (Mapillary’s “captured_at” is ms)
start_ms = int(START_DATE.timestamp() * 1000)
end_ms   = int(END_DATE.timestamp()   * 1000)

# Output CSV
OUTFILE = "dc_mapillary_image_data.csv"

# Create a function to fetch the Mapillary vector tile and decode it to GeoJSON
def fetch_tile_geojson(x, y, z):
    url = (
        f"https://tiles.mapillary.com/maps/vtp/mly1_public/2/"
        f"{z}/{x}/{y}?access_token={ACCESS_TOKEN}"
    )
    r = requests.get(url)
    r.raise_for_status()
    return vt_bytes_to_geojson(r.content, x, y, z, layer="image")

def fetch_image_url(image_id):
    endpoint = f"https://graph.mapillary.com/{image_id}"
    params = {
        "fields": "thumb_2048_url",
        "access_token": ACCESS_TOKEN
    }
    r = requests.get(endpoint, params=params)
    r.raise_for_status()
    return r.json().get("thumb_2048_url")

def process_tile(tile):
    recs = []
    try:
        geojson = fetch_tile_geojson(tile.x, tile.y, tile.z)
        for feat in geojson["features"]:
            lon, lat = feat["geometry"]["coordinates"]
            if not (WEST <= lon <= EAST and SOUTH <= lat <= NORTH):
                continue
            props = feat["properties"]
            cap_at = props.get("captured_at", 0)
            if cap_at < start_ms or cap_at > end_ms:
                continue
            img_id = props.get("id")
            if not img_id:
                continue
            img_url = fetch_image_url(img_id)
            if img_url:
                recs.append((img_id, cap_at, lon, lat, img_url))
    except Exception as e:
        print(f"[Error] Tile {tile.x},{tile.y}: {e}")
    return recs


# MAIN WORKFLOW

def main():
    records = []
    tiles = list(mercantile.tiles(WEST, SOUTH, EAST, NORTH, 14))
    
    max_workers = min(32, (os.cpu_count() or 1) * 5)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_tile = {executor.submit(process_tile, t): t for t in tiles}
        for future in tqdm(as_completed(future_to_tile),
                           total=len(future_to_tile),
                           desc="Processing tiles",
                           unit="tile"):
            records.extend(future.result())

    # Write CSV: id, captured_at_ms, lon, lat, url
    with open(OUTFILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "captured_at_ms", "lon", "lat", "url"])
        writer.writerows(records)

    print(f"Done — saved {len(records)} records to {OUTFILE}")

if __name__ == "__main__":
    main()