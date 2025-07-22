import os
import requests
import mercantile
from datetime import datetime
from vt2geojson.tools import vt_bytes_to_geojson
from dotenv import load_dotenv
from tqdm import tqdm
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.exceptions import HTTPError, RequestException

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
# Build missing tile list
MISSING_COORDS = [
    (4685,6268),
    (4688,6269),
    (4687,6264),
    (4687,6268),
    (4685,6262)
]

# MISSING_COORDS = [
#     (4684,6269),
#     (4683,6267),
#     (4687,6271),
#     (4683,6264),
#     (4685,6262),
#     (4685,6268),
#     (4685,6266),
#     (4687,6264),
#     (4686,6266),
#     (4688,6266),
#     (4688,6269),
#     (4687,6266),
#     (4689,6270),
#     (4689,6269),
#     (4685,6265),
#     (4690,6264),
#     (4688,6265),
#     (4690,6266),
#     (4688,6267),
#     (4690,6267),
#     (4684,6266),
#     (4685,6267),
#     (4688,6270),
#     (4689,6265),
#     (4687,6267),
#     (4686,6263),
#     (4690,6269),
#     (4687,6268),
#     (4688,6268),
#     (4686,6268)
# ]

def main():
    records = []
    tiles = [mercantile.Tile(x, y, 14) for x, y in MISSING_COORDS]
    
    max_workers = min(32, (os.cpu_count() or 1) * 5)


    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_tile = {executor.submit(process_tile, t): t for t in tiles}
        for future in tqdm(as_completed(future_to_tile),
                            total=len(future_to_tile),
                        desc="Processing tiles",
                        unit="tile"):
                try:
                    records.extend(future.result())

                    # Immediately append results to CSV
                    if records:
                        with open(OUTFILE, "a", newline="") as f:
                            csv.writer(f).writerows(records)
                        records.clear()  # Clear records after writing to avoid duplicates
                        print(f"Processed and saved records for tile {future_to_tile[future].x},{future_to_tile[future].y}")
                
                except HTTPError as e:
                    code = e.response.status_code if e.response else None
                    if code and 500 <= code < 600:
                        print(f"[Skip] Tile {future_to_tile[future].x},{future_to_tile[future].y} due to server error {code}")
                        continue
                    else:
                        raise
                except RequestException as e:
                    print(f"[Skip] Tile {future_to_tile[future].x},{future_to_tile[future].y} network error: {e}")
                    continue
                except Exception as e:
                    print(f"[Error] Tile {future_to_tile[future].x},{future_to_tile[future].y}: {e}")
                    continue

if __name__ == "__main__":
    main()