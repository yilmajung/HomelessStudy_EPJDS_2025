import requests
import os
import csv
import mercantile
from datetime import datetime
from vt2geojson.tools import vt_bytes_to_geojson
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from requests.exceptions import HTTPError, RequestException

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
    """Fetch and decode a Mapillary vector tile to GeoJSON."""
    url = (
        f"https://tiles.mapillary.com/maps/vtp/mly1_public/2/"
        f"{z}/{x}/{y}?access_token={ACCESS_TOKEN}"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return vt_bytes_to_geojson(r.content, x, y, z, layer="image")

def fetch_image_url(image_id, retries=2, backoff=1):
    """Fetch the thumb_2048_url, retrying on 5xx up to `retries` times."""
    endpoint = f"https://graph.mapillary.com/{image_id}"
    params = {"fields": "thumb_2048_url", "access_token": ACCESS_TOKEN}
    for attempt in range(1, retries+1):
        try:
            r = requests.get(endpoint, params=params, timeout=10)
            r.raise_for_status()
            return r.json().get("thumb_2048_url")
        except HTTPError as e:
            print(f"[HTTP Error] {e} for image {image_id}, attempt {attempt}")
            code = e.response.status_code
            if 500 <= code < 600 and attempt < retries:
                time.sleep(backoff * 2**(attempt-1))
                continue
            raise
        except RequestException:
            # treat network blips as terminal for this tile
            raise

def process_tile(tile):
    """
    Download all image urls in a single tile.
    Returns a list of (id, captured_at, lon, lat, url).
    May raise HTTPError on 5xx or RequestException on network errors.
    """
    recs = []
    gj = fetch_tile_geojson(tile.x, tile.y, tile.z)
    for feat in gj["features"]:
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

        url = fetch_image_url(img_id)
        recs.append((img_id, cap_at, lon, lat, url))

    return recs

# MAIN WORKFLOW

if __name__ == "__main__":
    # Prepare CSV
    header = ["id", "captured_at_ms", "lon", "lat", "url"]
    if not os.path.exists(OUTFILE):
        with open(OUTFILE, "w", newline="") as f:
            csv.writer(f).writerow(header)

    # Build tile list
    tiles = list(mercantile.tiles(WEST, SOUTH, EAST, NORTH, 14))

    # Process in parallel, flush per tile
    max_workers = min(32, (os.cpu_count() or 1) * 5)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_tile, t): t for t in tiles}
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Tiles",
                           unit="tile"):
            tile = futures[future]
            try:
                records = future.result()
            except HTTPError as e:
                code = e.response.status_code if e.response else None
                if code and 500 <= code < 600:
                    print(f"[Skip] Tile {tile.x},{tile.y} due to server error {code}")
                    continue
                else:
                    raise
            except RequestException as e:
                print(f"[Skip] Tile {tile.x},{tile.y} network error: {e}")
                continue
            except Exception as e:
                print(f"[Error] Tile {tile.x},{tile.y}: {e}")
                continue

            # Immediately append this tile’s records
            if records:
                with open(OUTFILE, "a", newline="") as f:
                    csv.writer(f).writerows(records)

    print("All done. Missing tiles were skipped; all others flushed as they finished.")


# ##################################
# def fetch_tile_geojson(x, y, z):
#     url = (
#         f"https://tiles.mapillary.com/maps/vtp/mly1_public/2/"
#         f"{z}/{x}/{y}?access_token={ACCESS_TOKEN}"
#     )
#     r = requests.get(url); r.raise_for_status()
#     return vt_bytes_to_geojson(r.content, x, y, z, layer="image")

# def fetch_image_url(image_id):
#     # include your retry/backoff logic here if you like
#     endpoint = f"https://graph.mapillary.com/{image_id}"
#     params = {"fields": "thumb_2048_url", "access_token": ACCESS_TOKEN}
#     r = requests.get(endpoint, params=params); r.raise_for_status()
#     return r.json().get("thumb_2048_url")

# def process_tile(tile):
#     recs = []
#     geojson = fetch_tile_geojson(tile.x, tile.y, tile.z)
#     for feat in geojson["features"]:
#         lon, lat = feat["geometry"]["coordinates"]
#         if not (WEST <= lon <= EAST and SOUTH <= lat <= NORTH):
#             continue
#         cap_at = feat["properties"].get("captured_at", 0)
#         if cap_at < start_ms or cap_at > end_ms:
#             continue
#         img_id = feat["properties"].get("id")
#         if not img_id:
#             continue
#         img_url = fetch_image_url(img_id)
#         if img_url:
#             recs.append((img_id, cap_at, lon, lat, img_url))
#     return recs

# # RETRY‐ONLY WORKFLOW
# def retry_missing_tiles(missing_coords, max_workers=8):
#     tiles = [mercantile.Tile(x, y, 14) for x, y in missing_coords]
#     all_new = []
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         for recs in tqdm(executor.map(process_tile, tiles), total=len(tiles), desc="Retrying missing tiles", unit="tile"):
#             all_new.extend(recs)
#     return all_new

# if __name__ == "__main__":
    
#     MISSING_COORDS = [
#     #    (4684,6269),
#     #     (4683,6267),
#         (4687,6271),
#         (4683,6264),
#         (4685,6262),
#         (4685,6268),
#         (4685,6266),
#         (4687,6264),
#         (4686,6266),
#         (4688,6266),
#         (4688,6269),
#         (4687,6266),
#         (4689,6270),
#         (4689,6269),
#         (4685,6265),
#         (4690,6264),
#         (4688,6265),
#         (4690,6266),
#         (4688,6267),
#         (4690,6267),
#         (4684,6266),
#         (4685,6267),
#         (4688,6270),
#         (4689,6265),
#         (4687,6267),
#         (4686,6263),
#         (4690,6269),
#         (4687,6268),
#         (4688,6268),
#         (4686,6268)
#     ]

#     new_records = retry_missing_tiles(MISSING_COORDS, max_workers=8)
#     print(f"Fetched {len(new_records)} records from {len(MISSING_COORDS)} tiles")

#     with open(OUTFILE, "a", newline="") as f:
#         writer = csv.writer(f)
#         for row in new_records:
#             writer.writerow(row)

#     print("Appended missing records. All done.")
