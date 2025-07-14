import os
import requests
import mercantile
from datetime import datetime
from vt2geojson.tools import vt_bytes_to_geojson

#Mapillary access token
MAPILLARY_ACCESS_TOKEN = os.getenv("MAPILLARY_ACCESS_TOKEN")

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
        f"{z}/{x}/{y}?access_token={MAPILLARY_ACCESS_TOKEN}"
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

# MAIN WORKFLOW

def main():
    records = []
    tiles = list(mercantile.tiles(WEST, SOUTH, EAST, NORTH, 14))
    
    for tile in tiles:
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
                records.append((img_id, cap_at, img_url))

    # Write CSV: id, captured_at (ms), url
    with open(OUTFILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "captured_at_ms", "url"])
        writer.writerows(records)

    print(f"Done — saved {len(records)} records to {OUTFILE}")

if __name__ == "__main__":
    main()