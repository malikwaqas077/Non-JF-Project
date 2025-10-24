#!/usr/bin/env python3
"""
Generate CLEAN gas-poor images - manually crop out ALL axes and colorbars
"""
import csv, pathlib, requests, time
from PIL import Image, ImageChops
import io, numpy as np

print("="*70)
print("CLEAN GAS-POOR IMAGE GENERATOR")
print("Manually removes: X-axis, Y-axis, colorbar, all labels")
print("="*70)

API_KEY = "faa0959886d51fa3258568782eca5f78"
CSV_FILE = pathlib.Path("gas_poor_100_high_gas.csv")
OUT_DIR = pathlib.Path("gas_poor_clean_images")
OUT_DIR.mkdir(exist_ok=True)

MAX_IMAGES = 100  # Generate 100 high gas mass images

# Crop percentages to remove axes, colorbar, and all labels
LEFT_CROP = 0.10   # Remove left y-axis (10%)
RIGHT_CROP = 0.86  # Remove right colorbar (keep 86%, remove 14%)
TOP_CROP = 0.08    # Remove top labels/title (8%)
BOTTOM_CROP = 0.90 # Remove bottom x-axis (keep 90%)

print(f"\nSettings:")
print(f"  CSV: {CSV_FILE}")
print(f"  Output: {OUT_DIR}")
print(f"  Max images: {MAX_IMAGES}")
print(f"  Cropping: Left={LEFT_CROP*100:.0f}%, Right={(1-RIGHT_CROP)*100:.0f}%, Top={TOP_CROP*100:.0f}%, Bottom={(1-BOTTOM_CROP)*100:.0f}%")
print(f"  Final size: 384x384\n")

def make_url(subhalo_id):
    """Build visualization URL."""
    base = f"https://www.tng-project.org/api/TNG50-1/snapshots/99/subhalos/{subhalo_id}/"
    params = {
        "api_key": API_KEY,
        "partType": "gas",
        "partField": "dens",
        "method": "sphMap_subhalo",
        "size": "0.25",
        "sizeType": "rViral",
        "depthFac": "1",
        "nPixels": "1024,1024",
        "color": "jet"
    }
    
    import urllib.parse
    return f"{base}vis.png?{urllib.parse.urlencode(params)}"

def crop_clean_image(raw_bytes):
    """Crop out axes, labels, and colorbar, then resize to 384x384."""
    img = Image.open(io.BytesIO(raw_bytes))
    w, h = img.size
    
    # Calculate crop box (left, upper, right, lower)
    left = int(w * LEFT_CROP)
    top = int(h * TOP_CROP)
    right = int(w * RIGHT_CROP)
    bottom = int(h * BOTTOM_CROP)
    
    # Crop to remove axes and colorbar
    img_cropped = img.crop((left, top, right, bottom))
    
    # Trim any remaining uniform borders
    bg = Image.new(img_cropped.mode, img_cropped.size, img_cropped.getpixel((0, 0)))
    diff = ImageChops.difference(img_cropped, bg)
    bbox = diff.getbbox()
    if bbox:
        img_cropped = img_cropped.crop(bbox)
    
    # Resize to 384x384
    img_final = img_cropped.resize((384, 384), Image.BICUBIC)
    
    return img_final

saved = 0
visible = 0
failed = 0

with CSV_FILE.open(newline="") as f:
    for row in csv.reader(f):
        if saved >= MAX_IMAGES:
            break
        
        if len(row) < 2:
            continue
        
        filename, base_url = row[0], row[1]
        subhalo_id = int(base_url.split("/subhalos/")[1].rstrip("/").split("/")[0])
        
        out_path = OUT_DIR / filename
        
        print(f"[{saved+1}/{MAX_IMAGES}] {filename} (subhalo {subhalo_id})")
        print(f"  Downloading...", end=" ", flush=True)
        
        try:
            url = make_url(subhalo_id)
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            print(f"{len(r.content)/1024:.1f} KB -", end=" ", flush=True)
            
            # Crop and save
            img_clean = crop_clean_image(r.content)
            img_clean.save(out_path)
            print(f"Cropped & saved", end=" ", flush=True)
            
            # Verify visibility
            img_array = np.array(img_clean)
            mean_val = np.mean(img_array)
            std_val = np.std(img_array)
            
            saved += 1
            
            if mean_val > 20 and std_val > 15:
                visible += 1
                print(f"- [GAS VISIBLE] br={mean_val:.0f}, std={std_val:.0f}")
            else:
                print(f"- [WEAK] br={mean_val:.0f}, std={std_val:.0f}")
            
            time.sleep(0.3)
            
        except Exception as e:
            print(f"[FAIL] {e}")
            failed += 1

print(f"\n" + "="*70)
print(f"RESULTS")
print(f"="*70)
print(f"Successfully saved: {saved}/{MAX_IMAGES}")
print(f"With VISIBLE gas: {visible}/{saved} ({visible/saved*100:.0f}%)" if saved > 0 else "With VISIBLE gas: 0")
print(f"Failed: {failed}")
print(f"\nImages saved to: {OUT_DIR.resolve()}")
print(f"\n[CHECK] Open images - should have NO axes, NO labels, NO colorbar!")
print(f"="*70)

