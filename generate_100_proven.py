#!/usr/bin/env python3
"""
Generate 100 images using proven high gas mass candidates
"""
import csv, pathlib, requests, time
from PIL import Image, ImageOps
import io, numpy as np

print("="*70)
print("100 IMAGES FROM PROVEN CANDIDATES")
print("Using only subhalos with confirmed high gas mass")
print("="*70)

API_KEY = "faa0959886d51fa3258568782eca5f78"
CSV_FILE = pathlib.Path("gas_poor_proven_candidates.csv")
OUT_DIR = pathlib.Path("gas_poor_clean_images")
OUT_DIR.mkdir(exist_ok=True)

# Crop percentages to remove axes, colorbar, and all labels
LEFT_CROP = 0.10   # Remove left y-axis (10%)
RIGHT_CROP = 0.86  # Remove right colorbar (keep 86%, remove 14%)
TOP_CROP = 0.08    # Remove top labels/title (8%)
BOTTOM_CROP = 0.90 # Remove bottom x-axis (keep 90%, remove 10%)
TARGET_SIZE = (384, 384)

def make_vis_url(base: str, size_factor: float = 0.25) -> str:
    """Return a label-free vis.png URL for gas density - NO AXES."""
    if not base.endswith("/"):
        base += "/"
    params = dict(
        api_key=API_KEY,
        partType="gas", partField="dens", method="sphMap_subhalo",
        size=str(size_factor), sizeType="rViral", depthFac="1",
        nPixels="800,800", rasterPx="1100", color="jet",
        plotStyle="bare",  # Remove axes
        labelScale="False", labelSim="False", labelHalo="False",
        labelZ="False", colorbars="False", title="False",
        relCoords="False", axesUnits="none",
    )
    import urllib.parse
    return f"{base}vis.png?{urllib.parse.urlencode(params, quote_via=urllib.parse.quote)}"

def tidy_image(raw_bytes: bytes) -> Image.Image:
    """Manually crop out axes, labels, and colorbar, then resize."""
    img = Image.open(io.BytesIO(raw_bytes))
    w, h = img.size
    
    # Calculate crop box
    left = int(w * LEFT_CROP)
    top = int(h * TOP_CROP)
    right = int(w * RIGHT_CROP)
    bottom = int(h * BOTTOM_CROP)
    
    # Ensure crop box is valid
    if right <= left or bottom <= top:
        print(f"  [WARNING] Invalid crop dimensions: left={left}, top={top}, right={right}, bottom={bottom}")
        # Fallback to just resizing if cropping is invalid
        return ImageOps.fit(img, TARGET_SIZE, Image.BICUBIC, centering=(0.5, 0.5))
    
    img = img.crop((left, top, right, bottom))
    
    # Fit to target size
    return ImageOps.fit(img, TARGET_SIZE, Image.BICUBIC, centering=(0.5, 0.5))

def download_with_retry(url: str, max_retries: int = 3) -> bytes:
    """Download with retry logic for network issues."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            return r.content
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise e

def is_gas_visible(img: Image.Image) -> bool:
    """Check if gas is visible in the image."""
    img_array = np.array(img.convert('L'))
    brightness = np.mean(img_array)
    std_dev = np.std(img_array)
    
    # Gas is visible if there's sufficient variation and brightness
    return std_dev > 50 and brightness > 50

print(f"\nSettings:")
print(f"  CSV: {CSV_FILE}")
print(f"  Output: {OUT_DIR}")
print(f"  Target: 100 images from proven candidates")

# Read CSV
if not CSV_FILE.exists():
    print(f"\n[ERROR] CSV file not found: {CSV_FILE}")
    exit(1)

entries = []
with CSV_FILE.open("r") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) >= 2:
            entries.append((row[0], row[1]))

print(f"\nFound {len(entries)} proven candidates")

# Check existing images
existing_images = set()
for img_file in OUT_DIR.glob("*.png"):
    existing_images.add(img_file.name)

print(f"Found {len(existing_images)} existing images")

# Generate 100 images by cycling through proven candidates with different parameters
success_count = 0
failed_count = 0
visible_gas_count = 0
target_images = 100

# Different size factors to create variation
size_factors = [0.20, 0.25, 0.30, 0.35, 0.40]

for i in range(target_images):
    # Cycle through candidates
    candidate_idx = i % len(entries)
    filename_base, base_url = entries[candidate_idx]
    
    # Create unique filename with variation
    size_factor = size_factors[i % len(size_factors)]
    filename = f"gas_poor_{i+1:03d}_{filename_base.split('_')[2].split('.')[0]}_s{size_factor:.2f}.png"
    
    if filename in existing_images:
        print(f"[{i+1}/{target_images}] {filename} - SKIPPED (already exists)")
        success_count += 1
        continue
    
    print(f"[{i+1}/{target_images}] {filename}")
    
    try:
        url = make_vis_url(base_url, size_factor)
        print(f"  Downloading...", end="", flush=True)
        
        raw_bytes = download_with_retry(url)
        print(f" {len(raw_bytes)/1024:.1f} KB", end="", flush=True)
        
        # Process image
        img = tidy_image(raw_bytes)
        
        # Check gas visibility
        if is_gas_visible(img):
            print(f" - [GAS VISIBLE]", end="", flush=True)
            visible_gas_count += 1
        else:
            print(f" - [LOW GAS]", end="", flush=True)
        
        # Save image
        img.save(OUT_DIR / filename)
        print(f" - Saved")
        
        success_count += 1
        
    except Exception as e:
        print(f" - FAILED: {e}")
        failed_count += 1
    
    # Progress update every 10 images
    if (i + 1) % 10 == 0:
        print(f"\n  Progress: {i+1}/{target_images} processed, {success_count} successful, {visible_gas_count} with visible gas")

print(f"\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Successfully saved: {success_count}/{target_images}")
print(f"With VISIBLE gas: {visible_gas_count}/{success_count} ({visible_gas_count/success_count*100:.1f}%)" if success_count > 0 else "With VISIBLE gas: 0/0")
print(f"Failed: {failed_count}")
print(f"Images saved to: {OUT_DIR.absolute()}")
print(f"[CHECK] Open images - should have NO axes, NO labels, NO colorbar!")
print("="*70)
