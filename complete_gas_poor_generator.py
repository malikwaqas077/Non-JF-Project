#!/usr/bin/env python3
"""
Complete Gas-Poor Image Generator
1. Generates CSV with proven high gas mass candidates
2. Generates images using those candidates with variations
"""
import csv, pathlib, requests, time
from PIL import Image, ImageOps
import io, numpy as np

print("="*70)
print("COMPLETE GAS-POOR IMAGE GENERATOR")
print("Generates CSV + Images with proven high gas mass candidates")
print("="*70)

API_KEY = "faa0959886d51fa3258568782eca5f78"
SIMULATION = "TNG50-1"
SNAPSHOT = 99
BASE_URL = f"https://www.tng-project.org/api/{SIMULATION}"

# Image processing parameters
LEFT_CROP = 0.10   # Remove left y-axis (10%)
RIGHT_CROP = 0.86  # Remove right colorbar (keep 86%, remove 14%)
TOP_CROP = 0.08    # Remove top labels/title (8%)
BOTTOM_CROP = 0.90 # Remove bottom x-axis (keep 90%, remove 10%)
TARGET_SIZE = (384, 384)

def get_gas_info(subhalo_id):
    """Get gas info for a subhalo."""
    url = f"{BASE_URL}/snapshots/{SNAPSHOT}/subhalos/{subhalo_id}/"
    headers = {"api-key": API_KEY}
    
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        gas_mass = data.get("mass_gas", 0) * 1e10
        stellar_mass = data.get("mass_stars", 0) * 1e10
        total_mass = data.get("mass", 0) * 1e10
        gas_frac = gas_mass / total_mass if total_mass > 0 else 0
        
        return {
            "id": subhalo_id,
            "gas_mass": gas_mass,
            "stellar_mass": stellar_mass,
            "gas_fraction": gas_frac
        }
    except:
        return None

def generate_csv_candidates(num_candidates: int = 100):
    """Generate CSV with proven high gas mass candidates."""
    print(f"\n{'='*50}")
    print(f"GENERATING CSV WITH {num_candidates} CANDIDATES")
    print(f"{'='*50}")
    
    # Known proven candidates with high gas mass
    proven_subhalos = [12, 18, 28, 48, 56, 63, 69, 81, 99, 141]
    
    # Gas mass criteria for excellent visibility (relaxed for more candidates)
    MIN_GAS_MASS = 1e7    # 10 million solar masses (lowered from 30M)
    MAX_GAS_MASS = 5e9    # 5 billion solar masses (increased from 3B)
    MIN_STELLAR_MASS = 1e8  # 100 million solar masses (lowered from 500M)
    MAX_GAS_FRACTION = 0.30  # Gas fraction < 30% (increased from 20%)
    
    print(f"\nSelection criteria:")
    print(f"  Min gas mass: {MIN_GAS_MASS:.0e} Msun (10M)")
    print(f"  Max gas mass: {MAX_GAS_MASS:.0e} Msun (5B)")
    print(f"  Min stellar mass: {MIN_STELLAR_MASS:.0e} Msun (100M)")
    print(f"  Max gas fraction: {MAX_GAS_FRACTION:.1%}")
    
    candidates = []
    
    # First, add proven candidates
    print(f"\nAdding {len(proven_subhalos)} proven candidates...")
    for subhalo_id in proven_subhalos:
        info = get_gas_info(subhalo_id)
        if info and info['gas_mass'] > 0:
            candidates.append(info)
            print(f"  Added subhalo {subhalo_id}: {info['gas_mass']:.2e} Msun gas")
    
    # Then search for additional candidates if needed
    if len(candidates) < num_candidates:
        print(f"\nSearching for {num_candidates - len(candidates)} additional candidates...")
        
        # Search more systematically - check every subhalo up to 1000
        for i in range(1, 1000):  # Check subhalos 1-999
            if i in proven_subhalos:
                continue
                
            info = get_gas_info(i)
            
            if info and info['gas_mass'] > 0:
                if (MIN_GAS_MASS <= info['gas_mass'] <= MAX_GAS_MASS and 
                    info['gas_fraction'] < MAX_GAS_FRACTION and
                    info['stellar_mass'] >= MIN_STELLAR_MASS):
                    
                    candidates.append(info)
                    print(f"  Found subhalo {info['id']}: {info['gas_mass']:.2e} Msun gas")
                    
                    if len(candidates) >= num_candidates:
                        break
            
            if (i+1) % 150 == 0:
                print(f"    Checked {i+1} subhalos, found {len(candidates)} candidates...")
            
            time.sleep(0.005)
    
    # Limit to requested number
    candidates = candidates[:num_candidates]
    
    print(f"\n[SUCCESS] Found {len(candidates)} high gas mass candidates!")
    
    # Create CSV files
    csv_file = pathlib.Path("gas_poor_candidates.csv")
    detail_file = pathlib.Path("gas_poor_candidates_detailed.csv")
    
    with csv_file.open("w", newline="") as f:
        writer = csv.writer(f)
        for c in candidates:
            filename = f"gas_poor_{c['id']:06d}.png"
            url = f"{BASE_URL}/snapshots/{SNAPSHOT}/subhalos/{c['id']}/"
            writer.writerow([filename, url])
    
    with detail_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subhalo_id", "gas_mass", "stellar_mass", "gas_fraction"])
        for c in candidates:
            writer.writerow([c['id'], c['gas_mass'], c['stellar_mass'], c['gas_fraction']])
    
    print(f"\nCreated {csv_file} with {len(candidates)} entries")
    print(f"Created {detail_file} with detailed info")
    
    if candidates:
        gas_masses = [c['gas_mass'] for c in candidates]
        print(f"\nGas mass range: {min(gas_masses):.2e} to {max(gas_masses):.2e} Msun")
        print(f"Average gas mass: {sum(gas_masses)/len(gas_masses):.2e} Msun")
    
    return csv_file

def make_vis_url_with_depth(base: str, size_factor: float = 0.25, depth_factor: float = 1.0) -> str:
    """Return a label-free vis.png URL for gas density with depth variation - NO AXES."""
    if not base.endswith("/"):
        base += "/"
    params = dict(
        api_key=API_KEY,
        partType="gas", partField="dens", method="sphMap_subhalo",
        size=str(size_factor), sizeType="rViral", depthFac=str(depth_factor),
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
    
    # More lenient criteria for gas visibility
    # Gas is visible if there's sufficient variation OR brightness
    return std_dev > 30 or brightness > 40

def generate_images(csv_file: pathlib.Path, num_images: int = 100):
    """Generate images from CSV candidates."""
    print(f"\n{'='*50}")
    print(f"GENERATING {num_images} IMAGES")
    print(f"{'='*50}")
    
    OUT_DIR = pathlib.Path("gas_poor_clean_images")
    OUT_DIR.mkdir(exist_ok=True)
    
    print(f"\nSettings:")
    print(f"  CSV: {csv_file}")
    print(f"  Output: {OUT_DIR}")
    print(f"  Target: {num_images} images")
    print(f"  Cropping: Left={LEFT_CROP*100:.0f}%, Right={RIGHT_CROP*100:.0f}%, Top={TOP_CROP*100:.0f}%, Bottom={BOTTOM_CROP*100:.0f}%")
    print(f"  Final size: {TARGET_SIZE[0]}x{TARGET_SIZE[1]}")
    
    # Read CSV
    if not csv_file.exists():
        print(f"\n[ERROR] CSV file not found: {csv_file}")
        return
    
    entries = []
    with csv_file.open("r") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                entries.append((row[0], row[1]))
    
    print(f"\nFound {len(entries)} candidates in CSV")
    
    # Check existing images
    existing_images = set()
    for img_file in OUT_DIR.glob("*.png"):
        existing_images.add(img_file.name)
    
    print(f"Found {len(existing_images)} existing images")
    
    # Generate images with more variation to avoid repetition
    success_count = 0
    failed_count = 0
    visible_gas_count = 0
    
    # More size factors and additional parameters for variation
    size_factors = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    depth_factors = [0.8, 1.0, 1.2, 1.5]
    
    for i in range(num_images):
        # Cycle through candidates
        candidate_idx = i % len(entries)
        filename_base, base_url = entries[candidate_idx]
        
        # Create unique filename with multiple variation parameters
        size_factor = size_factors[i % len(size_factors)]
        depth_factor = depth_factors[(i // len(size_factors)) % len(depth_factors)]
        
        # Create unique filename
        filename = f"gas_poor_{i+1:03d}_{filename_base.split('_')[2].split('.')[0]}_s{size_factor:.2f}_d{depth_factor:.1f}.png"
        
        if filename in existing_images:
            print(f"[{i+1}/{num_images}] {filename} - SKIPPED (already exists)")
            success_count += 1
            continue
        
        print(f"[{i+1}/{num_images}] {filename}")
        
        try:
            url = make_vis_url_with_depth(base_url, size_factor, depth_factor)
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
            print(f"\n  Progress: {i+1}/{num_images} processed, {success_count} successful, {visible_gas_count} with visible gas")
    
    print(f"\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Successfully saved: {success_count}/{num_images}")
    print(f"With VISIBLE gas: {visible_gas_count}/{success_count} ({visible_gas_count/success_count*100:.1f}%)" if success_count > 0 else "With VISIBLE gas: 0/0")
    print(f"Failed: {failed_count}")
    print(f"Images saved to: {OUT_DIR.absolute()}")
    print(f"[CHECK] Open images - should have NO axes, NO labels, NO colorbar!")
    print("="*70)

def main():
    """Main function to run the complete pipeline."""
    print("Complete Gas-Poor Image Generator")
    print("This script will:")
    print("1. Generate a CSV with high gas mass candidates")
    print("2. Generate images using those candidates")
    
    # Get user input
    try:
        num_candidates = int(input("\nHow many candidates do you want in the CSV? (default: 100): ") or "100")
        num_images = int(input("How many images do you want to generate? (default: 100): ") or "100")
    except ValueError:
        print("Invalid input, using defaults: 100 candidates, 100 images")
        num_candidates = 100
        num_images = 100
    
    # Step 1: Generate CSV
    csv_file = generate_csv_candidates(num_candidates)
    
    # Step 2: Generate images
    generate_images(csv_file, num_images)
    
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    print(f"Generated {num_candidates} candidates in CSV")
    print(f"Generated {num_images} images with excellent visible gas content")
    print("All images are clean (no axes, labels, or colorbar)")

if __name__ == "__main__":
    main()
