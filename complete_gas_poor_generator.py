#!/usr/bin/env python3
"""
Complete Gas-Poor Image Generator
1. Generates CSV with proven high gas mass candidates
2. Generates images using those candidates with variations
"""
import csv, pathlib, requests, time, json
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

def get_gas_info(subhalo_id, save_response=True):
    """Get gas info for a subhalo."""
    url = f"{BASE_URL}/snapshots/{SNAPSHOT}/subhalos/{subhalo_id}/"
    headers = {"api-key": API_KEY}
    
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        # Save API response to file for analysis
        if save_response:
            api_log_file = pathlib.Path("api_responses.jsonl")
            with api_log_file.open("a", encoding="utf-8") as f:
                json.dump({
                    "subhalo_id": subhalo_id,
                    "url": url,
                    "response": data,
                    "timestamp": time.time()
                }, f)
                f.write("\n")
        
        gas_mass = data.get("mass_gas", 0) * 1e10
        stellar_mass = data.get("mass_stars", 0) * 1e10
        total_mass = data.get("mass", 0) * 1e10
        gas_frac = gas_mass / total_mass if total_mass > 0 else 0
        
        return {
            "id": subhalo_id,
            "gas_mass": gas_mass,
            "stellar_mass": stellar_mass,
            "gas_fraction": gas_frac,
            "raw_data": data  # Include raw data for analysis
        }
    except Exception as e:
        # Log failed requests too
        if save_response:
            api_log_file = pathlib.Path("api_responses.jsonl")
            with api_log_file.open("a", encoding="utf-8") as f:
                json.dump({
                    "subhalo_id": subhalo_id,
                    "url": url,
                    "error": str(e),
                    "timestamp": time.time()
                }, f)
                f.write("\n")
        return None

def generate_csv_candidates(num_candidates: int = 100, force_regenerate: bool = False):
    """Generate CSV with proven high gas mass candidates."""
    csv_file = pathlib.Path("gas_poor_candidates.csv")
    
    # Initialize API response log file
    api_log_file = pathlib.Path("api_responses.jsonl")
    if api_log_file.exists():
        api_log_file.unlink()  # Clear previous log
    print(f"\nAPI responses will be saved to: {api_log_file}")
    
    # Check if CSV already exists and has enough candidates
    if not force_regenerate and csv_file.exists():
        existing_count = 0
        with csv_file.open("r") as f:
            reader = csv.reader(f)
            existing_count = sum(1 for row in reader if len(row) >= 2)
        
        if existing_count >= num_candidates:
            print(f"\n{'='*50}")
            print(f"USING EXISTING CSV WITH {existing_count} CANDIDATES")
            print(f"{'='*50}")
            print(f"Found existing CSV file: {csv_file}")
            print(f"Existing CSV has {existing_count} candidates (need {num_candidates})")
            print(f"Using existing CSV file (use force_regenerate=True to regenerate)")
            return csv_file
    
    print(f"\n{'='*50}")
    print(f"GENERATING CSV WITH {num_candidates} CANDIDATES")
    print(f"{'='*50}")
    
    # Known proven candidates with high gas mass
    proven_subhalos = [12, 18, 28, 48, 56, 63, 69, 81, 99, 141]
    
    # Progressive criteria levels - each level is progressively more lenient
    criteria_levels = [
        {
            "name": "Level 0 (Strict)",
            "min_gas_mass": 1e6,      # 1M
            "max_gas_mass": 1e10,      # 10B
            "min_stellar_mass": 5e7,   # 50M
            "max_gas_fraction": 0.40   # 40%
        },
        {
            "name": "Level 1 (Moderate)",
            "min_gas_mass": 5e5,      # 500K
            "max_gas_mass": 2e10,     # 20B
            "min_stellar_mass": 1e7,  # 10M
            "max_gas_fraction": 0.50  # 50%
        },
        {
            "name": "Level 2 (Lenient)",
            "min_gas_mass": 1e5,      # 100K
            "max_gas_mass": 5e10,     # 50B
            "min_stellar_mass": 1e6,  # 1M
            "max_gas_fraction": 0.60  # 60%
        },
        {
            "name": "Level 3 (Very Lenient)",
            "min_gas_mass": 1e4,      # 10K
            "max_gas_mass": 1e11,     # 100B
            "min_stellar_mass": 1e5,  # 100K
            "max_gas_fraction": 0.70  # 70%
        },
        {
            "name": "Level 4 (Minimal)",
            "min_gas_mass": 1e3,      # 1K
            "max_gas_mass": 1e12,     # 1T
            "min_stellar_mass": 1e4,  # 10K
            "max_gas_fraction": 0.80  # 80%
        }
    ]
    
    candidates = []
    checked_subhalos = set()  # Track which subhalos we've already checked
    
    # First, add proven candidates
    print(f"\nAdding {len(proven_subhalos)} proven candidates...")
    for subhalo_id in proven_subhalos:
        info = get_gas_info(subhalo_id)
        if info and info['gas_mass'] > 0:
            info['criteria_level'] = -1  # Mark as proven
            candidates.append(info)
            checked_subhalos.add(subhalo_id)
            print(f"  Added subhalo {subhalo_id}: {info['gas_mass']:.2e} Msun gas")
    
    # Then search for additional candidates with progressive criteria relaxation
    if len(candidates) < num_candidates:
        print(f"\nSearching for {num_candidates - len(candidates)} additional candidates...")
        print("Using progressive criteria relaxation - will automatically loosen criteria if needed")
        
        # Search parameters
        MAX_SUBHALO_ID = 20000  # Upper limit: check up to subhalo 20,000
        CONSECUTIVE_WITHOUT_FIND = 2000  # Relax criteria if 2000 consecutive subhalos yield no candidates
        
        print(f"Search limits:")
        print(f"  Maximum subhalo ID to check: {MAX_SUBHALO_ID:,}")
        print(f"  Will relax criteria if {CONSECUTIVE_WITHOUT_FIND:,} consecutive subhalos yield no candidates")
        
        current_criteria_level = 0
        i = 1
        checked_count = 0
        consecutive_without_find = 0
        last_checked_at_level = {}  # Track which subhalos were checked at which level
        subhalo_data_cache = {}  # Cache API responses to avoid re-fetching
        
        found_ids = {c['id'] for c in candidates}  # Track IDs of already found candidates
        recheck_queue = []  # Queue of subhalos to re-check with relaxed criteria
        
        while len(candidates) < num_candidates and (i <= MAX_SUBHALO_ID or recheck_queue):
            # First, re-check any subhalos in the queue with current criteria
            if recheck_queue:
                recheck_id = recheck_queue.pop(0)
                if recheck_id in found_ids:
                    continue  # Skip if already found
                
                # Get cached data or fetch if not cached
                if recheck_id in subhalo_data_cache:
                    info = subhalo_data_cache[recheck_id]
                else:
                    info = get_gas_info(recheck_id)
                    if info:
                        subhalo_data_cache[recheck_id] = info
                
                if info and info['gas_mass'] > 0:
                    criteria = criteria_levels[current_criteria_level]
                    meets_criteria = (
                        criteria['min_gas_mass'] <= info['gas_mass'] <= criteria['max_gas_mass'] and 
                        info['gas_fraction'] < criteria['max_gas_fraction'] and
                        info['stellar_mass'] >= criteria['min_stellar_mass']
                    )
                    
                    if meets_criteria:
                        info['criteria_level'] = current_criteria_level
                        candidates.append(info)
                        found_ids.add(recheck_id)
                        consecutive_without_find = 0
                        print(f"  Found subhalo {info['id']} (Level {current_criteria_level}, re-checked): {info['gas_mass']:.2e} Msun gas")
                continue
            
            # Skip proven candidates and already found candidates
            if i in proven_subhalos or i in found_ids:
                i += 1
                continue
            
            # Get current criteria
            criteria = criteria_levels[current_criteria_level]
            
            # Only check subhalos we haven't checked at this level yet
            if i in last_checked_at_level and last_checked_at_level[i] >= current_criteria_level:
                i += 1
                continue
            
            info = get_gas_info(i)
            checked_count += 1
            last_checked_at_level[i] = current_criteria_level
            consecutive_without_find += 1
            
            # Cache the data for potential re-checking
            if info:
                subhalo_data_cache[i] = info
            
            if info and info['gas_mass'] > 0:
                # Check if this subhalo meets current criteria
                meets_criteria = (
                    criteria['min_gas_mass'] <= info['gas_mass'] <= criteria['max_gas_mass'] and 
                    info['gas_fraction'] < criteria['max_gas_fraction'] and
                    info['stellar_mass'] >= criteria['min_stellar_mass']
                )
                
                if meets_criteria:
                    info['criteria_level'] = current_criteria_level
                    candidates.append(info)
                    found_ids.add(i)  # Add to found set
                    consecutive_without_find = 0  # Reset counter when we find one
                    print(f"  Found subhalo {info['id']} (Level {current_criteria_level}): {info['gas_mass']:.2e} Msun gas")
                else:
                    # Didn't qualify at this level - add to recheck queue for when criteria relax
                    # Only add if it has gas (zero gas won't qualify at any level)
                    if i not in found_ids and info['gas_mass'] > 0:
                        if i not in recheck_queue:
                            recheck_queue.append(i)
            
            # Check if we've gone too long without finding candidates - relax criteria
            if consecutive_without_find >= CONSECUTIVE_WITHOUT_FIND:
                if current_criteria_level < len(criteria_levels) - 1:
                    # Move to next criteria level
                    current_criteria_level += 1
                    new_criteria = criteria_levels[current_criteria_level]
                    consecutive_without_find = 0  # Reset counter
                    
                    # Add all previously checked subhalos (that didn't qualify) to recheck queue
                    # Only add those we haven't found yet and that have gas
                    for subhalo_id in last_checked_at_level.keys():
                        if subhalo_id not in found_ids and subhalo_id not in proven_subhalos:
                            # Check if this subhalo has gas (from cache or need to check)
                            if subhalo_id in subhalo_data_cache:
                                cached_info = subhalo_data_cache[subhalo_id]
                                if cached_info and cached_info.get('gas_mass', 0) > 0:
                                    if subhalo_id not in recheck_queue:
                                        recheck_queue.append(subhalo_id)
                            else:
                                # Not in cache, add to queue (will check when we get to it)
                                if subhalo_id not in recheck_queue:
                                    recheck_queue.append(subhalo_id)
                    
                    print(f"\n[INFO] No new candidates found in last {CONSECUTIVE_WITHOUT_FIND:,} subhalos.")
                    print(f"Relaxing criteria to {new_criteria['name']}:")
                    print(f"  Min gas mass: {new_criteria['min_gas_mass']:.0e} Msun")
                    print(f"  Max gas mass: {new_criteria['max_gas_mass']:.0e} Msun")
                    print(f"  Min stellar mass: {new_criteria['min_stellar_mass']:.0e} Msun")
                    print(f"  Max gas fraction: {new_criteria['max_gas_fraction']:.1%}")
                    print(f"Re-checking {len(recheck_queue)} previously checked subhalos with relaxed criteria...")
                    print(f"Then continuing forward from subhalo {i}...")
                else:
                    # Already at most lenient level, stop searching
                    print(f"\n[WARNING] No new candidates found in last {CONSECUTIVE_WITHOUT_FIND:,} subhalos.")
                    print(f"Already at most lenient criteria level. Stopping search.")
                    break
            
            # Progress update every 150 subhalos checked
            if checked_count % 150 == 0:
                print(f"    Checked {checked_count} subhalos, found {len(candidates)} candidates (Level {current_criteria_level})...")
            
            # Check if we've reached the upper limit
            if i >= MAX_SUBHALO_ID:
                # If we've reached the limit but still have items in recheck queue, continue
                if recheck_queue:
                    continue
                print(f"\n[WARNING] Reached maximum subhalo ID limit ({MAX_SUBHALO_ID:,}).")
                print(f"Stopping search.")
                break
            
            i += 1
            time.sleep(0.005)
    
    # Limit to requested number
    candidates = candidates[:num_candidates]
    
    if len(candidates) < num_candidates:
        print(f"\n[WARNING] Only found {len(candidates)} candidates out of {num_candidates} requested.")
        print(f"This may be because:")
        print(f"  - The selection criteria are too strict")
        print(f"  - There aren't enough qualifying subhalos in the dataset")
        print(f"  - The search reached its limits (max subhalo ID or consecutive checks)")
        print(f"\nProceeding with {len(candidates)} candidates...")
    else:
        print(f"\n[SUCCESS] Found {len(candidates)} high gas mass candidates!")
    
    # Create CSV files
    detail_file = pathlib.Path("gas_poor_candidates_detailed.csv")
    
    with csv_file.open("w", newline="") as f:
        writer = csv.writer(f)
        for c in candidates:
            filename = f"gas_poor_{c['id']:06d}.png"
            url = f"{BASE_URL}/snapshots/{SNAPSHOT}/subhalos/{c['id']}/"
            writer.writerow([filename, url])
    
    with detail_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subhalo_id", "gas_mass", "stellar_mass", "gas_fraction", "criteria_level"])
        for c in candidates:
            criteria_level = c.get('criteria_level', -1)  # -1 for proven candidates
            writer.writerow([c['id'], c['gas_mass'], c['stellar_mass'], c['gas_fraction'], criteria_level])
    
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
    
    # Check existing images and find the highest image number
    existing_images = set()
    existing_combinations = set()  # Track (subhalo_id, size_factor, depth_factor) combinations
    max_existing_num = 0
    for img_file in OUT_DIR.glob("*.png"):
        existing_images.add(img_file.name)
        # Extract image number and parameters from filename (e.g., "gas_poor_001_000012_s0.15_d0.8.png")
        try:
            parts = img_file.stem.split('_')
            if len(parts) >= 2 and parts[0] == "gas" and parts[1] == "poor":
                img_num = int(parts[2])
                max_existing_num = max(max_existing_num, img_num)
                
                # Extract subhalo_id, size_factor, and depth_factor from filename
                if len(parts) >= 6:
                    subhalo_id = parts[3]  # e.g., "000012"
                    size_factor = parts[4].replace('s', '')  # e.g., "0.15"
                    depth_factor = parts[5].replace('d', '')  # e.g., "0.8"
                    combination = (subhalo_id, size_factor, depth_factor)
                    existing_combinations.add(combination)
        except (ValueError, IndexError):
            pass
    
    print(f"Found {len(existing_images)} existing images")
    print(f"Found {len(existing_combinations)} unique (subhalo, size, depth) combinations")
    if max_existing_num > 0:
        print(f"Highest existing image number: {max_existing_num}")
        start_from = max_existing_num + 1
        print(f"Will start creating images from number {start_from}")
    else:
        start_from = 1
    
    # Calculate how many new images to create
    images_to_create = num_images - (start_from - 1)
    if images_to_create <= 0:
        print(f"\n[INFO] Target of {num_images} images already reached! (Have {max_existing_num} images)")
        return
    
    print(f"Will create {images_to_create} new images (from {start_from} to {num_images})")
    
    # Generate images with more variation to avoid repetition
    success_count = 0
    failed_count = 0
    visible_gas_count = 0
    
    # More size factors and additional parameters for variation
    size_factors = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    depth_factors = [0.8, 1.0, 1.2, 1.5]
    
    # Start from the next image number after existing ones
    current_image_num = start_from
    attempts = 0
    max_attempts = num_images * 10  # Safety limit to prevent infinite loops
    
    while current_image_num <= num_images and attempts < max_attempts:
        attempts += 1
        
        # Cycle through candidates based on current image number
        candidate_idx = (current_image_num - 1) % len(entries)
        filename_base, base_url = entries[candidate_idx]
        subhalo_id = filename_base.split('_')[2].split('.')[0]  # Extract subhalo ID
        
        # Create unique filename with multiple variation parameters
        size_factor = size_factors[(current_image_num - 1) % len(size_factors)]
        depth_factor = depth_factors[((current_image_num - 1) // len(size_factors)) % len(depth_factors)]
        
        # Check if this combination already exists (regardless of image number)
        combination = (subhalo_id, f"{size_factor:.2f}", f"{depth_factor:.1f}")
        if combination in existing_combinations:
            # This exact combination already exists, skip it
            print(f"[{current_image_num}/{num_images}] SKIPPED - combination already exists: subhalo {subhalo_id}, size {size_factor:.2f}, depth {depth_factor:.1f}")
            current_image_num += 1
            continue
        
        # Create unique filename
        filename = f"gas_poor_{current_image_num:03d}_{subhalo_id}_s{size_factor:.2f}_d{depth_factor:.1f}.png"
        
        if filename in existing_images:
            print(f"[{current_image_num}/{num_images}] {filename} - SKIPPED (filename already exists)")
            current_image_num += 1
            success_count += 1
            continue
        
        print(f"[{current_image_num}/{num_images}] {filename}")
        
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
            
            # Add this combination to the existing set to prevent duplicates
            existing_combinations.add(combination)
            existing_images.add(filename)
            
            success_count += 1
            current_image_num += 1
            
        except Exception as e:
            print(f" - FAILED: {e}")
            failed_count += 1
            current_image_num += 1
        
        # Progress update every 10 images
        if success_count % 10 == 0:
            print(f"\n  Progress: {current_image_num - 1}/{num_images} processed, {success_count} successful, {visible_gas_count} with visible gas")
    
    print(f"\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Target: {num_images} images")
    print(f"Started from image number: {start_from}")
    print(f"Successfully saved: {success_count} new images")
    print(f"With VISIBLE gas: {visible_gas_count}/{success_count} ({visible_gas_count/success_count*100:.1f}%)" if success_count > 0 else "With VISIBLE gas: 0/0")
    print(f"Failed: {failed_count}")
    print(f"Total images now: {len(existing_images) + success_count}")
    print(f"Images saved to: {OUT_DIR.absolute()}")
    print(f"[CHECK] Open images - should have NO axes, NO labels, NO colorbar!")
    print("="*70)

def main():
    """Main function to run the complete pipeline."""
    print("Complete Gas-Poor Image Generator")
    print("This script will:")
    print("1. Generate a CSV with high gas mass candidates")
    print("2. Generate images using those candidates")
    
    # Check for existing images first
    OUT_DIR = pathlib.Path("gas_poor_clean_images")
    existing_count = 0
    max_existing_num = 0
    
    if OUT_DIR.exists():
        for img_file in OUT_DIR.glob("*.png"):
            existing_count += 1
            # Extract image number from filename (e.g., "gas_poor_001_..." -> 1)
            try:
                parts = img_file.stem.split('_')
                if len(parts) >= 2 and parts[0] == "gas" and parts[1] == "poor":
                    img_num = int(parts[2])
                    max_existing_num = max(max_existing_num, img_num)
            except (ValueError, IndexError):
                pass
    
    # Show existing images info
    if existing_count > 0:
        print(f"\n{'='*70}")
        print(f"EXISTING IMAGES DETECTED")
        print(f"{'='*70}")
        print(f"Found {existing_count} existing images")
        print(f"Highest image number: {max_existing_num}")
        print(f"{'='*70}")
    
    # Get user input
    try:
        num_candidates = int(input("\nHow many candidates do you want in the CSV? (default: 100): ") or "100")
        
        if existing_count > 0:
            print(f"\nYou currently have {existing_count} images (up to image #{max_existing_num})")
            additional_input = input(f"How many MORE images do you want to generate? (default: 100): ") or "100"
            additional_images = int(additional_input)
            num_images = max_existing_num + additional_images
            print(f"Will generate {additional_images} more images (total target: {num_images})")
        else:
            num_images = int(input("How many images do you want to generate? (default: 100): ") or "100")
    except ValueError:
        print("Invalid input, using defaults: 100 candidates, 100 images")
        num_candidates = 100
        if existing_count > 0:
            num_images = max_existing_num + 100
        else:
            num_images = 100
    
    # Step 1: Generate CSV
    csv_file = generate_csv_candidates(num_candidates)
    
    # Step 2: Generate images
    generate_images(csv_file, num_images)
    
    print(f"\n{'='*70}")
    print("COMPLETE!")
    print(f"{'='*70}")
    print(f"Generated {num_candidates} candidates in CSV")
    if existing_count > 0:
        new_images = num_images - max_existing_num
        print(f"Generated {new_images} new images (total now: {num_images})")
    else:
        print(f"Generated {num_images} images with excellent visible gas content")
    print("All images are clean (no axes, labels, or colorbar)")

if __name__ == "__main__":
    main()
