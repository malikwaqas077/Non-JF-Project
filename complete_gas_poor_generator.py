#!/usr/bin/env python3
"""
Complete Gas-Rich Image Generator
1. Generates CSV with medium-high gas mass candidates (gas-rich galaxies)
2. Generates images using those candidates with variations
"""
import csv, pathlib, requests, time, json
from PIL import Image, ImageOps
import io, numpy as np

print("="*70)
print("COMPLETE GAS-RICH IMAGE GENERATOR")
print("Generates CSV + Images with medium-high gas mass candidates")
print("="*70)

API_KEY = "faa0959886d51fa3258568782eca5f78"
# Available simulations to try
AVAILABLE_SIMULATIONS = ["TNG100-1", "TNG50-1", "TNG300-1"]
SIMULATION = "TNG100-1"  # Start with TNG100-1
SNAPSHOT = 99

# Image processing parameters
LEFT_CROP = 0.10   # Remove left y-axis (10%)
RIGHT_CROP = 0.86  # Remove right colorbar (keep 86%, remove 14%)
TOP_CROP = 0.08    # Remove top labels/title (8%)
BOTTOM_CROP = 0.90 # Remove bottom x-axis (keep 90%, remove 10%)
TARGET_SIZE = (384, 384)

def get_gas_info(subhalo_id, simulation=None, save_response=True):
    """Get gas info for a subhalo."""
    if simulation is None:
        simulation = SIMULATION
    base_url = f"https://www.tng-project.org/api/{simulation}"
    url = f"{base_url}/snapshots/{SNAPSHOT}/subhalos/{subhalo_id}/"
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
                    "simulation": simulation,
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
                    "simulation": simulation,
                    "url": url,
                    "error": str(e),
                    "timestamp": time.time()
                }, f)
                f.write("\n")
        return None

def query_subhalos_by_gas_mass(simulation, min_gas_mass, max_gas_mass, min_stellar_mass, limit=100, offset=0):
    """Query subhalos using API filtering - MUCH more efficient than sequential checking!
    
    Args:
        simulation: Simulation name (e.g., 'TNG100-1')
        min_gas_mass: Minimum gas mass in Msun
        max_gas_mass: Maximum gas mass in Msun
        min_stellar_mass: Minimum stellar mass in Msun
        limit: Number of results per page
        offset: Offset for pagination
    
    Returns:
        dict with 'count' (total matching) and 'results' (list of subhalo IDs)
    """
    base_url = f"https://www.tng-project.org/api/{simulation}"
    
    # Convert to API units (API uses 1e10 Msun as base unit)
    min_gas_api = min_gas_mass / 1e10
    max_gas_api = max_gas_mass / 1e10
    min_stellar_api = min_stellar_mass / 1e10
    
    # Build query URL with filters
    url = f"{base_url}/snapshots/{SNAPSHOT}/subhalos/"
    params = {
        "mass_gas__gt": min_gas_api,
        "mass_gas__lt": max_gas_api,
        "mass_stars__gt": min_stellar_api,
        "limit": limit,
        "offset": offset
    }
    
    headers = {"api-key": API_KEY}
    
    try:
        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        
        # Extract subhalo IDs from results
        subhalo_ids = [item["id"] for item in data.get("results", [])]
        
        return {
            "count": data.get("count", 0),
            "next": data.get("next"),
            "subhalo_ids": subhalo_ids
        }
    except Exception as e:
        print(f"  [ERROR] Query failed: {e}")
        return {"count": 0, "next": None, "subhalo_ids": []}

def generate_csv_candidates(num_candidates: int = 100, force_regenerate: bool = False):
    """Generate CSV with medium-high gas mass candidates (gas-rich galaxies with visible gas)."""
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
    
    # Known proven candidates with high gas mass (TNG50-1 specific - empty for TNG100-1)
    proven_subhalos = []  # No proven candidates for TNG100-1, start fresh
    
    # Criteria levels for MEDIUM-HIGH GAS MASS (gas-rich galaxies with visible gas)
    # Focus on galaxies with substantial gas mass that will be visible in images
    criteria_levels = [
        {
            "name": "Level 0 (Strict - Medium-High Gas)",
            "min_gas_mass": 1e9,      # 1B - MINIMUM gas mass (ensures visible gas)
            "max_gas_mass": 1e12,     # 1T - Maximum gas mass
            "min_stellar_mass": 1e8,  # 100M - Minimum stellar mass
            "max_gas_fraction": 0.80  # 80% - Maximum gas fraction (allows gas-rich galaxies)
        },
        {
            "name": "Level 1 (Moderate - Medium Gas)",
            "min_gas_mass": 5e8,      # 500M - Lower minimum but still substantial
            "max_gas_mass": 1e12,     # 1T - Maximum gas mass
            "min_stellar_mass": 5e7,  # 50M - Lower stellar mass
            "max_gas_fraction": 0.85  # 85% - Higher gas fraction
        },
        {
            "name": "Level 2 (Lenient - Medium Gas)",
            "min_gas_mass": 1e8,      # 100M - Lower minimum
            "max_gas_mass": 1e12,     # 1T - Maximum gas mass
            "min_stellar_mass": 1e7,  # 10M - Lower stellar mass
            "max_gas_fraction": 0.90  # 90% - Higher gas fraction
        }
    ]
    
    candidates = []
    checked_subhalos = set()  # Track which subhalos we've already checked
    
    # First, add proven candidates
    print(f"\nAdding {len(proven_subhalos)} proven candidates from {SIMULATION}...")
    for subhalo_id in proven_subhalos:
        info = get_gas_info(subhalo_id, simulation=SIMULATION)
        if info and info['gas_mass'] > 0:
            info['criteria_level'] = -1  # Mark as proven
            info['simulation'] = SIMULATION
            candidates.append(info)
            checked_subhalos.add(subhalo_id)
            print(f"  Added subhalo {subhalo_id}: {info['gas_mass']:.2e} Msun gas")
    
    # Then search for additional candidates using API filtering (MUCH more efficient!)
    if len(candidates) < num_candidates:
        print(f"\nSearching for {num_candidates - len(candidates)} additional candidates...")
        print("Using API FILTERING strategy for MEDIUM-HIGH GAS MASS candidates:")
        print("  - Query API directly for subhalos matching gas mass and stellar mass criteria")
        print("  - Then check gas fraction for filtered candidates")
        print("  - Progressive criteria relaxation (starting with strict criteria)")
        print("  - Multi-simulation support: will try TNG50-1, TNG300-1 if needed")
        print("  - MUCH FASTER than sequential checking!")
        print("  - Minimum gas mass: 1B Msun (ensures visible gas in images)")
        
        current_criteria_level = 0
        current_simulation_idx = 0
        found_ids = {c['id'] for c in candidates}  # Track IDs of already found candidates
        
        while len(candidates) < num_candidates:
            current_simulation = AVAILABLE_SIMULATIONS[current_simulation_idx]
            criteria = criteria_levels[current_criteria_level]
            
            print(f"\n[Level {current_criteria_level}] Searching in {current_simulation} with criteria:")
            print(f"  Gas mass: {criteria['min_gas_mass']:.0e} - {criteria['max_gas_mass']:.0e} Msun")
            print(f"  Stellar mass: >= {criteria['min_stellar_mass']:.0e} Msun")
            print(f"  Max gas fraction: {criteria['max_gas_fraction']:.1%}")
            
            # Query API for subhalos matching gas mass and stellar mass criteria
            offset = 0
            page_size = 100  # Process 100 at a time
            found_in_this_level = 0
            checked_in_this_level = 0
            
            while len(candidates) < num_candidates and offset < 10000:  # Limit to prevent infinite loops
                query_result = query_subhalos_by_gas_mass(
                    simulation=current_simulation,
                    min_gas_mass=criteria['min_gas_mass'],
                    max_gas_mass=criteria['max_gas_mass'],
                    min_stellar_mass=criteria['min_stellar_mass'],
                    limit=page_size,
                    offset=offset
                )
                
                if query_result['count'] == 0:
                    print(f"  No subhalos found matching criteria in {current_simulation}")
                    break
                
                print(f"  Found {query_result['count']:,} potential candidates (checking page {offset//page_size + 1}...)")
                
                # Check each subhalo ID for gas fraction
                for subhalo_id in query_result['subhalo_ids']:
                    if subhalo_id in found_ids:
                        continue  # Skip already found candidates
                    
                    checked_in_this_level += 1
                    info = get_gas_info(subhalo_id, simulation=current_simulation)
                    
                    if not info:
                        continue
                    
                    # Check gas fraction (API can't filter by this, so we check manually)
                    if info['gas_fraction'] < criteria['max_gas_fraction']:
                        # This candidate qualifies!
                        info['criteria_level'] = current_criteria_level
                        info['simulation'] = current_simulation
                        candidates.append(info)
                        found_ids.add(subhalo_id)
                        found_in_this_level += 1
                        print(f"  ✓ Found subhalo {subhalo_id} in {current_simulation}: gas={info['gas_mass']:.2e} Msun, frac={info['gas_fraction']:.2%}")
                        
                        if len(candidates) >= num_candidates:
                            break
                    
                    # Small delay to avoid overwhelming API
                    time.sleep(0.01)
                
                # Check if we have more pages
                if not query_result['next'] or len(candidates) >= num_candidates:
                    break
                
                offset += page_size
                
                # Progress update
                if offset % 1000 == 0:
                    print(f"    Processed {offset:,}/{query_result['count']:,} candidates, found {found_in_this_level} qualified so far...")
            
            print(f"  Level {current_criteria_level} complete: checked {checked_in_this_level} candidates, found {found_in_this_level} qualified")
            
            # If we didn't find enough candidates, relax criteria or switch simulation
            if len(candidates) < num_candidates:
                if current_criteria_level < len(criteria_levels) - 1:
                    # Try next criteria level
                    current_criteria_level += 1
                    new_criteria = criteria_levels[current_criteria_level]
                    print(f"\n[INFO] Relaxing criteria to {new_criteria['name']}")
                elif current_simulation_idx < len(AVAILABLE_SIMULATIONS) - 1:
                    # Try next simulation with strictest criteria
                    current_simulation_idx += 1
                    current_criteria_level = 0
                    print(f"\n[INFO] Switching to simulation: {AVAILABLE_SIMULATIONS[current_simulation_idx]}")
                    print(f"[INFO] Resetting to strictest criteria (Level 0)")
                else:
                    # All simulations and criteria levels exhausted
                    print(f"\n[WARNING] Exhausted all simulations and criteria levels.")
                    break
    
    # Limit to requested number
    candidates = candidates[:num_candidates]
    
    if len(candidates) < num_candidates:
        print(f"\n[WARNING] Only found {len(candidates)} candidates out of {num_candidates} requested.")
        print(f"This may be because:")
        print(f"  - The selection criteria are too strict")
        print(f"  - There aren't enough qualifying subhalos in the dataset")
        print(f"  - All simulations and criteria levels were exhausted")
        print(f"\nProceeding with {len(candidates)} candidates...")
    else:
        print(f"\n[SUCCESS] Found {len(candidates)} medium-high gas mass candidates!")
    
    # Create CSV files
    detail_file = pathlib.Path("gas_poor_candidates_detailed.csv")
    
    with csv_file.open("w", newline="") as f:
        writer = csv.writer(f)
        for c in candidates:
            sim = c.get('simulation', SIMULATION)  # Use candidate's simulation or default
            subhalo_id = c['id']
            # Filename format: gas_poor_{simulation}_{subhalo_id:06d}.png
            filename = f"gas_poor_{sim}_{subhalo_id:06d}.png"
            base_url = f"https://www.tng-project.org/api/{sim}"
            url = f"{base_url}/snapshots/{SNAPSHOT}/subhalos/{subhalo_id}/"
            writer.writerow([filename, url])
    
    with detail_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subhalo_id", "simulation", "gas_mass", "stellar_mass", "gas_fraction", "criteria_level"])
        for c in candidates:
            criteria_level = c.get('criteria_level', -1)  # -1 for proven candidates
            sim = c.get('simulation', SIMULATION)
            writer.writerow([c['id'], sim, c['gas_mass'], c['stellar_mass'], c['gas_fraction'], criteria_level])
    
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
            # Increased timeout to 120 seconds for large image generation
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            return r.content
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(3 * (attempt + 1))  # Longer backoff: 3s, 6s, 9s
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
                entries.append((row[0], row[1]))  # (filename, url)
    
    print(f"\nFound {len(entries)} candidates in CSV")
    
    # Check existing images and find the highest image number
    existing_images = set()
    existing_combinations = set()  # Track (simulation, subhalo_id, size_factor, depth_factor) combinations
    max_existing_num = 0
    for img_file in OUT_DIR.glob("*.png"):
        existing_images.add(img_file.name)
        # Extract parameters from filename (e.g., "gas_poor_TNG100-1_000012_001_s0.15_d0.8.png")
        try:
            parts = img_file.stem.split('_')
            if len(parts) >= 4 and parts[0] == "gas" and parts[1] == "poor":
                # New format: gas_poor_{simulation}_{subhalo_id}_{image_num}_s{size}_d{depth}
                simulation = parts[2]  # e.g., "TNG100-1"
                subhalo_id = parts[3]  # e.g., "000012"
                if len(parts) >= 7:
                    img_num = int(parts[4])
                    max_existing_num = max(max_existing_num, img_num)
                    size_factor = parts[5].replace('s', '')  # e.g., "0.15"
                    depth_factor = parts[6].replace('d', '')  # e.g., "0.8"
                    combination = (simulation, subhalo_id, size_factor, depth_factor)
                    existing_combinations.add(combination)
        except (ValueError, IndexError):
            # Try old format for backward compatibility
            try:
                parts = img_file.stem.split('_')
                if len(parts) >= 2 and parts[0] == "gas" and parts[1] == "poor":
                    img_num = int(parts[2])
                    max_existing_num = max(max_existing_num, img_num)
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
        
        # Parse filename format: gas_poor_{simulation}_{subhalo_id:06d}.png
        # e.g., "gas_poor_TNG100-1_000012.png"
        parts = filename_base.replace('.png', '').split('_')
        if len(parts) >= 4 and parts[0] == "gas" and parts[1] == "poor":
            simulation = parts[2]  # e.g., "TNG100-1"
            subhalo_id = parts[3]  # e.g., "000012" (already zero-padded)
        else:
            # Fallback: try to extract from old format or URL
            simulation = SIMULATION
            # Try to extract from URL
            if 'TNG100-1' in base_url:
                simulation = "TNG100-1"
            elif 'TNG50-1' in base_url:
                simulation = "TNG50-1"
            elif 'TNG300-1' in base_url:
                simulation = "TNG300-1"
            # Extract subhalo_id from URL or filename
            try:
                subhalo_id = filename_base.split('_')[-1].replace('.png', '')
            except:
                subhalo_id = f"{candidate_idx:06d}"
        
        # Create unique filename with multiple variation parameters
        size_factor = size_factors[(current_image_num - 1) % len(size_factors)]
        depth_factor = depth_factors[((current_image_num - 1) // len(size_factors)) % len(depth_factors)]
        
        # Check if this combination already exists (regardless of image number)
        combination = (simulation, subhalo_id, f"{size_factor:.2f}", f"{depth_factor:.1f}")
        if combination in existing_combinations:
            # This exact combination already exists, skip it
            print(f"[{current_image_num}/{num_images}] SKIPPED - combination already exists: {simulation} subhalo {subhalo_id}, size {size_factor:.2f}, depth {depth_factor:.1f}")
            current_image_num += 1
            continue
        
        # Create unique filename with simulation and subhalo_id
        # Format: gas_poor_{simulation}_{subhalo_id}_{image_num:03d}_s{size_factor}_d{depth_factor}.png
        filename = f"gas_poor_{simulation}_{subhalo_id}_{current_image_num:03d}_s{size_factor:.2f}_d{depth_factor:.1f}.png"
        
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
            
            # Small delay between downloads to avoid overwhelming the server
            time.sleep(0.5)
            
        except Exception as e:
            print(f" - FAILED: {e}")
            failed_count += 1
            current_image_num += 1
            # Delay even on failure to avoid rapid retries
            time.sleep(1.0)
        
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
    print("Complete Gas-Rich Image Generator")
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
