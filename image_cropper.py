#!/usr/bin/env python3
"""
Background Cropper for Gas-Poor Galaxy Images
Removes excessive blue background to zoom in on the galaxy.
Crops all images in gas_poor_clean_images folder by removing background from all sides.
"""

import pathlib
from PIL import Image, ImageOps
import os

def crop_background_zoom(img_path: pathlib.Path, crop_percentage: float, output_dir: pathlib.Path) -> bool:
    """
    Crop out excessive background to zoom in on the galaxy.
    
    Args:
        img_path: Path to the input image
        crop_percentage: Percentage of background to crop from each side (0.0-0.5)
        output_dir: Directory to save the cropped image
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Open the image
        with Image.open(img_path) as img:
            # Convert to RGB if necessary (handles RGBA, P mode, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get original dimensions
            original_width, original_height = img.size
            
            # Calculate crop amounts
            crop_pixels_x = int(original_width * crop_percentage)
            crop_pixels_y = int(original_height * crop_percentage)
            
            # Calculate new dimensions
            new_width = original_width - (2 * crop_pixels_x)
            new_height = original_height - (2 * crop_pixels_y)
            
            # Ensure we don't crop too much
            if new_width <= 0 or new_height <= 0:
                print(f"  [WARNING] Crop percentage too high, using original image")
                new_width, new_height = original_width, original_height
                crop_pixels_x, crop_pixels_y = 0, 0
            
            # Define crop box (left, top, right, bottom)
            left = crop_pixels_x
            top = crop_pixels_y
            right = original_width - crop_pixels_x
            bottom = original_height - crop_pixels_y
            
            # Crop the image
            cropped_img = img.crop((left, top, right, bottom))
            
            # Save the cropped image
            output_path = output_dir / img_path.name
            cropped_img.save(output_path, 'PNG')
            
        return True
        
    except Exception as e:
        print(f"  [ERROR] Failed to crop {img_path.name}: {e}")
        return False

def main():
    """Main function to crop all images in the gas_poor_clean_images folder."""
    print("="*60)
    print("BACKGROUND CROPPER FOR GAS-POOR GALAXY IMAGES")
    print("Removes excessive blue background to zoom in on galaxies")
    print("="*60)
    
    # Define directories
    input_dir = pathlib.Path("gas_poor_clean_images")
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"[ERROR] Input directory not found: {input_dir}")
        print("Please make sure the gas_poor_clean_images folder exists.")
        return
    
    # Get all PNG files in the input directory
    image_files = list(input_dir.glob("*.png"))
    
    if not image_files:
        print(f"[ERROR] No PNG files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} PNG files in {input_dir}")
    
    # Get user input for crop percentage
    print("\nEnter the crop percentage to zoom in on the galaxy:")
    print("This removes background from all sides to focus on the galaxy.")
    print("Examples:")
    print("  10% = Remove 10% from each side (moderate zoom)")
    print("  20% = Remove 20% from each side (strong zoom)")
    print("  30% = Remove 30% from each side (maximum zoom)")
    
    while True:
        try:
            crop_percentage = float(input("\nCrop percentage (0-50): "))
            
            if crop_percentage < 0 or crop_percentage > 50:
                print("[ERROR] Crop percentage must be between 0 and 50.")
                continue
                
            # Convert to decimal
            crop_percentage = crop_percentage / 100.0
            break
            
        except ValueError:
            print("[ERROR] Please enter a valid number.")
    
    print(f"\nCrop percentage: {crop_percentage*100:.1f}% from each side")
    
    # Ask user if they want to overwrite existing files
    overwrite = input("\nOverwrite existing cropped images? (y/n): ").lower().strip() == 'y'
    
    # Create output directory (same as input for now, but we'll handle overwriting)
    output_dir = input_dir
    
    print(f"\n{'='*50}")
    print("CROPPING IMAGES")
    print(f"{'='*50}")
    
    success_count = 0
    skipped_count = 0
    failed_count = 0
    
    for i, img_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {img_path.name}", end="")
        
        # Check if output file already exists
        output_path = output_dir / img_path.name
        if output_path.exists() and not overwrite:
            print(" - SKIPPED (already exists)")
            skipped_count += 1
            continue
        
        # Crop the image
        if crop_background_zoom(img_path, crop_percentage, output_dir):
            print(f" - ZOOMED (removed {crop_percentage*100:.1f}% background)")
            success_count += 1
        else:
            failed_count += 1
        
        # Progress update every 10 images
        if i % 10 == 0:
            print(f"\n  Progress: {i}/{len(image_files)} processed, {success_count} successful, {skipped_count} skipped, {failed_count} failed")
    
    # Final results
    print(f"\n" + "="*60)
    print("CROPPING RESULTS")
    print("="*60)
    print(f"Total images processed: {len(image_files)}")
    print(f"Successfully cropped: {success_count}")
    print(f"Skipped (already exist): {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Crop percentage: {crop_percentage*100:.1f}% from each side")
    print(f"Output directory: {output_dir.absolute()}")
    print("="*60)

if __name__ == "__main__":
    main()
