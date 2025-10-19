#!/usr/bin/env python3
"""
Script to delete all StarCraft maps that aren't 512x512 from three folders.
Checks dimensions from .scen files and removes corresponding files from all three folders.
"""

import os
from pathlib import Path

# Folders to clean
FOLDERS = ["sc1-map", "sc1-png", "sc1-scen"]
SCEN_FOLDER = "sc1-scen"
TARGET_WIDTH = 512
TARGET_HEIGHT = 512

def get_map_dimensions(scen_file_path):
    """
    Extract map dimensions from a .scen file.
    Format: version 1
            bucket map_name width height ...

    Returns: (width, height) or None if unable to parse
    """
    try:
        with open(scen_file_path, 'r') as f:
            # Skip version line
            f.readline()
            # Read first data line
            first_line = f.readline().strip()
            if not first_line:
                return None

            parts = first_line.split()
            if len(parts) >= 4:
                width = int(parts[2])
                height = int(parts[3])
                return (width, height)
    except (IOError, ValueError, IndexError) as e:
        print(f"Error reading {scen_file_path}: {e}")
        return None

    return None

def get_base_name(filename):
    """
    Get base name from different file types:
    - BigGameHunters.map -> BigGameHunters
    - BigGameHunters.png -> BigGameHunters
    - BigGameHunters.map.scen -> BigGameHunters
    """
    if filename.endswith('.map.scen'):
        return filename[:-9]  # Remove .map.scen
    elif filename.endswith('.map'):
        return filename[:-4]  # Remove .map
    elif filename.endswith('.png'):
        return filename[:-4]  # Remove .png
    return filename

def main():
    script_dir = Path(__file__).parent

    # Get all .scen files
    scen_dir = script_dir / SCEN_FOLDER
    if not scen_dir.exists():
        print(f"Error: {SCEN_FOLDER} folder not found!")
        return

    scen_files = list(scen_dir.glob("*.scen"))
    print(f"Found {len(scen_files)} .scen files")

    maps_to_delete = []
    maps_to_keep = []

    # Check each map's dimensions
    for scen_file in scen_files:
        base_name = get_base_name(scen_file.name)
        dimensions = get_map_dimensions(scen_file)

        if dimensions is None:
            print(f"⚠️  Warning: Could not read dimensions for {base_name}")
            continue

        width, height = dimensions

        if width != TARGET_WIDTH or height != TARGET_HEIGHT:
            maps_to_delete.append((base_name, width, height))
            print(f"❌ {base_name}: {width}x{height} - WILL DELETE")
        else:
            maps_to_keep.append(base_name)
            print(f"✓ {base_name}: {width}x{height} - keeping")

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Maps to keep (512x512): {len(maps_to_keep)}")
    print(f"  Maps to delete: {len(maps_to_delete)}")
    print(f"{'='*60}\n")

    if not maps_to_delete:
        print("No maps to delete. All maps are 512x512!")
        return

    # Confirm deletion
    response = input(f"Delete {len(maps_to_delete)} maps from all 3 folders? (yes/no): ")
    if response.lower() != 'yes':
        print("Deletion cancelled.")
        return

    # Delete files
    deleted_count = 0
    for base_name, width, height in maps_to_delete:
        files_to_delete = [
            script_dir / "sc1-map" / f"{base_name}.map",
            script_dir / "sc1-png" / f"{base_name}.png",
            script_dir / "sc1-scen" / f"{base_name}.map.scen"
        ]

        for file_path in files_to_delete:
            if file_path.exists():
                try:
                    file_path.unlink()
                    deleted_count += 1
                    print(f"  Deleted: {file_path.name}")
                except OSError as e:
                    print(f"  Error deleting {file_path}: {e}")
            else:
                print(f"  Not found: {file_path.name}")

    print(f"\n✓ Deletion complete! Deleted {deleted_count} files.")
    print(f"✓ Kept {len(maps_to_keep)} maps that are 512x512")

if __name__ == "__main__":
    main()
