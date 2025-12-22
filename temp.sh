#!/bin/bash

# Script to organize zip files into zip subdirectories
# within each 4-digit patent ID folder

# Get the base directory (current directory or specified path)
BASE_DIR="${1:-.}"

# Navigate to the base directory
cd "$BASE_DIR" || exit 1

echo "Organizing zip files in: $(pwd)"
echo "----------------------------------------"

# Loop through all 4-digit directories
for dir in */; do
    # Remove trailing slash
    dir_name="${dir%/}"
    
    # Check if it's a 4-digit directory
    if [[ $dir_name =~ ^[0-9]{4}$ ]]; then
        echo "Processing directory: $dir_name"
        
        # Check if there are any zip files in the directory
        zip_count=$(find "$dir_name" -maxdepth 1 -name "*.zip" -type f | wc -l)
        
        if [ "$zip_count" -gt 0 ]; then
            # Create zip subdirectory if it doesn't exist
            mkdir -p "$dir_name/zip"
            
            # Move all zip files to the zip subdirectory
            mv "$dir_name"/*.zip "$dir_name/zip/" 2>/dev/null
            
            echo "  ✓ Moved $zip_count zip files to $dir_name/zip/"
        else
            echo "  ℹ No zip files found in $dir_name"
        fi
    fi
done

echo "----------------------------------------"
echo "Organization complete!"