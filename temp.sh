#!/bin/bash

# Script to extract doc.xml from zip files and rename them
# to match the zip file basename

# Get the base directory (current directory or specified path)
BASE_DIR="${1:-.}"

# Navigate to the base directory
cd "$BASE_DIR" || exit 1

echo "Extracting doc.xml files from zips in: $(pwd)"
echo "=========================================="

# Counter for statistics
total_extracted=0
total_failed=0

# Loop through all 4-digit directories
for dir in */; do
    # Remove trailing slash
    dir_name="${dir%/}"
    
    # Check if it's a 4-digit directory
    if [[ $dir_name =~ ^[0-9]{4}$ ]]; then
        echo ""
        echo "Processing directory: $dir_name"
        echo "----------------------------------------"
        
        # Check if zip directory exists
        if [ ! -d "$dir_name/zip" ]; then
            echo "  ⚠ No zip folder found in $dir_name, skipping..."
            continue
        fi
        
        # Create xml directory if it doesn't exist
        mkdir -p "$dir_name/xml"
        
        # Count zip files
        zip_count=$(find "$dir_name/zip" -maxdepth 1 -name "*.zip" -type f | wc -l)
        
        if [ "$zip_count" -eq 0 ]; then
            echo "  ℹ No zip files found in $dir_name/zip"
            continue
        fi
        
        echo "  Found $zip_count zip files to process"
        
        # Counter for this directory
        dir_extracted=0
        dir_failed=0
        
        # Process each zip file
        for zip_file in "$dir_name/zip"/*.zip; do
            # Get the basename without extension
            zip_basename=$(basename "$zip_file" .zip)
            
            # Target XML filename
            xml_output="$dir_name/xml/${zip_basename}.xml"
            
            # Skip if already exists
            if [ -f "$xml_output" ]; then
                echo "  ⏭  Skipping $zip_basename (already exists)"
                continue
            fi
            
            # Extract doc.xml to a temporary location
            temp_xml=$(mktemp)
            
            if unzip -p "$zip_file" doc.xml > "$temp_xml" 2>/dev/null; then
                # Check if extraction was successful (file not empty)
                if [ -s "$temp_xml" ]; then
                    mv "$temp_xml" "$xml_output"
                    echo "  ✓ Extracted: ${zip_basename}.xml"
                    ((dir_extracted++))
                else
                    echo "  ✗ Failed: $zip_basename (doc.xml not found or empty)"
                    rm -f "$temp_xml"
                    ((dir_failed++))
                fi
            else
                echo "  ✗ Failed: $zip_basename (extraction error)"
                rm -f "$temp_xml"
                ((dir_failed++))
            fi
        done
        
        echo "  Summary: $dir_extracted extracted, $dir_failed failed"
        ((total_extracted += dir_extracted))
        ((total_failed += dir_failed))
    fi
done

echo ""
echo "=========================================="
echo "Extraction complete!"
echo "Total extracted: $total_extracted"
echo "Total failed: $total_failed"