#!/bin/bash

# Script to rename XML files to their correct patent IDs
# based on doc-number and kind extracted from the XML content

# Get the base directory (current directory or specified path)
BASE_DIR="${1:-.}"

# Navigate to the base directory
cd "$BASE_DIR" || exit 1

echo "Renaming XML files to correct patent IDs in: $(pwd)"
echo "========================================"

# Counter for statistics
total_files=0
total_renamed=0
total_skipped=0
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
        
        # Check if xml directory exists
        if [ ! -d "$dir_name/xml" ]; then
            echo "  ⚠ No xml folder found in $dir_name, skipping..."
            continue
        fi
        
        # Count xml files
        xml_count=$(find "$dir_name/xml" -maxdepth 1 -name "*.xml" -type f | wc -l)
        
        if [ "$xml_count" -eq 0 ]; then
            echo "  ℹ No XML files found in $dir_name/xml"
            continue
        fi
        
        echo "  Found $xml_count XML files"
        echo ""
        
        # Process each XML file
        for xml_file in "$dir_name/xml"/*.xml; do
            ((total_files++))
            
            # Get the current basename
            current_name=$(basename "$xml_file")
            
            # Extract doc-number and kind using grep
            doc_info=$(grep -oP '<ep-patent-document[^>]*' "$xml_file" | head -1)
            
            if [ -n "$doc_info" ]; then
                # Extract doc-number
                doc_number=$(echo "$doc_info" | grep -oP 'doc-number="\K[^"]+')
                
                # Extract kind
                kind=$(echo "$doc_info" | grep -oP 'kind="\K[^"]+')
                
                if [ -n "$doc_number" ] && [ -n "$kind" ]; then
                    # Create new filename: EP + doc-number + kind + .xml
                    new_name="EP${doc_number}${kind}.xml"
                    
                    # Check if rename is needed
                    if [ "$current_name" = "$new_name" ]; then
                        echo "  ⏭  Already correct: $current_name"
                        ((total_skipped++))
                    else
                        # Check if target file already exists
                        if [ -f "$dir_name/xml/$new_name" ]; then
                            echo "  ⚠  Target exists: $current_name -> $new_name (skipping)"
                            ((total_failed++))
                        else
                            # Rename the file
                            mv "$xml_file" "$dir_name/xml/$new_name"
                            echo "  ✓ Renamed: $current_name -> $new_name"
                            ((total_renamed++))
                        fi
                    fi
                else
                    echo "  ✗ Failed to extract attributes from: $current_name"
                    ((total_failed++))
                fi
            else
                echo "  ✗ No ep-patent-document tag found in: $current_name"
                ((total_failed++))
            fi
        done
    fi
done

echo ""
echo "========================================"
echo "Summary:"
echo "  Total XML files: $total_files"
echo "  Renamed: $total_renamed"
echo "  Already correct: $total_skipped"
echo "  Failed: $total_failed"