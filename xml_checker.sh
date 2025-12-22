#!/bin/bash

# Script to extract and print doc-number and kind from patent XML files

# Get the base directory (current directory or specified path)
BASE_DIR="${1:-.}"

# Navigate to the base directory
cd "$BASE_DIR" || exit 1

echo "Checking XML files in: $(pwd)"
echo "========================================"

# Counter for statistics
total_files=0
total_processed=0
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
            
            # Get the basename
            xml_basename=$(basename "$xml_file")
            
            # Extract doc-number and kind using grep
            # Look for the ep-patent-document tag and extract attributes
            doc_info=$(grep -oP '<ep-patent-document[^>]*' "$xml_file" | head -1)
            
            if [ -n "$doc_info" ]; then
                # Extract doc-number
                doc_number=$(echo "$doc_info" | grep -oP 'doc-number="\K[^"]+')
                
                # Extract kind
                kind=$(echo "$doc_info" | grep -oP 'kind="\K[^"]+')
                
                if [ -n "$doc_number" ] && [ -n "$kind" ]; then
                    printf "  %-30s | doc-number: %-10s | kind: %s\n" "$xml_basename" "$doc_number" "$kind"
                    ((total_processed++))
                else
                    echo "  ✗ Failed to extract attributes from: $xml_basename"
                    ((total_failed++))
                fi
            else
                echo "  ✗ No ep-patent-document tag found in: $xml_basename"
                ((total_failed++))
            fi
        done
    fi
done

echo ""
echo "========================================"
echo "Summary:"
echo "  Total XML files: $total_files"
echo "  Successfully processed: $total_processed"
echo "  Failed: $total_failed"