#!/bin/bash

# Script to extract .md files from patent ID subdirectories in markdown/
# and delete the subdirectories after extraction

# Get the base directory (current directory or specified path)
BASE_DIR="${1:-.}"

# Navigate to the base directory
cd "$BASE_DIR" || exit 1

echo "Extracting markdown files from subdirectories in: $(pwd)"
echo "========================================"

# Counter for statistics
total_dirs=0
total_md_files=0
total_dirs_removed=0
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
        
        # Check if markdown directory exists
        if [ ! -d "$dir_name/markdown" ]; then
            echo "  ⚠ No markdown folder found in $dir_name, skipping..."
            continue
        fi
        
        # Count subdirectories in markdown/
        subdir_count=$(find "$dir_name/markdown" -maxdepth 1 -type d ! -path "$dir_name/markdown" | wc -l)
        
        if [ "$subdir_count" -eq 0 ]; then
            echo "  ℹ No subdirectories found in $dir_name/markdown"
            continue
        fi
        
        echo "  Found $subdir_count subdirectories"
        echo ""
        
        # Process each subdirectory
        for subdir in "$dir_name/markdown"/*/; do
            # Skip if not a directory
            [ -d "$subdir" ] || continue
            
            ((total_dirs++))
            
            # Get the subdirectory name (patent ID)
            patent_id=$(basename "$subdir")
            
            # Count .md files in this subdirectory
            md_count=$(find "$subdir" -maxdepth 1 -name "*.md" -type f | wc -l)
            
            if [ "$md_count" -eq 0 ]; then
                echo "  ⚠ No .md files found in $patent_id/"
                # Still remove the directory even if no .md files
                rm -rf "$subdir"
                ((total_dirs_removed++))
                continue
            fi
            
            # Process .md files
            if [ "$md_count" -eq 1 ]; then
                # Single .md file - rename it to patent_id.md
                md_file=$(find "$subdir" -maxdepth 1 -name "*.md" -type f | head -1)
                target_file="$dir_name/markdown/${patent_id}.md"
                
                if mv "$md_file" "$target_file" 2>/dev/null; then
                    echo "  ✓ Extracted: ${patent_id}.md"
                    ((total_md_files++))
                else
                    echo "  ✗ Failed to extract from $patent_id/"
                    ((total_failed++))
                    continue
                fi
            else
                # Multiple .md files - preserve original names with prefix
                echo "  ℹ Found $md_count .md files in $patent_id/"
                local_count=0
                for md_file in "$subdir"/*.md; do
                    [ -f "$md_file" ] || continue
                    
                    original_name=$(basename "$md_file")
                    target_file="$dir_name/markdown/${patent_id}_${original_name}"
                    
                    if mv "$md_file" "$target_file" 2>/dev/null; then
                        echo "    ✓ Extracted: ${patent_id}_${original_name}"
                        ((total_md_files++))
                        ((local_count++))
                    else
                        echo "    ✗ Failed: ${original_name}"
                        ((total_failed++))
                    fi
                done
                echo "    Extracted $local_count of $md_count files"
            fi
            
            # Remove the subdirectory after extracting .md files
            if rm -rf "$subdir" 2>/dev/null; then
                ((total_dirs_removed++))
            else
                echo "  ✗ Failed to remove directory: $patent_id/"
            fi
        done
    fi
done

echo ""
echo "========================================"
echo "Summary:"
echo "  Total subdirectories processed: $total_dirs"
echo "  Total .md files extracted: $total_md_files"
echo "  Total directories removed: $total_dirs_removed"
echo "  Failed operations: $total_failed"