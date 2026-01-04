#!/bin/bash

# Script to identify failed parsing jobs from logs and move corresponding markdown files

# Get the base directory (current directory or specified path)
BASE_DIR="${1:-.}"

# Navigate to the base directory
cd "$BASE_DIR" || exit 1

echo "Processing log files to identify failed documents in: $(pwd)"
echo "========================================"

# Create failed directory if it doesn't exist
mkdir -p failed

# Counter for statistics
total_log_files=0
total_failed_docs=0
total_moved=0
total_not_found=0

# Check if parsing_logs directory exists
if [ ! -d "parsing_logs" ]; then
    echo "Error: parsing_logs directory not found"
    exit 1
fi

# Process each log file
for log_file in parsing_logs/*.log; do
    [ -f "$log_file" ] || continue
    
    ((total_log_files++))
    
    # Extract set_id from log filename (e.g., 1311_parsing.log -> 1311)
    log_basename=$(basename "$log_file")
    set_id=$(echo "$log_basename" | grep -oP '^\d{4}')
    
    if [ -z "$set_id" ]; then
        echo "⚠ Could not extract set_id from: $log_basename, skipping..."
        continue
    fi
    
    echo ""
    echo "Processing log: $log_basename (Set: $set_id)"
    echo "----------------------------------------"
    
    # Check if the set directory exists
    if [ ! -d "$set_id" ]; then
        echo "  ⚠ Set directory $set_id not found, skipping..."
        continue
    fi
    
    # Check if markdown directory exists
    if [ ! -d "$set_id/markdown" ]; then
        echo "  ⚠ No markdown directory in $set_id, skipping..."
        continue
    fi
    
    # Create failed subdirectory for this set
    mkdir -p "failed/$set_id"
    
    # Extract failed PDF filenames using awk
    failed_files=$(awk '
        /Processing:.*\.pdf$/ {
            pdf=$NF
            gsub(/^.*\//, "", pdf)  # Remove path, keep only filename
        }
        /Detected repeat token/ {
            c++
            if(c==5) {
                failed[pdf]=1
            }
        }
        !/Detected repeat token/ {
            c=0
        }
        END {
            for (f in failed) {
                print f
            }
        }
    ' "$log_file")
    
    if [ -z "$failed_files" ]; then
        echo "  ✓ No failed documents found"
        continue
    fi
    
    # Count failed files
    failed_count=$(echo "$failed_files" | wc -l)
    echo "  Found $failed_count failed document(s)"
    echo ""
    
    # Process each failed file
    while IFS= read -r pdf_file; do
        [ -z "$pdf_file" ] && continue
        
        ((total_failed_docs++))
        
        # Convert PDF filename to markdown filename
        # Remove .pdf extension and add .md
        md_basename=$(basename "$pdf_file" .pdf)
        md_file="$set_id/markdown/${md_basename}.md"
        
        # Check if markdown file exists
        if [ ! -f "$md_file" ]; then
            echo "  ✗ Not found: $md_basename.md"
            ((total_not_found++))
            continue
        fi
        
        # Move to failed directory
        if mv "$md_file" "failed/$set_id/" 2>/dev/null; then
            echo "  ✓ Moved: $md_basename.md -> failed/$set_id/"
            ((total_moved++))
        else
            echo "  ✗ Failed to move: $md_basename.md"
        fi
    done <<< "$failed_files"
done

echo ""
echo "========================================"
echo "Summary:"
echo "  Log files processed: $total_log_files"
echo "  Failed documents identified: $total_failed_docs"
echo "  Markdown files moved: $total_moved"
echo "  Markdown files not found: $total_not_found"
echo ""
echo "Failed documents by set:"
for set_dir in failed/*/; do
    [ -d "$set_dir" ] || continue
    set_name=$(basename "$set_dir")
    count=$(find "$set_dir" -maxdepth 1 -name "*.md" -type f | wc -l)
    if [ "$count" -gt 0 ]; then
        printf "  %-10s : %5d files\n" "$set_name" "$count"
    fi
done