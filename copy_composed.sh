#!/bin/bash

# Source and output directories
SRC="output/chameleonscalar"
DST="output/chameleonscalar_out"

# Create the output directory
mkdir -p "$DST"

# Find each TF* folder
for TF_DIR in "$SRC"/TF*; do
    # Only process directories
    [ -d "$TF_DIR" ] || continue
    # Target path for copying files
    TARGET="$DST/$(basename "$TF_DIR")/neilf/point_cloud/iteration_40000"
    # Source path for files
    SRC_FILES="$TF_DIR/neilf/point_cloud/iteration_40000"
    # Only if source iteration_40000 exists
    if [ -d "$SRC_FILES" ]; then
        mkdir -p "$TARGET"
        # Copy only the files, not subdirectories
        find "$SRC_FILES" -maxdepth 1 -type f -exec cp {} "$TARGET" \;
    fi
done

# Zip the X_out directory
zip -r composed.zip "$DST"
