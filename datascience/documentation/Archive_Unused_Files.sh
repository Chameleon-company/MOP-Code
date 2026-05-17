#!/bin/bash

# GitHub Repository Archiving Script

# Configuration Settings
ARCHIVE_FOLDER=".ARCHIVE"
MONTHS_THRESHOLD=3

# Create archive folder if it doesn't exist
mkdir -p "$ARCHIVE_FOLDER"

# Find and move files/directories older than 3 months. Iterate for each file in the directory and ignore Git Files and Archive folder itself
find . -maxdepth 1 -type f -mtime +$((MONTHS_THRESHOLD * 30)) -not -path './.git*' -not -path "./$ARCHIVE_FOLDER*" | while read -r file; do
    mv "$file" "$ARCHIVE_FOLDER/"
    echo "Archived file: $file"
done

# Similar to above, do the same for each directory
find . -maxdepth 1 -type d -mtime +$((MONTHS_THRESHOLD * 30)) -not -path './.git*' -not -path "./$ARCHIVE_FOLDER*" | while read -r dir; do
    mv "$dir" "$ARCHIVE_FOLDER/"
    echo "Archived directory: $dir"
done


echo "Archiving and pushing completed successfully!"