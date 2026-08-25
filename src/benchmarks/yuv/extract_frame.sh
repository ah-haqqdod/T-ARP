#!/usr/bin/env bash

# Check if required arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input.mp4> <frame_index> [output.png]"
    echo "Note: frame_index is 0-indexed (0 = 1st frame, 1 = 2nd frame, etc.)"
    exit 1
fi

INPUT_FILE="$1"
FRAME_INDEX="$2"
# Default to "frame_<INDEX>.png" if output filename is omitted
OUTPUT_FILE="${3:-frame_${FRAME_INDEX}.png}"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File '$INPUT_FILE' not found."
    exit 1
fi

# Extract the exact frame using ffmpeg's select filter
ffmpeg -i "$INPUT_FILE" -vf "select=eq(n\,$FRAME_INDEX)" -vframes 1 "$OUTPUT_FILE" -y

echo "Successfully saved frame $FRAME_INDEX to $OUTPUT_FILE"
