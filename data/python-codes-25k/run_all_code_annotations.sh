#!/bin/bash
# run_all_code_annotations.sh

# Color codes
BLUE='\033[0;34m'
GREEN='\033[0;32m'
MAGENTA='\033[0;35m'
NC='\033[0m'

# filename to convert
filename="$1"
output_folder="annotated_files"
mapped_file="${filename}.mapped"

if [[ ! -d "$output_folder" ]]; then
  mkdir -p  "$output_folder"
fi

# List of every supported mode
MODES=(
  general exact keywords nesting param_nesting argnum
  dot_nesting name_kind literals semantic comments scope
)

for mode in "${MODES[@]}"; do
  echo -e "${MAGENTA}=== MODE: $mode ===${NC}"
  echo

  # Run the highlighter and color its stdout magenta
  python code_highlighter.py --mode "$mode" "$filename" 2>&1

  # Move the .mapped file to a mode-specific file
  mv "$mapped_file" "${output_folder}/$mode.txt"
  echo

  # Show the mapped output in green
  echo -e "${GREEN}--- ${output_folder}/$mode.txt (mapped output) ---${NC}"
  cat "$mode.txt"
  echo

  # Show the original source in blue
  echo -e "${BLUE}--- $filename (original source) ---${NC}"
  cat "$filename"
  echo
done

