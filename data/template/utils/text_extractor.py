import argparse
from pathlib import Path
import re

LANG_HEADER_RE = re.compile(r"^#[A-Za-z]{2}:\s*$")

def extract_lang(in_path: str, out_path: str, lang: str) -> str:
    """Extracts text in the specified language from a multilingual text block."""
    lang = lang.upper()
    marker = f"#{lang}:"

    collecting = False
    out_lines = []

    with open(in_path, "r") as file:
        for line in file:
            stripped = line.strip()

            if LANG_HEADER_RE.match(stripped):
                if stripped == marker:
                    collecting = True
                else:
                    collecting = False
                continue
            if collecting:
                out_lines.append(line)

    with open(out_path, "w") as file:
        file.writelines(out_lines)

def main():
    parser = argparse.ArgumentParser(description="Extract text in a specified language from multilingual text files.")
    parser.add_argument("input_file", type=Path, help="Path to the input text file.")
    parser.add_argument("output_file", type=Path, help="Path to the output text file.")
    parser.add_argument("language", type=str, help="Language code to extract (e.g., 'EN', 'KO').")
    
    args = parser.parse_args()

    extract_lang(args.input_file, args.output_file, args.language)


if __name__ == "__main__":
    main()