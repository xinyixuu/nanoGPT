import json
import os
from tqdm import tqdm

def extract_text_from_jsonl(jsonl_file_path, output_text_file_path, encoding='utf-8'):
    """
    Extracts the 'text' field from each JSON object in a JSONL file and
    appends it to a specified text file. Includes a progress bar.

    Args:
        jsonl_file_path (str): Path to the input JSONL file.
        output_text_file_path (str): Path to the output text file.
        encoding (str): Encoding to use for reading the JSONL and writing the text file.
                        Defaults to 'utf-8'.
    """
    if not os.path.exists(jsonl_file_path):
        print(f"Error: JSONL file not found at '{jsonl_file_path}'")
        return

    print(f"Extracting text from '{jsonl_file_path}' to '{output_text_file_path}'...")

    try:
        # Get the total number of lines for the progress bar
        total_lines = sum(1 for line in open(jsonl_file_path, 'r', encoding=encoding))

        with open(jsonl_file_path, 'r', encoding=encoding) as infile, \
             open(output_text_file_path, 'a', encoding=encoding) as outfile:
            for line_num, line in tqdm(enumerate(infile), total=total_lines, desc="Processing JSONL"):
                try:
                    data = json.loads(line)
                    if "text" in data:
                        outfile.write(data["text"] + "\n")
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed JSON on line {line_num + 1} in '{jsonl_file_path}'")
                except TypeError:
                    print(f"Warning: 'text' field not a string on line {line_num + 1} in '{jsonl_file_path}'")
    except IOError as e:
        print(f"Error during file operation: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print("Extraction complete!")

if __name__ == "__main__":
    # --- Configuration ---
    input_jsonl_file = "aozorabunko_dedupe_clean.jsonl"  # Replace with your JSONL file name
    output_text_file = "input_kanji.txt"  # Replace with your desired output text file name

    # --- Run the extraction ---
    extract_text_from_jsonl(input_jsonl_file, output_text_file)
