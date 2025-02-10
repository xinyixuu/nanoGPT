import os
import re
import json
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

def download_file(url, filename):
    """
    Download a file from a given URL with a progress bar, if it doesn't exist yet.
    """
    if os.path.exists(filename):
        print(f"{filename} already exists; skipping download.")
        return

    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KiB
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)

    with open(filename, "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

    if total_size != 0 and progress_bar.n != total_size:
        print("Warning: Download may have been incomplete.")
    else:
        print(f"Downloaded {filename}")

def find_parquet_links(url, range_start=None, range_end=None):
    """
    Scrape the given URL for .parquet?download=true links.
    If range_start/end is provided, only return files where the numeric portion
    (e.g., '00005' in 'train-00005-of-00203.parquet') is within that range.
    """
    # Grab the HTML
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")

    # Collect all .parquet?download=true links
    all_links = [
        "https://huggingface.co" + a["href"]
        for a in soup.find_all("a", href=True)
        if a["href"].endswith(".parquet?download=true")
    ]

    # If no range filtering, just return everything
    if range_start is None and range_end is None:
        return all_links

    # Filter by numeric portion of 'train-XXXX-of-YYYY.parquet'
    pattern = re.compile(r"train-(\d+)-of-\d+\.parquet\?download=true")
    filtered_links = []
    for link in all_links:
        match = pattern.search(link)
        if not match:
            # Skip files that don't match the naming pattern
            continue

        idx = int(match.group(1))  # e.g. "00005" -> 5
        if (range_start is None or idx >= range_start) and \
           (range_end is None   or idx <= range_end):
            filtered_links.append(link)

    return filtered_links

def clean_strings(df):
    """
    Replace invalid UTF-8 characters with the replacement char (�),
    to avoid OverflowError or encoding issues in df.to_json().
    """
    def safe_unicode(s):
        if isinstance(s, str):
            return s.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        return s

    return df.applymap(safe_unicode)

def main(url, range_start, range_end, include_keys, output_json):
    # 1) Find all relevant Parquet links
    parquet_links = find_parquet_links(url, range_start, range_end)
    if not parquet_links:
        print("No Parquet files found within the specified range.")
        return

    # Create directory to store downloaded Parquet files
    download_dir = "downloaded_parquets"
    os.makedirs(download_dir, exist_ok=True)

    # This will hold all rows for the final JSON array
    all_records = []

    # 2) Download and process each Parquet file
    for link in parquet_links:
        filename = link.split("/")[-1].split("?")[0]  # e.g. 'train-00000-of-00203.parquet'
        parquet_path = os.path.join(download_dir, filename)

        # Download the file if needed
        download_file(link, parquet_path)

        # 3) Read Parquet into a pandas DataFrame
        print(f"Reading Parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)

        # 4) Filter columns to only include requested keys that actually exist
        existing_cols = [col for col in include_keys if col in df.columns]
        if not existing_cols:
            print(f"None of the requested keys ({include_keys}) exist in {filename}; skipping.")
            continue

        df = df[existing_cols]

        # Optionally fix invalid UTF-8 sequences
        df = clean_strings(df)

        # Convert to a list of dicts
        records = df.to_dict(orient="records")
        all_records.extend(records)

    # 5) Write final JSON array to file (contains only the requested keys)
    if all_records:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)
        print(f"Done! Wrote {len(all_records)} records to {output_json}")
    else:
        print("No records to write.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download specified Parquet files, then save only certain keys to one JSON array.")

    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="URL to a Hugging Face dataset folder containing .parquet?download=true links."
    )
    parser.add_argument(
        "--range_start",
        type=int,
        default=None,
        help="Numeric start index of the parquet files (e.g., 0 for train-00000-of-XXXX)."
    )
    parser.add_argument(
        "--range_end",
        type=int,
        default=None,
        help="Numeric end index of the parquet files (e.g., 5 for train-00005-of-XXXX)."
    )
    parser.add_argument(
        "--include_keys",
        type=str,
        nargs="+",
        required=True,
        help="Column names (keys) from the parquet to include in the final JSON."
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="output.json",
        help="Path to the final JSON file containing all records from the selected columns."
    )
    args = parser.parse_args()

    main(
        url=args.url,
        range_start=args.range_start,
        range_end=args.range_end,
        include_keys=args.include_keys,
        output_json=args.output_json
    )

