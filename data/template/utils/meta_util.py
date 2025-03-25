# meta_util.py

import pickle
import argparse
from rich.console import Console
from rich.theme import Theme
from rich.table import Table

def load_meta(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def view_tokens(meta_path):
    meta = load_meta(meta_path)
    print(f"Vocabulary Size: {meta['vocab_size']}")
    print("String to Index Mapping:")
    for k, v in list(meta["stoi"].items())[:]:
        print(f"{k}: {v}")
    print("Index to String Mapping:")
    for k, v in list(meta["itos"].items())[:]:
        print(f"{k}: {v}")

def visualize_histogram(meta_path, top_n=20):
    """
    Visualizes token frequencies (if present) in a meta.pkl as a bar chart.
    Requires 'token_counts' to exist in the meta dictionary.
    """
    meta = load_meta(meta_path)
    token_counts = meta.get("token_counts")
    if not token_counts:
        print("No 'token_counts' found in the meta. Cannot visualize histogram.")
        return

    console = Console(theme=Theme({"info": "bold blue"}))
    console.print("[info]Histogram of Token Counts:[/info]")
    table = Table("Token ID", "Token String", "Count", "Bar", title="Token Count Histogram")

    # Sort tokens by descending frequency
    sorted_counts = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    max_count = max(token_counts.values())

    # If we have 'itos', we can display the actual token strings
    itos = meta.get("itos", {})

    # Print the top_n tokens (or all if you prefer)
    for token_id, count in sorted_counts[:top_n]:
        token_str = itos.get(token_id, f"<UNK:{token_id}>")
        # Build a simple bar
        bar_len = 20
        filled = int((count / max_count) * bar_len)
        bar_str = "█" * filled

        table.add_row(str(token_id), repr(token_str), str(count), bar_str)

    console.print(table)

def merge_metas(meta_path1, meta_path2, output_path):
    meta1 = load_meta(meta_path1)
    meta2 = load_meta(meta_path2)

    # Start with the stoi and itos from the first meta file
    stoi = meta1["stoi"].copy()
    itos = meta1["itos"].copy()

    # Update with tokens from the second meta, resolving conflicts by prioritizing the first meta
    for token, id in meta2["stoi"].items():
        if token not in stoi:
            new_id = max(itos.keys()) + 1
            stoi[token] = new_id
            itos[new_id] = token

    vocab_size = len(stoi)
    meta = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos}
    with open(output_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"Merged meta saved to {output_path}, prioritizing {meta_path1}.")

def create_meta_from_text(text_file, output_path, special_chars={"<ukn>": 0}):
    with open(text_file, "r") as f:
        tokens = f.read().split("\n")

    stoi = {token: i for i, token in enumerate(tokens, start=len(special_chars))}
    stoi.update(special_chars)  # Add special characters with predefined IDs
    itos = {i: token for token, i in stoi.items()}
    vocab_size = len(stoi)

    meta = {"vocab_size": vocab_size, "stoi": stoi, "itos": itos}
    with open(output_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"Meta created from text and saved to {output_path}.")

def export_tokens(meta_path, output_path):
    meta = load_meta(meta_path)
    with open(output_path, "w") as f:
        for i in range(meta["vocab_size"]):
            token = meta["itos"][i]
            if token == "\n":
                token = "\\n"
            elif token == "\t":
                token = "\\t"
            elif token == "\r":
                token = "\\r"
            f.write(token + "\n")
    print(f"Tokens exported to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Utility for handling token metadata.")

    parser.add_argument("--view", type=str, help="Path to the meta.pkl file to view.")
    parser.add_argument("--merge", nargs=2, help="Paths to the two meta.pkl files to merge.")
    parser.add_argument(
        "--create",
        nargs=2,
        help="Path to the input text file and the output meta.pkl file for creation.",
    )
    parser.add_argument(
        "--export",
        nargs=2,
        help="Path to the meta.pkl file and the output text file for exporting tokens.",
    )
    parser.add_argument(
        "--hist",
        type=str,
        help="Path to the meta.pkl file for visualizing token frequency histogram.",
    )
    parser.add_argument(
        "--hist-top",
        type=int,
        default=20,
        help="(Optional) Number of top tokens to show in histogram. Defaults to 20.",
    )

    args = parser.parse_args()

    if args.view:
        view_tokens(args.view)
    elif args.merge:
        merge_metas(args.merge[0], args.merge[1], "merged_meta.pkl")
    elif args.create:
        create_meta_from_text(args.create[0], args.create[1])
    elif args.export:
        export_tokens(args.export[0], args.export[1])
    elif args.hist:
        visualize_histogram(args.hist, args.hist_top)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

