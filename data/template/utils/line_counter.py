def count_nonempty_lines(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def main():
    import sys
    if len(sys.argv) != 2:
        print("Usage: python line_counter.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    line_count = count_nonempty_lines(file_path)
    print(f"Number of non-empty lines in '{file_path}': {line_count}")

if __name__ == "__main__":
    main()