import string
import argparse
import re
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def transform_cvp(text):
    """
    The original basic transformation:
      - Punctuation -> '1'
      - Vowels -> '2'
      - Consonants -> '3'
      - Everything else -> '_'
    """
    punctuation = string.punctuation
    vowels = 'aeiouAEIOU'
    consonants = ''.join([c for c in string.ascii_letters if c not in vowels])

    transformed = []
    for char in text:
        if char in punctuation:
            transformed.append('1')
        elif char in vowels:
            transformed.append('2')
        elif char in consonants:
            transformed.append('3')
        else:
            # Whitespace, digits, etc.
            transformed.append('_')
    return ''.join(transformed)


def transform_part_of_speech(text):
    """
    Use spaCy to transform text based on Part-of-Speech (POS):

    Table of POS -> single-char:
    -----------------------------------
    ADJ   -> 'a'     (adjective)
    ADP   -> 'b'     (adposition: in, to, during)
    ADV   -> 'c'     (adverb)
    AUX   -> 'd'     (auxiliary verb: is, has, will)
    CONJ  -> 'e'     (conjunction)
    CCONJ -> 'f'     (coordinating conjunction: and, or, but)
    DET   -> 'g'     (determiner: the, a, an)
    INTJ  -> 'h'     (interjection: oh, wow, hey)
    NOUN  -> 'i'     (common noun)
    NUM   -> 'j'     (numeral: one, 2017)
    PART  -> 'k'     (particle: 's, not)
    PRON  -> 'l'     (pronoun: I, you, she)
    PROPN -> 'm'     (proper noun: London, Mary)
    PUNCT -> 'n'     (punctuation: . , ? ! )
    SCONJ -> 'o'     (subordinating conjunction: if, that)
    SYM   -> 'p'     (symbol: $, %, ©, 😝)
    VERB  -> 'q'     (verb)
    X     -> 'r'     (other, e.g. misclassified tokens)
    SPACE -> (special handling)

    Special replacements:
      - Any space character ' ' -> '_'
      - Any newline '\n'       -> '\n'
      - Anything not recognized -> 'x'

    We ensure the output has the **same number of characters** as the input,
    even for large files, by processing in chunks.
    """

    # Mapping from spaCy POS to single character
    pos_map = {
        "ADJ":   'a',
        "ADP":   'b',
        "ADV":   'c',
        "AUX":   'd',
        "CONJ":  'e',
        "CCONJ": 'f',
        "DET":   'g',
        "INTJ":  'h',
        "NOUN":  'i',
        "NUM":   'j',
        "PART":  'k',
        "PRON":  'l',
        "PROPN": 'm',
        "PUNCT": 'n',
        "SCONJ": 'o',
        "SYM":   'p',
        "VERB":  'q',
        "X":     'r',
        # 'SPACE' handled separately
    }

    # We'll store the transformed result as a list of characters (one per input char)
    result = list(text)
    text_length = len(text)

    # Helper function: yield (chunk_of_text, chunk_start_index)
    # so that each chunk is below spaCy's nlp.max_length
    def chunk_text(full_text, chunk_size):
        for i in range(0, len(full_text), chunk_size):
            yield full_text[i : i + chunk_size], i

    # Decide a chunk size that is safely below nlp.max_length
    # (We use some margin; you can adjust as needed.)
    chunk_size = min(nlp.max_length, 1000000)

    # Process each chunk separately and fill in `result`
    for chunk_str, chunk_start in chunk_text(text, chunk_size):
        doc = nlp(chunk_str)
        # For each token in this chunk:
        for token in tqdm(doc, desc="Transforming chunk", leave=False):
            start_i = chunk_start + token.idx  # offset in the *full* text
            token_len = len(token.text)

            if token.is_space:
                # Replace each whitespace character
                for offset in range(token_len):
                    c = token.text[offset]
                    if c == '\n':
                        result[start_i + offset] = '\n'
                    else:
                        result[start_i + offset] = '_'
            else:
                # Non-whitespace token
                mapped_char = pos_map.get(token.pos_, 'x')
                for offset in range(token_len):
                    c = token.text[offset]
                    if c == '\n':
                        result[start_i + offset] = '\\n'
                    else:
                        result[start_i + offset] = mapped_char

    transformed_text = ''.join(result)

    # # Final sanity check
    # if len(transformed_text) != text_length:
    #     raise ValueError(
    #         f"Length mismatch! Input length={text_length}, "
    #         f"Output length={len(transformed_text)}"
    #     )

    return transformed_text

# def transform_pos(text):
#     """
#     Naive POS-based transformation:
#       - Split the text into 'chunks' (words vs. whitespace) using a regex that preserves delimiters.
#       - For each non-whitespace chunk, run nltk.pos_tag([chunk]) to get a single (token, tag) pair.
#       - Convert the tag's first letter to lowercase (e.g. 'NN' -> 'n', 'VB' -> 'v') and
#         repeat it for each character in the chunk.
#       - For whitespace chunks, output underscores.
#       - Example: " dog cat" -> "_nnn_nnn"
#     """
#     # Split so that whitespace is kept as separate chunks
#     chunks = re.split(r'(\s+)', text)
#     transformed = []

#     for chunk in chunks:
#         if not chunk:
#             # Empty string (beginning or end) – skip or turn into ''
#             continue
#         if chunk.isspace():
#             # Turn each whitespace character into '_'
#             transformed.append('_' * len(chunk))
#         else:
#             # Tag this chunk as a single token (naive approach)
#             # If the chunk includes punctuation, pos_tag might label it differently,
#             # but we'll keep it simple here.
#             tagged = nltk.pos_tag([chunk])[0]  # returns (word, POS)
#             word, pos = tagged
#             # Take the first letter of the POS tag in lowercase
#             letter_for_pos = pos[0].lower()
#             # Repeat for each character in the chunk
#             transformed.append(letter_for_pos * len(chunk))

#     return ''.join(transformed)


def transform_position(text):
    """
    Positional transformation:
      - For each 'word' chunk (non-whitespace), replace each character with its 1-based position in that chunk.
      - Whitespace becomes underscores.
      - Example: " dog cat" -> "_123_123"
    """
    chunks = re.split(r'(\s+)', text)
    transformed = []

    for chunk in chunks:
        if not chunk:
            continue
        if chunk.isspace():
            # Replace each space with an underscore
            transformed.append('_' * len(chunk))
        else:
            # Replace each character by 1-based position
            out = []
            for i, _ in enumerate(chunk):
                out.append(str(i+1))
            transformed.append(''.join(out))

    return ''.join(transformed)


def transform_file(filename, method):
    """
    Transforms a file in-place using the selected method.
    """
    try:
        with open(filename, 'r+', encoding='utf-8') as file:
            # Read the entire file content
            file_content = file.read()

            if method == 'cvp':
                transformed_content = transform_cvp(file_content)
            elif method == 'part_of_speech':
                transformed_content = transform_part_of_speech(file_content)
            elif method == 'position':
                transformed_content = transform_position(file_content)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Overwrite file with transformed text
            file.seek(0)
            file.write(transformed_content)
            file.truncate()

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform a text file by replacing characters.")
    parser.add_argument("input_file", help="The input text file to transform.")
    parser.add_argument(
        "--method", 
        choices=["cvp", "part_of_speech", "position"],
        default="cvp",
        help="Which transformation method to use."
    )
    args = parser.parse_args()

    transform_file(args.input_file, args.method)

