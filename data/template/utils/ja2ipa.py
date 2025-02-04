#!/usr/bin/env python3
import argparse
import json
import sys
from collections import OrderedDict

from tqdm import tqdm
import pykakasi.kakasi as kakasi

# 1) Attempt optional imports
try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


# ========== Kakasi Converter Setup ==========
kks = kakasi()
kks.setMode('J', 'H')  # Kanji -> Hiragana
kks.setMode('H', 'H')  # Hiragana -> Hiragana
kks.setMode('K', 'H')  # Katakana -> Hiragana
conv = kks.getConverter()

kana_mapper = OrderedDict([
    ("ゔぁ","bˈa"),
    ("ゔぃ","bˈi"),
    ("ゔぇ","bˈe"),
    ("ゔぉ","bˈo"),
    ("ゔゃ","bˈʲa"),
    ("ゔゅ","bˈʲɯ"),
    ("ゔゃ","bˈʲa"),
    ("ゔょ","bˈʲo"),

    ("ゔ","bˈɯ"),

    ("あぁ","aː"),
    ("いぃ","iː"),
    ("いぇ","je"),
    ("いゃ","ja"),
    ("うぅ","ɯː"),
    ("えぇ","eː"),
    ("おぉ","oː"),
    ("かぁ","kˈaː"),
    ("きぃ","kˈiː"),
    ("くぅ","kˈɯː"),
    ("くゃ","kˈa"),
    ("くゅ","kˈʲɯ"),
    ("くょ","kˈʲo"),
    ("けぇ","kˈeː"),
    ("こぉ","kˈoː"),
    ("がぁ","gˈaː"),
    ("ぎぃ","gˈiː"),
    ("ぐぅ","gˈɯː"),
    ("ぐゃ","gˈʲa"),
    ("ぐゅ","gˈʲɯ"),
    ("ぐょ","gˈʲo"),
    ("げぇ","gˈeː"),
    ("ごぉ","gˈoː"),
    ("さぁ","sˈaː"),
    ("しぃ","ɕˈiː"),
    ("すぅ","sˈɯː"),
    ("すゃ","sˈʲa"),
    ("すゅ","sˈʲɯ"),
    ("すょ","sˈʲo"),
    ("せぇ","sˈeː"),
    ("そぉ","sˈoː"),
    ("ざぁ","zˈaː"),
    ("じぃ","dʑˈiː"),
    ("ずぅ","zˈɯː"),
    ("ずゃ","zˈʲa"),
    ("ずゅ","zˈʲɯ"),
    ("ずょ","zˈʲo"),
    ("ぜぇ","zˈeː"),
    ("ぞぉ","zˈeː"),
    ("たぁ","tˈaː"),
    ("ちぃ","tɕˈiː"),
    ("つぁ","tsˈa"),
    ("つぃ","tsˈi"),
    ("つぅ","tsˈɯː"),
    ("つゃ","tɕˈa"),
    ("つゅ","tɕˈɯ"),
    ("つょ","tɕˈo"),
    ("つぇ","tsˈe"),
    ("つぉ","tsˈo"),
    ("てぇ","tˈeː"),
    ("とぉ","tˈoː"),
    ("だぁ","dˈaː"),
    ("ぢぃ","dʑˈiː"),
    ("づぅ","dˈɯː"),
    ("づゃ","zˈʲa"),
    ("づゅ","zˈʲɯ"),
    ("づょ","zˈʲo"),
    ("でぇ","dˈeː"),
    ("どぉ","dˈoː"),
    ("なぁ","nˈaː"),
    ("にぃ","nˈiː"),
    ("ぬぅ","nˈɯː"),
    ("ぬゃ","nˈʲa"),
    ("ぬゅ","nˈʲɯ"),
    ("ぬょ","nˈʲo"),
    ("ねぇ","nˈeː"),
    ("のぉ","nˈoː"),
    ("はぁ","hˈaː"),
    ("ひぃ","çˈiː"),
    ("ふぅ","ɸˈɯː"),
    ("ふゃ","ɸˈʲa"),
    ("ふゅ","ɸˈʲɯ"),
    ("ふょ","ɸˈʲo"),
    ("へぇ","hˈeː"),
    ("ほぉ","hˈoː"),
    ("ばぁ","bˈaː"),
    ("びぃ","bˈiː"),
    ("ぶぅ","bˈɯː"),
    ("ふゃ","ɸˈʲa"),
    ("ぶゅ","bˈʲɯ"),
    ("ふょ","ɸˈʲo"),
    ("べぇ","bˈeː"),
    ("ぼぉ","bˈoː"),
    ("ぱぁ","pˈaː"),
    ("ぴぃ","pˈiː"),
    ("ぷぅ","pˈɯː"),
    ("ぷゃ","pˈʲa"),
    ("ぷゅ","pˈʲɯ"),
    ("ぷょ","pˈʲo"),
    ("ぺぇ","pˈeː"),
    ("ぽぉ","pˈoː"),
    ("まぁ","mˈaː"),
    ("みぃ","mˈiː"),
    ("むぅ","mˈɯː"),
    ("むゃ","mˈʲa"),
    ("むゅ","mˈʲɯ"),
    ("むょ","mˈʲo"),
    ("めぇ","mˈeː"),
    ("もぉ","mˈoː"),
    ("やぁ","jˈaː"),
    ("ゆぅ","jˈɯː"),
    ("ゆゃ","jˈaː"),
    ("ゆゅ","jˈɯː"),
    ("ゆょ","jˈoː"),
    ("よぉ","jˈoː"),
    ("らぁ","ɽˈaː"),
    ("りぃ","ɽˈiː"),
    ("るぅ","ɽˈɯː"),
    ("るゃ","ɽˈʲa"),
    ("るゅ","ɽˈʲɯ"),
    ("るょ","ɽˈʲo"),
    ("れぇ","ɽˈeː"),
    ("ろぉ","ɽˈoː"),
    ("わぁ","ɯˈaː"),
    ("をぉ","oː"),

    ("う゛","bˈɯ"),
    ("でぃ","dˈi"),
    ("でぇ","dˈeː"),
    ("でゃ","dˈʲa"),
    ("でゅ","dˈʲɯ"),
    ("でょ","dˈʲo"),
    ("てぃ","tˈi"),
    ("てぇ","tˈeː"),
    ("てゃ","tˈʲa"),
    ("てゅ","tˈʲɯ"),
    ("てょ","tˈʲo"),
    ("すぃ","sˈi"),
    ("ずぁ","zˈɯa"),
    ("ずぃ","zˈi"),
    ("ずぅ","zˈɯ"),
    ("ずゃ","zˈʲa"),
    ("ずゅ","zˈʲɯ"),
    ("ずょ","zˈʲo"),
    ("ずぇ","zˈe"),
    ("ずぉ","zˈo"),
    ("きゃ","kˈʲa"),
    ("きゅ","kˈʲɯ"),
    ("きょ","kˈʲo"),
    ("しゃ","ɕˈʲa"),
    ("しゅ","ɕˈʲɯ"),
    ("しぇ","ɕˈʲe"),
    ("しょ","ɕˈʲo"),
    ("ちゃ","tɕˈa"),
    ("ちゅ","tɕˈɯ"),
    ("ちぇ","tɕˈe"),
    ("ちょ","tɕˈo"),
    ("とぅ","tˈɯ"),
    ("とゃ","tˈʲa"),
    ("とゅ","tˈʲɯ"),
    ("とょ","tˈʲo"),
    ("どぁ","dˈoa"),
    ("どぅ","dˈɯ"),
    ("どゃ","dˈʲa"),
    ("どゅ","dˈʲɯ"),
    ("どょ","dˈʲo"),
    ("どぉ","dˈoː"),
    ("にゃ","nˈʲa"),
    ("にゅ","nˈʲɯ"),
    ("にょ","nˈʲo"),
    ("ひゃ","çˈʲa"),
    ("ひゅ","çˈʲɯ"),
    ("ひょ","çˈʲo"),
    ("みゃ","mˈʲa"),
    ("みゅ","mˈʲɯ"),
    ("みょ","mˈʲo"),
    ("りゃ","ɽˈʲa"),
    ("りぇ","ɽˈʲe"),
    ("りゅ","ɽˈʲɯ"),
    ("りょ","ɽˈʲo"),
    ("ぎゃ","gˈʲa"),
    ("ぎゅ","gˈʲɯ"),
    ("ぎょ","gˈʲo"),
    ("ぢぇ","dʑˈe"),
    ("ぢゃ","dʑˈa"),
    ("ぢゅ","dʑˈɯ"),
    ("ぢょ","dʑˈo"),
    ("じぇ","dʑˈe"),
    ("じゃ","dʑˈa"),
    ("じゅ","dʑˈɯ"),
    ("じょ","dʑˈo"),
    ("びゃ","bˈʲa"),
    ("びゅ","bˈʲɯ"),
    ("びょ","bˈʲo"),
    ("ぴゃ","pˈʲa"),
    ("ぴゅ","pˈʲɯ"),
    ("ぴょ","pˈʲo"),
    ("うぁ","ɯˈa"),
    ("うぃ","ɯˈi"),
    ("うぇ","ɯˈe"),
    ("うぉ","ɯˈo"),
    ("うゃ","ɯˈʲa"),
    ("うゅ","ɯˈʲɯ"),
    ("うょ","ɯˈʲo"),
    ("ふぁ","ɸˈa"),
    ("ふぃ","ɸˈi"),
    ("ふぅ","ɸˈɯ"),
    ("ふゃ","ɸˈʲa"),
    ("ふゅ","ɸˈʲɯ"),
    ("ふょ","ɸˈʲo"),
    ("ふぇ","ɸˈe"),
    ("ふぉ","ɸˈo"),

    ("あ","a"),
    ("い","i"),
    ("う","ɯ"),
    ("え","e"),
    ("お","o"),
    ("か","kˈa"),
    ("き","kˈi"),
    ("く","kˈɯ"),
    ("け","kˈe"),
    ("こ","kˈo"),
    ("さ","sˈa"),
    ("し","ɕˈi"),
    ("す","sˈɯ"),
    ("せ","sˈe"),
    ("そ","sˈo"),
    ("た","tˈa"),
    ("ち","tɕˈi"),
    ("つ","tsˈɯ"),
    ("て","tˈe"),
    ("と","tˈo"),
    ("な","nˈa"),
    ("に","nˈi"),
    ("ぬ","nˈɯ"),
    ("ね","nˈe"),
    ("の","nˈo"),
    ("は","hˈa"),
    ("ひ","çˈi"),
    ("ふ","ɸˈɯ"),
    ("へ","hˈe"),
    ("ほ","hˈo"),
    ("ま","mˈa"),
    ("み","mˈi"),
    ("む","mˈɯ"),
    ("め","mˈe"),
    ("も","mˈo"),
    ("ら","ɽˈa"),
    ("り","ɽˈi"),
    ("る","ɽˈɯ"),
    ("れ","ɽˈe"),
    ("ろ","ɽˈo"),
    ("が","gˈa"),
    ("ぎ","gˈi"),
    ("ぐ","gˈɯ"),
    ("げ","gˈe"),
    ("ご","gˈo"),
    ("ざ","zˈa"),
    ("じ","dʑˈi"),
    ("ず","zˈɯ"),
    ("ぜ","zˈe"),
    ("ぞ","zˈo"),
    ("だ","dˈa"),
    ("ぢ","dʑˈi"),
    ("づ","zˈɯ"),
    ("で","dˈe"),
    ("ど","dˈo"),
    ("ば","bˈa"),
    ("び","bˈi"),
    ("ぶ","bˈɯ"),
    ("べ","bˈe"),
    ("ぼ","bˈo"),
    ("ぱ","pˈa"),
    ("ぴ","pˈi"),
    ("ぷ","pˈɯ"),
    ("ぺ","pˈe"),
    ("ぽ","pˈo"),
    ("や","jˈa"),
    ("ゆ","jˈɯ"),
    ("よ","jˈo"),
    ("わ","ɯˈa"),
    ("ゐ","i"),
    ("ゑ","e"),
    ("ん","ɴ"),
    ("っ","ʔ"),
    ("ー","ː"),

    ("ぁ","a"),
    ("ぃ","i"),
    ("ぅ","ɯ"),
    ("ぇ","e"),
    ("ぉ","o"),
    ("ゎ","ɯˈa"),
    ("ぉ","o"),

    ("を","o")
])

nasal_sound = OrderedDict([
    # before m, p, b
    ("ɴm","mm"),
    ("ɴb", "mb"),
    ("ɴp", "mp"),
    
    # before k, g
    ("ɴk","ŋk"),
    ("ɴg", "ŋg"),
    
    # before t, d, n, s, z, ɽ
    ("ɴt","nt"),
    ("ɴd", "nd"),
    ("ɴn","nn"),
    ("ɴs", "ns"),
    ("ɴz","nz"),
    ("ɴɽ", "nɽ"),
    
    ("ɴɲ", "ɲɲ"),
    
])

# ========== Basic Conversions ==========
def to_hiragana(text: str) -> str:
    """Convert JP text to Hiragana via Kakasi."""
    return conv.do(text)

def hiragana_to_ipa(text: str) -> str:
    """Convert Hiragana to IPA using kana_mapper + nasal_sound."""
    for k, v in kana_mapper.items():
        text = text.replace(k, v)
    for k, v in nasal_sound.items():
        text = text.replace(k, v)
    return text


# ========== 2) MeCab Morphological Tokenization ==========
def mecab_spaced_reading(text: str) -> str:
    """
    Use MeCab for morphological analysis and produce a spaced reading.
    Special override: if token is particle (助詞) and surface == "は",
    we treat it as "わ" so final IPA becomes "wa" instead of "ha."
    """
    if not MECAB_AVAILABLE:
        # If MeCab is not installed, fallback to the raw text
        return text

    tagger = MeCab.Tagger()
    node = tagger.parseToNode(text)

    tokens = []
    while node:
        surface = node.surface
        features = node.feature.split(",")

        # Typically: features = [pos, pos_sub1, pos_sub2, pos_sub3, conjugation, base_form, reading, pronunciation]
        if len(features) >= 8:
            pos = features[0]        # e.g. 名詞, 助詞, 動詞, 形容詞...
            # base_form = features[6]
            # reading   = features[7]
            # If it's "助詞" and surface=="は", override
            if pos == "助詞" and surface == "は":
                tokens.append("わ")
            else:
                # Otherwise, just append surface
                tokens.append(surface)
        else:
            # In rare cases, no features
            tokens.append(surface)

        node = node.next

    # Join tokens with space
    spaced_str = " ".join(tokens)
    # Convert that entire spaced string to Hiragana (for any Kanji, Katakana)
    spaced_hira = to_hiragana(spaced_str)
    return spaced_hira


# ========== 3) spaCy Morphological Tokenization ==========
_spacy_nlp = None

def load_spacy_japanese():
    """
    Lazy-load the spaCy model. We'll call nlp() on it.
    Requires 'ja_core_news_sm' or similar to be installed.
    """
    global _spacy_nlp
    if _spacy_nlp is None:
        _spacy_nlp = spacy.load("ja_core_news_sm")
    return _spacy_nlp

def spacy_spaced_reading(text: str) -> str:
    """
    Use spaCy morphological analysis. If we see a token that is
    'は' with pos_='ADP' (particle), override to 'わ'.
    Then convert everything to Hiragana with Kakasi.
    Join tokens by space.

    (pos_='ADP' is typical for a particle in spaCy's universal POS, 
     but check your model if it uses a different tag for JP.)
    """
    if not SPACY_AVAILABLE:
        return text  # fallback if spaCy is not installed

    nlp = load_spacy_japanese()
    doc = nlp(text)

    tokens = []
    for token in doc:
        # If it's the single character "は" and pos_ is ADP (a particle)
        if token.text == "は" and token.pos_ == "ADP":
            tokens.append("わ")
        else:
            tokens.append(token.text)

    # Join, then convert to Hiragana
    spaced_str = " ".join(tokens)
    spaced_hira = to_hiragana(spaced_str)
    return spaced_hira


# ========== 4) Unified "get spaced reading" function ==========
def get_spaced_reading(text: str, method: str) -> str:
    """
    If method=='mecab', use MeCab morphological approach.
    If method=='spacy', use spaCy morphological approach.
    Otherwise, return original text (no spacing).
    """
    if method == "mecab":
        return mecab_spaced_reading(text)
    elif method == "spacy":
        return spacy_spaced_reading(text)
    else:
        # no morphological approach
        return text


# ========== 5) Main Processing Logic ==========
def process_japanese_text(
    input_file: str,
    output_file: str,
    json_inplace_update: bool = False,
    json_input_field: str = "sentence",
    json_output_field: str = "sentence_ipa",
    use_mecab: bool = False,
    use_spacy: bool = False,
):
    """
    Processes Japanese text to IPA. 
    - If JSON, each entry gets up to 4 fields:
        1) {json_input_field} -> original text
        2) {json_input_field}_spaced -> morphological spaced text (if use_mecab/use_spacy)
        3) {json_output_field} -> IPA from unspaced
        4) {json_output_field}_spaced -> IPA from spaced reading
    - If plain text, we do similarly but just write out lines in a textual format.
    """
    # Decide morphological method:
    morph_method = None
    if use_mecab and use_spacy:
        print("Error: Please choose either MeCab or spaCy, not both.")
        sys.exit(1)
    elif use_mecab:
        morph_method = "mecab"
    elif use_spacy:
        morph_method = "spacy"
    else:
        morph_method = None

    # JSON MODE
    if json_inplace_update:
        try:
            with open(input_file, "r", encoding="utf-8") as fin:
                data = json.load(fin)

            for entry in tqdm(data, desc="Processing JSON entries"):
                if json_input_field not in entry:
                    continue

                original_text = entry[json_input_field]
                # 1) Unspaced → IPA
                hira_unspaced = to_hiragana(original_text)
                ipa_unspaced = hiragana_to_ipa(hira_unspaced)
                entry[json_output_field] = ipa_unspaced

                # 2) If morphological approach, get spaced reading, then spaced IPA
                if morph_method is not None:
                    spaced_hira = get_spaced_reading(original_text, morph_method)
                    entry[f"{json_input_field}_spaced"] = spaced_hira
                    ipa_spaced = hiragana_to_ipa(spaced_hira)
                    entry[f"{json_output_field}_spaced"] = ipa_spaced

            with open(output_file, "w", encoding="utf-8") as fout:
                json.dump(data, fout, ensure_ascii=False, indent=4)

        except FileNotFoundError:
            print(f"Error: Input file '{input_file}' not found.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in '{input_file}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

    else:
        # PLAIN TEXT MODE
        try:
            with open(input_file, "r", encoding="utf-8") as fin:
                lines = fin.readlines()

            with open(output_file, "w", encoding="utf-8") as fout:
                for line in tqdm(lines, desc="Processing lines"):
                    line = line.strip()
                    if not line:
                        fout.write("\n")
                        continue

                    # A) Unspaced -> IPA
                    hira_unspaced = to_hiragana(line)
                    ipa_unspaced = hiragana_to_ipa(hira_unspaced)

                    if morph_method is not None:
                        # B) Spaced reading -> spaced IPA
                        spaced_hira = get_spaced_reading(line, morph_method)
                        ipa_spaced = hiragana_to_ipa(spaced_hira)

                        fout.write(f"[Original]    : {line}\n")
                        fout.write(f"[Unspaced IPA]: {ipa_unspaced}\n")
                        fout.write(f"[Spaced Hira] : {spaced_hira}\n")
                        fout.write(f"[Spaced IPA]  : {ipa_spaced}\n\n")
                    else:
                        fout.write(ipa_unspaced + "\n")

        except FileNotFoundError:
            print(f"Error: Input file '{input_file}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")


# ========== 6) Command-Line Entry Point ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JP text to IPA, forcing 'は' particle => 'わ' using either MeCab or spaCy.")
    parser.add_argument("input_file", nargs="?", default="input.txt",
                        help="Path to the input file (text or JSON).")
    parser.add_argument("output_file", nargs="?", default="output_ipa.txt",
                        help="Path to the output file.")

    parser.add_argument("-j", "--json_inplace_update", action="store_true",
                        help="Treat input file as JSON and update in-place with IPA fields.")
    parser.add_argument("--json_input_field", default="sentence",
                        help="JSON field for original text (default: sentence).")
    parser.add_argument("--json_output_field", default="sentence_ipa",
                        help="JSON field for IPA output (default: sentence_ipa).")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--use_mecab", action="store_true",
                       help="Use MeCab morphological approach for spacing and forcing 'は' => 'わ'.")
    group.add_argument("--use_spacy", action="store_true",
                       help="Use spaCy morphological approach for spacing and forcing 'は' => 'わ'.")

    args = parser.parse_args()

    process_japanese_text(
        input_file=args.input_file,
        output_file=args.output_file,
        json_inplace_update=args.json_inplace_update,
        json_input_field=args.json_input_field,
        json_output_field=args.json_output_field,
        use_mecab=args.use_mecab,
        use_spacy=args.use_spacy
    )

