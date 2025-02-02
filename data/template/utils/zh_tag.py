import subprocess
from dragonmapper import hanzi
import jieba
import jieba.posseg as pseg
import argparse
import re
import json


def ch_part_of_speech_tag(sentence):
    """Transcribe a Chinese sentence into its phonemes using dragonmapper."""
    try:
        result = pseg.cut(sentence)
        return result
    except Exception as e:
        return f"Error in transcribing Chinese: {str(e)}"
    

def transcribe_chinese(sentence):
    """Transcribe a Chinese sentence into its phonemes using dragonmapper."""
    try:
        result = hanzi.to_ipa(sentence)
        return "".join(result)
    except Exception as e:
        return f"Error in transcribing Chinese: {str(e)}"


def handle_mixed_language(word):
    """Handle a word with potential Chinese, Language, or number content."""
    if word.isdigit():  # Detect numbers but just pass through for now (different in each language)
        return word
    elif any(hanzi.is_simplified(char) for char in word):  # Detect Chinese
        return transcribe_chinese(word)
    else:  # Non-Chinese Word
        return "[[[[[" + word + "]]]]]"


def convert_zh_to_ipa(words):
    """Transcribe Chinese words into their phonemes."""
    result = []
    # Split sentence but keep punctuation
    words = re.findall(r'\w+|[^\w\s]', words, re.UNICODE)
    for word in words:
        if re.match(r'\w+', word):  # Only process words
            # process words one by one
            for w in word:
                result.append(handle_mixed_language(w))
        else:
            result.append(word)  # Preserve punctuation
    return "-".join(result)

def transcribe_multilingual(data, output_file, json_storage=False, json_input_field="sentence"):
    """
    Transcribe multilingual sentences (English and Chinese, with numbers) and save to a file.

    Args:
        data: The input data (list of dictionaries if JSON, list of strings if plain text).
        output_file: Path to the output file.
        json_inplace_update: If True, process JSON input and add IPA to the same JSON.
        json_input_field: The field in the JSON data to transcribe (default: "sentence").
        json_output_field: The field to write the IPA transcription to (default: "sentence_ipa").
    """
    if json_storage:
        # In-place update for JSON data
        result = []
        for item in data:
            if json_input_field in item:
                sentence = item[json_input_field]

                # get the result of part of speech tagging
                temps = ch_part_of_speech_tag(sentence)
                seg_list = jieba.cut(sentence, cut_all=False)
                seg_sentence = " ".join(seg_list)
                word_list = []
                flag_list = []
                ipa_list = []
                for word, flag in temps:
                    word_list.append(word)
                    # assign flage to each signal char in word.
                    times = len(word)
                    flags = flag * times
                    flag_list.append(flags)
                    ipa_list.append(convert_zh_to_ipa(word))
                
                data = {
                    "sentence": sentence,
                    "sentence_with_spaces": seg_sentence,
                    "phonetic": " ".join(word_list),
                    "phonetic_length": len(word_list),
                    "part_of_speech": "_".join(flag_list),
                    "tagging_length": len(flag_list),
                    "ipa": "_".join(ipa_list),
                    "ipa_length": len(ipa_list)
                }
                result.append(data)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)
            print(f"In-place JSON transcription saved to {output_file}")

    else:
        # Standard transcription (either JSON or plain text to plain text output)
        result = []
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data:
                if isinstance(item, dict):
                    sentence = item.get(json_input_field, "")
                else:
                    sentence = item

                # get the result of part of speech tagging
                temps = ch_part_of_speech_tag(sentence)
                seg_list = jieba.cut(sentence, cut_all=False)
                seg_sentence = " ".join(seg_list)
                word_list = []
                flag_list = []
                ipa_list = []
                for word, flag in temps:
                    word_list.append(word)
                    # assign flage to each signal char in word.
                    times = len(word)
                    flags = [flag] * times
                    flag_list.append(flags)
                    ipa_list.append(convert_zh_to_ipa(word))
                
                phonetics = " ".join(word_list)
                taggings = "_".join(flag_list)
                ipas = " ".join(ipa_list)
                f.write("phonetic dataset: " + phonetics + "\n")
                f.write("part of speech: " + taggings + "\n")
                f.write("ipa dataset: " + ipas + "\n")
                f.write("sentence with spaces: " + seg_sentence + "\n")
                f.write("original sentence: " + sentence + "\n")    
                print(phoneics, taggings)  # Print to console

def main():
    parser = argparse.ArgumentParser(description='Transcribe multilingual sentences into IPA phonemes.')
    parser.add_argument('input_file', type=str,
                        help='Path to the input file containing sentences in json format.')
    parser.add_argument('output_file', type=str, help='Path to the output file for IPA transcription.')
    parser.add_argument('--input_type', type=str, choices=['json', 'text'], default='json',
                        help='Type of input file: "json" or "text" (default: json)')
    parser.add_argument("-j", "--json_storage", action="store_true",
                        help="Store the output in JSON format.")
    parser.add_argument("--json_input_field", default="sentence",
                        help="JSON field to read from (default: sentence)")

    args = parser.parse_args()

    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            if args.input_type == 'json':
                data = json.load(f)
            else:
                data = f.readlines()

        transcribe_multilingual(data, args.output_file, args.json_storage, args.json_input_field)

    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{args.input_file}'.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    main()
