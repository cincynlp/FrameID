import argparse
import json
import os
from pathlib import Path
from typing import List

def index_to_letter(index):
    return chr(ord('A') + index)

# Format one sample to {question, answer, num_choices}
def format_sample(sample):
    sentence = sample["question"]
    target_word = sample["target_word"]
    options = sample["options"]
    correct_idx = sample["correct_answer"]
    correct_letter = index_to_letter(correct_idx)

    frame_letters = [index_to_letter(i) for i in range(len(options))]
    option_str = "\n".join(
    [f"{letter}. {option.strip()}" for letter, option in zip(frame_letters, options )])
    option_letter_str = "/".join(frame_letters)

    prompt = f"""Select the most appropriate frame that matches the meaning of the target word in the sentence. (This is a Frame Semantic Parsing task)

Target word: "{target_word}"
Sentence: {sentence.strip()}

Options:
{option_str}

Pick the best option ({option_letter_str}).

Answer:"""

    return {
        "question": prompt,
        "answer": correct_letter,
        "num_choices" : len(options)
}


def process_folder(input_folder: Path, output_folder: Path):
    output_folder.mkdir(parents=True, exist_ok=True)

    for split in ["train", "dev", "test"]:
        inp = input_folder / f"{split}.json"
        if not inp.exists():
            print(f"Skipping missing file: {inp}")
            continue

        with inp.open("r", encoding="utf-8") as f:
            data = json.load(f)

        formatted = [format_sample(sample) for sample in data]
        out_file = output_folder / f"{split}.jsonl"

        with out_file.open("w", encoding="utf-8") as f_out:
            for item in formatted:
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"Saved: {out_file} ({len(formatted)} samples)")


def main():
    parser = argparse.ArgumentParser(
        description="Format train/dev/test JSON datasets into JSONL for fine-tuning."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Path to folder containing train.json, dev.json, test.json"
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Directory to save formatted JSONL files"
    )
    args = parser.parse_args()

    input_folder = Path(args.folder)
    output_folder = Path(args.outdir)

    process_folder(input_folder, output_folder)

if __name__ == "__main__":
    main()

# Usage: python data_for_finetuning.py --folder ../data/fn1.5/json_output --outdir finetune_data 