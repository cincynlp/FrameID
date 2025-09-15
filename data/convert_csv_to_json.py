import csv
import json
import os
import argparse

def csv_to_json_with_zero_based_choices(csv_file, json_file):
    data = []

    with open(csv_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for i, row in enumerate(reader):
            # Split LU definitions, frame names, and frame definitions by ~$~
            lu_defs = row["lu_defs"].split("~$~")
            frame_names = row["frame_names"].split("~$~")
            frame_defs = row["frame_defs"].split("~$~")
            target_words = row['lu_name']


            if len(lu_defs) != len(frame_names) or len(frame_names) != len(frame_defs):
                raise ValueError("Mismatch in the number of LU definitions, frame names, and frame definitions")


            question = f"{row['sentence']}"


            options = []
            lu_definitions = []
            frames = []
            frame_definitions = []
            for j, (lu_def, frame_name, frame_def) in enumerate(zip(lu_defs, frame_names, frame_defs)):
                lu_definition = lu_def.strip()
                frame_definition = frame_def.strip()
                option = f"Frame: {frame_definition.replace(':', ' -')} ; Lexical Unit definition: {lu_definition.replace(':', ' -')}"
                options.append(option)
                lu_definitions.append(lu_definition)
                frames.append(frame_name)
                frame_definitions.append(frame_def.replace(':', " -"))

            # Get the correct answer as the numeric label (0-based)
            correct_answer = int(row["label"])

            # Get the target word position from lu_head_position
            target_word_position = int(row["lu_head_position"])

            data.append({
                "sample_id": i + 1,
                "question": question,
                "options": options,
                "frame_name" : frames,
                "frame_definitions" : frame_definitions,
                "lu_definitions":  lu_definitions,
                "target_word": target_words,
                "target_word_position": target_word_position,
                "correct_answer": correct_answer
            })

    # Write to JSON file
    with open(json_file, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def convert_folder(folder_path):
    # Create output folder inside given folder
    parent_dir = os.path.dirname(os.path.abspath(folder_path))
    output_folder = os.path.join(parent_dir, "json_output")
    os.makedirs(output_folder, exist_ok=True)

    for split in ["train", "dev", "test"]:
        csv_file = os.path.join(folder_path, f"{split}.csv")
        json_file = os.path.join(output_folder, f"{split}.json")
        if os.path.exists(csv_file):
            print(f"Converting {csv_file} -> {json_file}")
            csv_to_json_with_zero_based_choices(csv_file, json_file)
        else:
            print(f"Warning: {csv_file} not found!")

    print(f"\n All JSONs saved in: {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV files (train/dev/test) to JSON with 0-based labels")
    parser.add_argument("--folder", type=str, required=True, help="Path to folder containing train/dev/test CSVs")
    args = parser.parse_args()

    convert_folder(args.folder)

# Usage : python data_formatter_to_json.py --folder fn1.5/data-csv

