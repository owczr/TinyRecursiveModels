import ast
import json
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import tokenizers
from argdantic import ArgParser
from common import PuzzleDatasetMetadata
from pydantic import BaseModel
from sklearn.model_selection import train_test_split

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[BOP]",
    "[EOP]",
]

SPECIAL_ADDED_TOKENS = [
    tokenizers.AddedToken(
        token, single_word=False, lstrip=False, rstrip=False, normalized=False
    )
    for token in SPECIAL_TOKENS
]

cli = ArgParser()


class DataProcessConfig(BaseModel):
    dataset_url: str = (
        "hf://datasets/WildEval/ZebraLogic/grid_mode/test-00000-of-00001.parquet"
    )
    input_dir: str = "data"
    output_dir: str = "data/zebra-logic"
    tokenizer_path: Optional[str] = None


def read_data(config) -> pd.DataFrame:
    df = pd.read_parquet(config.dataset_url)
    return df


def augment_data(df: pd.DataFrame) -> pd.DataFrame:
    augmented_df = pd.DataFrame()

    for _, row in df.iterrows():
        names = np.vstack(row["solution"]["rows"])[:, 1]

        sequences = []
        for i in range(len(names)):
            for j in range(i + 1, len(names) + 1):
                sequences.append(names[i:j])

        puzzle = row["puzzle"]
        updated_puzzles = [puzzle + f"\n## Query\n{name_seq}" for name_seq in sequences]

        solution = row["solution"]
        rows = solution["rows"]

        rows_stacked = np.vstack(rows)
        updated_rows = [
            rows_stacked[np.isin(rows_stacked[:, 1], name_seq)]
            for name_seq in sequences
        ]
        updated_solutions = [
            {"header": solution["header"], "rows": row} for row in updated_rows
        ]

        updated_ids = [
            row["id"] + f"-{len(name_seq)}-{idx}"
            for idx, name_seq in enumerate(sequences)
        ]

        updated_sizes = [row["size"] + f"*{len(name_seq)}" for name_seq in sequences]

        now = datetime.now()

        augmented_row = pd.DataFrame(
            {
                "id": updated_ids,
                "size": updated_sizes,
                "puzzle": updated_puzzles,
                "solution": updated_solutions,
                "created_at": now.strftime("%Y-%m-%dT%H:%M:%S.%f"),
            }
        )

        augmented_df = pd.concat([augmented_df, augmented_row])

    return augmented_df


def _safe_literal_eval(value: str):
    if not isinstance(value, str):
        return value

    value = value.strip()
    if not value:
        return value

    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _ensure_sequence(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    if value in ("", None):
        return []
    return [value]


def _normalize_symbol(symbol: str) -> str:
    return symbol.strip().replace(" ", "_")


def process_answers(row: dict[str, np.ndarray]) -> str:
    parsed = {key: list(value) for key, value in row.items()}
    if not parsed:
        return "<EMPTY>"
    return str(parsed)


def process_rules(raw_value: str) -> str:
    parsed = _safe_literal_eval(raw_value)
    clauses = _ensure_sequence(parsed)
    if not clauses:
        return "<EMPTY>"

    formatted_clauses = []
    for clause in clauses:
        clause_items = _ensure_sequence(clause)
        if clause_items:
            clause_text = " ".join(
                _normalize_symbol(str(item)) for item in clause_items
            )
        else:
            clause_text = "<EMPTY>"
        formatted_clauses.append(f"<RULE> {clause_text} </RULE>")

    return " ".join(formatted_clauses)


def add_special_tokens(answers):
    processed = []
    for answer in answers:
        sections = [
            "[BOP]",
            answer,
            "[EOP]",
        ]
        processed.append(" ".join(section for section in sections if section))

    return processed


def encoded_to_numpy(encoded):
    return np.array([np.array(enc.ids) for enc in encoded])


def save_tokenizer(tokenizer, config):
    """Save the trained tokenizer to the output directory."""
    save_dir = os.path.join(config.output_dir, "tokenizer")
    os.makedirs(save_dir, exist_ok=True)

    tokenizer_path = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)


def save(X, y, problems_processed, tokenizer, vocab_size, name, config):
    num_samples = len(problems_processed)

    # puzzle_indices: [0, 1, 2, ..., num_samples]
    puzzle_indices = np.arange(num_samples + 1, dtype=np.int32)

    # group_indices: [0, 1, 2, ..., num_samples]
    group_indices = np.arange(num_samples + 1, dtype=np.int32)

    results = {
        "inputs": X,
        "labels": y,
        "group_indices": group_indices,
        "puzzle_indices": puzzle_indices,
        "puzzle_identifiers": np.zeros(num_samples, dtype=np.int32),
    }

    # metadata = PuzzleDatasetMetadata(
    #     seq_len=81,
    #     vocab_size=10 + 1,  # PAD + "0" ... "9"
    #     pad_id=0,
    #     ignore_label_id=0,
    #     blank_identifier_id=0,
    #     num_puzzle_identifiers=1,
    #     total_groups=len(results["group_indices"]) - 1,
    #     mean_puzzle_examples=1,
    #     total_puzzles=len(results["group_indices"]) - 1,
    #     sets=["all"]
    # )
    metadata = PuzzleDatasetMetadata(
        **{
            "pad_id": tokenizer.token_to_id("[PAD]"),
            "ignore_label_id": tokenizer.token_to_id("[PAD]"),
            "blank_identifier_id": tokenizer.token_to_id("[PAD]"),
            "vocab_size": vocab_size,
            "seq_len": results["inputs"].shape[1],
            "num_puzzle_identifiers": 1,
            "total_groups": num_samples,
            "total_puzzles": num_samples,
            "mean_puzzle_examples": 1.0,
            "sets": ["all"],
        }
    )
    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)

    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


# def load_and_filter_test_data(config, original_category_to_indices):
#     """Load existing test data and filter it to include only original puzzles."""
#     try:
#         # Define the test data directory
#         test_data_dir = os.path.join(config.output_dir, "test")
#
#         # Check if test data exists
#         if not os.path.exists(test_data_dir):
#             print(f"Warning: Test data directory not found at {test_data_dir}")
#             print("Skipping filtered data saving.")
#             return
#
#         # Load the existing test data files
#         print(f"Loading existing test data from {test_data_dir}...")
#
#         # Load inputs, labels, and other arrays
#         inputs_path = os.path.join(test_data_dir, "all__inputs.npy")
#         labels_path = os.path.join(test_data_dir, "all__labels.npy")
#         group_indices_path = os.path.join(test_data_dir, "all__group_indices.npy")
#         puzzle_indices_path = os.path.join(test_data_dir, "all__puzzle_indices.npy")
#         puzzle_identifiers_path = os.path.join(
#             test_data_dir, "all__puzzle_identifiers.npy"
#         )
#
#         if not all(
#             os.path.exists(path)
#             for path in [
#                 inputs_path,
#                 labels_path,
#                 group_indices_path,
#                 puzzle_indices_path,
#                 puzzle_identifiers_path,
#             ]
#         ):
#             print("Warning: Not all required test data files found.")
#             print("Skipping filtered data saving.")
#             return
#
#         # Load the arrays
#         X_full = np.load(inputs_path)
#         y_full = np.load(labels_path)
#         group_indices_full = np.load(group_indices_path)
#         puzzle_indices_full = np.load(puzzle_indices_path)
#         puzzle_identifiers_full = np.load(puzzle_identifiers_path)
#
#         print(f"Loaded test data with {len(X_full)} samples")
#
#         # Collect all indices for original puzzles only
#         original_indices = []
#         for category_indices in original_category_to_indices.values():
#             original_indices.extend(category_indices)
#
#         # Convert to numpy array for indexing
#         original_indices = np.array(original_indices)
#
#         # Filter the data to include only original puzzles
#         X_filtered = X_full[original_indices]
#         y_filtered = y_full[original_indices]
#         group_indices_filtered = group_indices_full[original_indices]
#         puzzle_indices_filtered = puzzle_indices_full[original_indices]
#         puzzle_identifiers_filtered = puzzle_identifiers_full[original_indices]
#
#         # Create directory for filtered data
#         filtered_data_dir = os.path.join(config.output_dir, "test_original_only")
#         os.makedirs(filtered_data_dir, exist_ok=True)
#
#         # Save the filtered data
#         np.save(os.path.join(filtered_data_dir, "all__inputs.npy"), X_filtered)
#         np.save(os.path.join(filtered_data_dir, "all__labels.npy"), y_filtered)
#         np.save(
#             os.path.join(filtered_data_dir, "all__group_indices.npy"),
#             group_indices_filtered,
#         )
#         np.save(
#             os.path.join(filtered_data_dir, "all__puzzle_indices.npy"),
#             puzzle_indices_filtered,
#         )
#         np.save(
#             os.path.join(filtered_data_dir, "all__puzzle_identifiers.npy"),
#             puzzle_identifiers_filtered,
#         )
#
#         # Copy the dataset.json file from the original test directory
#         original_dataset_json = os.path.join(test_data_dir, "dataset.json")
#         if os.path.exists(original_dataset_json):
#             shutil.copy(original_dataset_json, filtered_data_dir)
#
#         print(
#             f"Saved filtered data with {len(X_filtered)} original puzzle samples to {filtered_data_dir}"
#         )
#
#         # Also save the indices mapping for reference
#         indices_mapping_file = os.path.join(filtered_data_dir, "original_indices.json")
#         with open(indices_mapping_file, "w") as f:
#             json.dump(original_indices.tolist(), f, indent=2)
#         print(f"Saved original indices mapping to {indices_mapping_file}")
#
#     except Exception as e:
#         print(f"Error loading or filtering test data: {e}")
#         traceback.print_exc()
#
#
def main(config):
    try:
        df = read_data(config)
        df_aug = augment_data(df)

        # Extract sizes before splitting
        original_sizes = df_aug["size"].tolist()
        puzzles = df_aug["puzzle"]
        answers = df_aug["solution"]

        puzzles = puzzles.fillna("").astype(str)
        answers = answers.fillna("")

        processed_puzzles = puzzles.tolist()
        processed_answers = answers.apply(process_answers).tolist()

        # Prepare problems with special tokens
        problems_processed_list = add_special_tokens(processed_puzzles)

        group_indices = np.load("data/zebra-logic/test/all__group_indices.npy")
        inputs = np.load("data/zebra-logic/test/all__inputs.npy")
        labels = np.load("data/zebra-logic/test/all__labels.npy")
        puzzle_identifiers = np.load(
            "data/zebra-logic/test/all__puzzle_identifiers.npy"
        )
        puzzle_indices = np.load("data/zebra-logic/test/all__puzzle_indices.npy")
        original_metadata = json.loads(
            Path("data/zebra-logic/test/dataset.json").read_text()
        )

    except Exception as e:
        print(f"Error reading data: {e}")
        return

    try:
        # Perform the same train/test split with the same random state
        all_indices = np.arange(len(problems_processed_list))
        train_indices, test_indices = train_test_split(
            all_indices, random_state=42, train_size=0.8
        )

        # Create mapping from test set index to original size
        test_index_to_size = {}
        for i, original_idx in enumerate(test_indices):
            test_index_to_size[i] = original_sizes[original_idx]

        # print("Test set index to size mapping:")
        # for test_idx, size in test_index_to_size.items():
        #     print(f"Index {test_idx}: Size {size}")

        # Save the mapping to a file
        mapping_file = os.path.join(config.output_dir, "test_indices_to_sizes.json")
        os.makedirs(config.output_dir, exist_ok=True)
        with open(mapping_file, "w") as f:
            json.dump(test_index_to_size, f, indent=2)

        # print(f"\nMapping saved to {mapping_file}")

        # Create subdirectories for each size and save indices for each size
        size_to_indices = {}
        for idx, size in test_index_to_size.items():
            if size not in size_to_indices:
                size_to_indices[size] = []
            size_to_indices[size].append(idx)

        # Create directories for original puzzles only (filtering out augmented ones)
        # Extract unique original sizes (first two numbers from format "X*Y*Z")
        # For each original size, we want the augmented puzzles with maximum sequence length
        # which represent the original puzzles
        original_sizes_map = {}
        for size in size_to_indices.keys():
            parts = size.split("*")
            if len(parts) >= 3:  # Should be X*Y*Z format
                original_size = f"{parts[0]}*{parts[1]}"
                sequence_length = int(parts[2])
                if original_size not in original_sizes_map:
                    original_sizes_map[original_size] = {}
                # Store all sequence lengths for this original size
                if sequence_length not in original_sizes_map[original_size]:
                    original_sizes_map[original_size][sequence_length] = []
                original_sizes_map[original_size][sequence_length].extend(
                    size_to_indices[size]
                )

        # For each original size, select only the puzzles with maximum sequence length
        # These represent the original puzzles (not subsequences)
        original_size_to_indices = {}
        for original_size, sequence_lengths in original_sizes_map.items():
            # Find maximum sequence length
            max_length = max(sequence_lengths.keys())
            # Take indices with maximum sequence length
            original_size_to_indices[original_size] = sequence_lengths[max_length]

        # Create category mapping for original puzzles only according to paper's classification
        # Paper's categories:
        # Small (|S| < 10^3): 2×2, 2×3, 2×4, 2×5, 2×6, 3×2, 3×3, 4×2
        # Medium (10^3 ≤ |S| < 10^6): 3×4, 3×5, 3×6, 4×3, 4×4, 5×2, 6×2
        # Large (10^6 ≤ |S| < 10^10): 4×5, 5×3, 4×6, 5×4, 6×3
        # X-Large (|S| ≥ 10^10): 5×5, 6×4, 5×6, 6×5, 6×6
        paper_categories = {
            "Small": ["2*2", "2*3", "2*4", "2*5", "2*6", "3*2", "3*3", "4*2"],
            "Medium": ["3*4", "3*5", "3*6", "4*3", "4*4", "5*2", "6*2"],
            "Large": ["4*5", "5*3", "4*6", "5*4", "6*3"],
            "X-Large": ["5*5", "6*4", "5*6", "6*5", "6*6"],
        }

        # Create reverse mapping for quick lookup
        size_to_paper_category = {}
        for category, sizes in paper_categories.items():
            for size in sizes:
                size_to_paper_category[size] = category

        # Create category mapping for original puzzles only
        original_category_mapping = {}
        for original_size in original_size_to_indices.keys():
            if original_size in size_to_paper_category:
                category = size_to_paper_category[original_size]
            else:
                # Fallback - this shouldn't happen with correct data
                category = "Unknown"
            original_category_mapping[original_size] = category

        # Group indices by category for original puzzles only
        original_category_to_indices = {}
        for original_size, indices in original_size_to_indices.items():
            category = original_category_mapping[original_size]
            if category not in original_category_to_indices:
                original_category_to_indices[category] = []
            original_category_to_indices[category].extend(indices)

        # Create category-specific subdirectories for original puzzles only
        for category, indices in original_category_to_indices.items():
            cat_dir = os.path.join(
                config.output_dir, "test_by_category_original_only", category, "test"
            )
            os.makedirs(cat_dir, exist_ok=True)

            # Save indices belonging to this category
            indices_file = os.path.join(cat_dir, "indices.json")

            with open(indices_file, "w") as f:
                json.dump(indices, f)

            # Save count file
            count_file = os.path.join(cat_dir, "count.txt")
            with open(count_file, "w") as f:
                f.write(f"{len(indices)}\n")

            # Save detailed breakdown of original sizes within this category
            sizes_in_category = []
            for size in original_size_to_indices.keys():
                if original_category_mapping[size] == category:
                    sizes_in_category.append(size)

            breakdown_file = os.path.join(cat_dir, "size_breakdown.json")
            size_breakdown = {}
            for size in sizes_in_category:
                size_breakdown[size] = len(original_size_to_indices[size])

            with open(breakdown_file, "w") as f:
                json.dump(size_breakdown, f, indent=2)

            inputs_filtered = inputs[indices]
            labels_filtered = labels[indices]
            group_indices_filtered = group_indices[indices]
            puzzle_identifiers_filtered = puzzle_identifiers[indices]
            puzzle_indices_filtered = puzzle_indices[indices]

            np.save(os.path.join(cat_dir, "all__inputs.npy"), inputs_filtered)
            np.save(os.path.join(cat_dir, "all__labels.npy"), labels_filtered)
            np.save(
                os.path.join(cat_dir, "all__group_indices.npy"), group_indices_filtered
            )
            np.save(
                os.path.join(cat_dir, "all__puzzle_identifiers.npy"),
                puzzle_identifiers_filtered,
            )
            np.save(
                os.path.join(cat_dir, "all__puzzle_indices.npy"),
                puzzle_indices_filtered,
            )

            metadata = PuzzleDatasetMetadata(
                seq_len=original_metadata["seq_len"],
                vocab_size=original_metadata["vocab_size"],
                pad_id=original_metadata["pad_id"],
                ignore_label_id=original_metadata["ignore_label_id"],
                blank_identifier_id=original_metadata["blank_identifier_id"],
                num_puzzle_identifiers=original_metadata["num_puzzle_identifiers"],
                total_groups=len(inputs_filtered),
                mean_puzzle_examples=original_metadata["mean_puzzle_examples"],
                total_puzzles=len(inputs_filtered),
                sets=original_metadata["sets"],
            )
            with open(os.path.join(cat_dir, "dataset.json"), "w") as f:
                json.dump(metadata.model_dump(), f)

        print(
            f"\nCategory-specific subdirectories for original puzzles only created in {os.path.join(config.output_dir, 'test_by_category_original_only')}"
        )

        # Print original category summary
        # print("\nCategory distribution (original puzzles only):")
        for category in ["Small", "Medium", "Large", "X-Large"]:
            if category in original_category_to_indices:
                count = len(original_category_to_indices[category])
                print(f"{category}: {count} samples")

        # Also create a summary of unique original sizes
        unique_original_sizes = set(original_size_to_indices.keys())
        # print(f"\nUnique original sizes in test set: {sorted(unique_original_sizes)}")
        # print(f"Total unique original sizes: {len(unique_original_sizes)}")
        total_original_samples = sum(
            len(indices) for indices in original_category_to_indices.values()
        )
        # print(f"Total original puzzle samples: {total_original_samples}")

        # Save the filtered indices for original puzzles only
        original_indices_file = os.path.join(
            config.output_dir, "original_puzzle_indices.json"
        )
        with open(original_indices_file, "w") as f:
            json.dump(original_category_to_indices, f, indent=2)
        # print(f"\nOriginal puzzle indices saved to {original_indices_file}")

        # # Load and filter the existing test data to save only original puzzles
        # load_and_filter_test_data(config, original_category_to_indices)

    except Exception as e:
        print(f"Error during train/test split or saving: {e}")
        traceback.print_exc()
        return


@cli.command(singleton=True)
def create_original_puzzles_dirs(config: DataProcessConfig):
    main(config)


if __name__ == "__main__":
    cli()
