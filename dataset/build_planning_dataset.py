import json
import os
import tarfile
import traceback
import urllib.request
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import tokenizers
from argdantic import ArgParser
from pydantic import BaseModel
from pyperplan import grounding
from pyperplan.pddl.parser import Parser
from sklearn.model_selection import train_test_split
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[BOD]",
    "[EOD]",
    "[BOO]",
    "[EOO]",
    "[BOC]",
    "[EOC]",
    "[BOG]",
    "[EOG]",
    "[BOQ]",
    "[EOQ]",
]
cli = ArgParser()


class DataProcessConfig(BaseModel):
    dataset_url: str = (
        "https://storage.googleapis.com/questbench/questbench_data.tar.gz"
    )
    input_dir: str = "data"
    input_file: str = "data/questbench_data/Planning-Q/planning_heldout_7500.csv"
    domain_file: str = "data/questbench_data/Planning-Q/task_pddls/blocks/domain.pddl"
    file_pattern: str = "questbench_data/Planning-Q/task_pddls/blocks/task*.pddl"
    output_dir: str = "data/questbench-planning"
    tokenizer_path: Optional[str] = None


def download_dataset(download_dir, dataset_url):
    # Ensure the directory exists
    Path(download_dir).mkdir(parents=True, exist_ok=True)

    tar_path = os.path.join(download_dir, "questbench_data.tar.gz")

    print(f"Downloading dataset from {dataset_url}...")
    try:
        urllib.request.urlretrieve(dataset_url, tar_path)
        print(f"Download complete. File saved to {tar_path}")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

    print("Extracting dataset...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=download_dir)
        print(f"Extraction complete. Files extracted to {download_dir}")
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        raise
    finally:
        # Always attempt to remove the archive file
        if os.path.exists(tar_path):
            os.remove(tar_path)
            print(f"Removed downloaded archive: {tar_path}")


def read_data(
    config,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, str, list[str]]:
    df = pd.read_csv(config.input_file)
    conditions = df["conditions"].str.replace(r"^frozenset\(|\)$", "", regex=True)
    goals = df["goals"].str.replace(r"^frozenset\(|\)$", "", regex=True)
    questions = df["all_qs"].str.replace(r"^frozenset\(|\)$", "", regex=True)
    answer = df["gt_qs"].str.replace(r"^frozenset\(|\)$", "", regex=True)

    dataset_dir = Path(config.input_dir)
    domain_pddl = Path(config.domain_file).read_text()

    def _parse(domain_file, problem_file):
        # Parsing
        parser = Parser(domain_file, problem_file)
        domain = parser.parse_domain()
        problem = parser.parse_problem(domain)
        return problem

    def _ground(
        problem,
        remove_statics_from_initial_state=True,
        remove_irrelevant_operators=True,
    ):
        task = grounding.ground(
            problem, remove_statics_from_initial_state, remove_irrelevant_operators
        )
        return task

    num_objs_to_problem_spec = {}

    for task_file in dataset_dir.glob(config.file_pattern):
        problem = _parse(config.domain_file, task_file)
        if not problem:
            continue
        task = _ground(problem)  # specific instance
        if len(problem.objects) not in num_objs_to_problem_spec:
            num_objs_to_problem_spec[len(problem.objects)] = {
                "facts": set(task.facts)
                - {f"(on {chr(i + 97)} {chr(i + 97)})" for i in range(26)},
                "operators": task.operators,
                "objects": problem.objects,
            }
        if (
            4 in num_objs_to_problem_spec
            and 5 in num_objs_to_problem_spec
            and 6 in num_objs_to_problem_spec
            and 7 in num_objs_to_problem_spec
        ):
            break

    problem_objects: list[str] = list(
        map(
            lambda objs: str(num_objs_to_problem_spec[int(objs)]), df["num_vars"].values
        )
    )

    return conditions, goals, questions, answer, domain_pddl, problem_objects


def get_vocabulary(
    conditions: pd.Series,
    goals: pd.Series,
    questions: pd.Series,
    answer: pd.Series,
    domain_pddl: str,
    num_objs_to_problem_spec: dict[str, Any],
):
    def _get_chars(data: pd.Series):
        cat = []

        for problem in data.unique():
            cat += problem

        return list(set(cat))

    conditions_chars = _get_chars(data=conditions)
    goals_chars = _get_chars(data=goals)
    questions_chars = _get_chars(data=questions)
    answer_chars = _get_chars(data=answer)
    domain_chars = list(set(domain_pddl))
    objects_chars = list(
        set(
            [
                num_objs_to_problem_spec[num_vars]["objects"]
                for num_vars in num_objs_to_problem_spec.keys()
            ]
        )
    )

    vocabulary = list(
        set(SPECIAL_TOKENS)
        | set(conditions_chars)
        | set(goals_chars)
        | set(questions_chars)
        | set(answer_chars)
        | set(domain_chars)
        | set(objects_chars)
    )
    vocab_size = len(vocabulary)

    return vocabulary, vocab_size


def get_tokenizer():
    tokenizer = tokenizers.Tokenizer(WordLevel(unk_token="[UNK]"))

    tokenizer.add_special_tokens(
        [
            tokenizers.AddedToken(
                token, single_word=False, lstrip=False, rstrip=False, normalized=False
            )
            for token in SPECIAL_TOKENS
        ]
    )

    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordLevelTrainer(min_frequency=1, special_tokens=SPECIAL_TOKENS)

    return tokenizer, trainer


def load_tokenizer_from_path(tokenizer_path: str) -> tokenizers.Tokenizer:
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")

    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
    if hasattr(tokenizer, "no_padding"):
        tokenizer.no_padding()
    if hasattr(tokenizer, "no_truncation"):
        tokenizer.no_truncation()
    return tokenizer


def get_training_data(
    conditions: pd.Series,
    goals: pd.Series,
    questions: pd.Series,
    answer: pd.Series,
    domain_pddl: str,
    problem_objects: list[str],
):
    data = np.concatenate(
        (
            conditions.unique(),
            goals.unique(),
            questions.unique(),
            answer.unique(),
            [domain_pddl],
            problem_objects,
        )  # type: ignore
    )

    return data


def train_tokenizer(tokenizer, trainer, data):
    tokenizer.train_from_iterator(data, trainer)

    temp_encoded = tokenizer.encode_batch(data)
    max_length = max(len(enc.ids) for enc in temp_encoded)

    # Set up padding with the correct token and ID
    tokenizer.enable_padding(
        length=max_length, pad_token="[PAD]", pad_id=tokenizer.token_to_id("[PAD]")
    )
    tokenizer.enable_truncation(max_length=max_length)
    return tokenizer


def add_special_tokens(
    conditions: pd.Series,
    goals: pd.Series,
    questions: pd.Series,
    domain_pddl: str,
    problem_objects: list[str],
):
    processed = []
    for condition, goal, question, objects in zip(
        conditions, goals, questions, problem_objects
    ):
        processed.append(
            " ".join(
                [
                    "[BOD]",
                    domain_pddl,
                    "[EOD]",
                    "[BOO]",
                    objects,
                    "[EOO]",
                    "[BOC]",
                    condition,
                    "[EOC]",
                    "[BOG]",
                    goal,
                    "[EOG]",
                    "[BOQ]",
                    question,
                    "[EOQ]",
                ]
            )
        )

    return processed


def encode(tokenizer, problems_processed, answers_processed):
    problems_encoded = tokenizer.encode_batch(problems_processed)
    answers_encoded = tokenizer.encode_batch(answers_processed)
    return problems_encoded, answers_encoded


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

    metadata = {
        "pad_id": tokenizer.token_to_id("[PAD]"),
        "ignore_label_id": tokenizer.token_to_id("[PAD]"),
        "blank_identifier_id": tokenizer.token_to_id("[PAD]"),
        "vocab_size": vocab_size,
        "seq_len": results["inputs"].shape[1],
        "num_puzzle_identifiers": 1,
        "total_groups": num_samples,
        "mean_puzzle_examples": 1.0,
        "sets": ["all"],
    }

    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata, f)

    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


def main(config):
    input_file_path = config.input_file

    # Check if the input file exists before downloading
    if not os.path.exists(input_file_path):
        print(
            f"Input file {input_file_path} not found. Proceeding to download dataset..."
        )
        try:
            download_dataset(config.input_dir, config.dataset_url)
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return
    else:
        print(f"Input file {input_file_path} already exists. Skipping download.")

    # Verify that the input file exists after attempting download
    if not os.path.exists(input_file_path):
        print(
            f"Error: Input file {input_file_path} does not exist after download attempt."
        )
        return

    try:
        conditions, goals, questions, answers, domain_pddl, problem_objects = read_data(
            config
        )
    except FileNotFoundError:
        traceback.print_exc()
        return
    except Exception as e:
        print(f"Error reading data: {e}")
        traceback.print_exc()
        return

    try:
        if config.tokenizer_path:
            tokenizer = load_tokenizer_from_path(config.tokenizer_path)
        else:
            # get_vocabulary(
            #     conditions,
            #     goals,
            #     questions,
            #     answers,
            #     domain_pddl,
            #     num_objs_to_problem_spec,
            # )
            tokenizer, trainer = get_tokenizer()
            data = get_training_data(
                conditions,
                goals,
                questions,
                answers,
                domain_pddl,
                problem_objects,
            )
            tokenizer = train_tokenizer(tokenizer, trainer, data)
        vocab_size = tokenizer.get_vocab_size()

        # Get the max length for consistent tensor shapes
        problems_processed_list = add_special_tokens(
            conditions, goals, questions, domain_pddl, problem_objects
        )
        if hasattr(tokenizer, "no_padding"):
            tokenizer.no_padding()
        if hasattr(tokenizer, "no_truncation"):
            tokenizer.no_truncation()
        temp_encoded = tokenizer.encode_batch(problems_processed_list)
        max_length = max(len(enc.ids) for enc in temp_encoded)

        # First, encode without padding to get raw sequences
        tokenizer.no_padding()
        answers_encoded_raw = [tokenizer.encode(answer).ids for answer in answers]

        # Then pad all to the same length for consistent tensor shapes
        tokenizer.enable_padding(
            length=max_length, pad_token="[PAD]", pad_id=tokenizer.token_to_id("[PAD]")
        )
        tokenizer.enable_truncation(max_length=max_length)

        # Re-encode with padding enabled
        problems_encoded = tokenizer.encode_batch(problems_processed_list)
        X = encoded_to_numpy(problems_encoded)

        # Create labels tensor - initialize with pad tokens
        y = np.full_like(X, tokenizer.token_to_id("[PAD]"))

        # Process answers and align with input sequences
        for i, answer_ids in enumerate(answers_encoded_raw):
            y[i, : len(answer_ids)] = answer_ids

    except Exception as e:
        print(f"Error during data processing: {e}")
        traceback.print_exc()
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42, train_size=0.8
        )

        # Split the processed data to match the train/test splits
        all_indices = np.arange(len(problems_processed_list))
        train_indices, test_indices = train_test_split(
            all_indices, random_state=42, train_size=0.8
        )

        train_problems_processed = [problems_processed_list[i] for i in train_indices]
        test_problems_processed = [problems_processed_list[i] for i in test_indices]

        save(
            X_train,
            y_train,
            train_problems_processed,
            tokenizer,
            vocab_size,
            "train",
            config,
        )
        save(
            X_test,
            y_test,
            test_problems_processed,
            tokenizer,
            vocab_size,
            "test",
            config,
        )

        # Save the tokenizer
        save_tokenizer(tokenizer, config)
    except Exception as e:
        print(f"Error during train/test split or saving: {e}")
        traceback.print_exc()
        return


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    main(config)


if __name__ == "__main__":
    cli()
