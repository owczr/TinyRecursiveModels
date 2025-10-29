import ast
import json
import os
import traceback

import numpy as np
import pandas as pd
import tokenizers
from argdantic import ArgParser
from common import PuzzleDatasetMetadata
from pydantic import BaseModel
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer, UnigramTrainer

SPECIAL_TOKENS = [
    "[PAD]",
    "[UNK]",
    "[BOQ]",
    "[EOQ]",
]

SPECIAL_ADDED_TOKENS = [
    tokenizers.AddedToken(
        token, single_word=False, lstrip=False, rstrip=False, normalized=False
    )
    for token in SPECIAL_TOKENS
]
cli = ArgParser()


class DataProcessConfig(BaseModel):
    dataset_url: str = "hf://datasets/openai/gsm8k/"
    input_dir: str = "data"
    output_dir: str = "data/gsm8k"
    tokenizer_path: str | None = None
    splits: dict[str, str] = {
        "train": "main/train-00000-of-00001.parquet",
        "test": "main/test-00000-of-00001.parquet",
    }


def read_data(
    config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_parquet(config.dataset_url + config.splits["train"])
    test_df = pd.read_parquet(config.dataset_url + config.splits["test"])
    return train_df, test_df


# def get_tokenizer():
#     tokenizer = tokenizers.Tokenizer(WordLevel(unk_token="[UNK]"))  # type: ignore
#     tokenizer.pre_tokenizer = Whitespace()  # type: ignore
#
#     trainer = WordLevelTrainer(min_frequency=1, special_tokens=SPECIAL_TOKENS)
#
#     return tokenizer, trainer
def get_tokenizer(model="bpe", vocab_size=8000, min_freq=2):
    if model == "bpe":
        tok = Tokenizer(BPE(unk_token="[UNK]", byte_fallback=True))
        tok.pre_tokenizer = ByteLevel()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_freq,
            special_tokens=SPECIAL_TOKENS,
            # optional: make sure digits are well represented
            initial_alphabet=list("0123456789.,+-*/=()%"),
        )
    elif model == "unigram":
        tok = Tokenizer(Unigram())
        tok.pre_tokenizer = ByteLevel()
        trainer = UnigramTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            unk_token="[UNK]",
            shrinking_factor=0.75,  # defaults are fine; optional
            n_sub_iterations=2,
        )
    else:
        raise ValueError("model must be 'bpe' or 'unigram'")
    return tok, trainer


def load_tokenizer_from_path(tokenizer_path: str) -> tokenizers.Tokenizer:
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")

    tokenizer = tokenizers.Tokenizer.from_file(tokenizer_path)
    if hasattr(tokenizer, "no_padding"):
        tokenizer.no_padding()
    if hasattr(tokenizer, "no_truncation"):
        tokenizer.no_truncation()
    return tokenizer


def get_training_data(*args):
    data = []
    for arg in args:
        data.extend(list(arg))
    return data


def _validate_vocab_is_compact(tokenizer: tokenizers.Tokenizer) -> None:
    vocab = tokenizer.get_vocab()
    ids = set(vocab.values())
    expected = set(range(len(vocab)))
    missing = sorted(expected - ids)
    if missing:
        raise ValueError(
            "Tokenizer vocabulary contains non-contiguous ids: "
            + ", ".join(map(str, missing))
        )


def train_tokenizer(tokenizer, trainer, data):
    tokenizer.train_from_iterator(data, trainer)
    tokenizer.add_special_tokens(SPECIAL_ADDED_TOKENS)
    _validate_vocab_is_compact(tokenizer)

    temp_encoded = tokenizer.encode_batch(data)
    max_length = max(len(enc.ids) for enc in temp_encoded)

    # Set up padding with the correct token and ID
    tokenizer.enable_padding(
        length=max_length, pad_token="[PAD]", pad_id=tokenizer.token_to_id("[PAD]")
    )
    tokenizer.enable_truncation(max_length=max_length)
    return tokenizer


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


def add_special_tokens(questions):
    processed = []
    for question in questions:
        sections = [
            "[BOQ]",
            question,
            "[EOQ]",
        ]
        processed.append(" ".join(section for section in sections if section))

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


def save(X, y, num_samples, tokenizer, vocab_size, name, config):
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


def encode_questions(tokenizer, problems_processed_list):
    if hasattr(tokenizer, "no_padding"):
        tokenizer.no_padding()
    if hasattr(tokenizer, "no_truncation"):
        tokenizer.no_truncation()
    temp_encoded = tokenizer.encode_batch(problems_processed_list)
    max_length = max(len(enc.ids) for enc in temp_encoded)

    # First, encode without padding to get raw sequences
    tokenizer.no_padding()

    # Then pad all to the same length for consistent tensor shapes
    tokenizer.enable_padding(
        length=max_length,
        pad_token="[PAD]",
        pad_id=tokenizer.token_to_id("[PAD]"),
    )
    tokenizer.enable_truncation(max_length=max_length)

    # Re-encode with padding enabled
    problems_encoded = tokenizer.encode_batch(problems_processed_list)
    X = encoded_to_numpy(problems_encoded)

    return X


def encode_answers(tokenizer, answers_processed_list, X):
    answers_encoded_raw = [
        tokenizer.encode(answer).ids for answer in answers_processed_list
    ]

    # Create labels tensor - initialize with pad tokens
    y = np.full_like(X, tokenizer.token_to_id("[PAD]"))

    # Process answers and align with input sequences
    for i, answer_ids in enumerate(answers_encoded_raw):
        y[i, : len(answer_ids)] = answer_ids

    return y


def main(config):
    try:
        train_df, test_df = read_data(config)
        df = pd.concat([train_df, test_df])
        questions = df["question"]
        answers = df["answer"]
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    try:
        if config.tokenizer_path:
            tokenizer = load_tokenizer_from_path(config.tokenizer_path)
        else:
            tokenizer, trainer = get_tokenizer()
            data = get_training_data(
                questions,
                answers,
            )
            tokenizer = train_tokenizer(tokenizer, trainer, data)

        # Get the max length for consistent tensor shapes
        problems_processed_list = add_special_tokens(
            questions,
        )

        answers_processed_list = answers.apply(
            lambda row: row.split("####")[-1].strip()
        ).to_list()

    except Exception as e:
        print(f"Error during data processing: {e}")
        traceback.print_exc()
        return

    try:
        X_train = encode_questions(tokenizer, train_df["question"])
        y_train = encode_answers(
            tokenizer,
            train_df["answer"].apply(lambda row: row.split("####")[-1].strip()),
            X_train,
        )
        X_test = encode_questions(tokenizer, test_df["question"])
        y_test = encode_answers(
            tokenizer,
            test_df["answer"].apply(lambda row: row.split("####")[-1].strip()),
            X_test,
        )

        save(
            X_train,
            y_train,
            len(X_train),
            tokenizer,
            tokenizer.get_vocab_size(),
            "train",
            config,
        )
        save(
            X_test,
            y_test,
            len(X_test),
            tokenizer,
            tokenizer.get_vocab_size(),
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
