from pathlib import Path
from transformers import AutoTokenizer

DATA_DIR = Path(__file__).resolve().parent / "data"

tokenizer = AutoTokenizer.from_pretrained(
    "google-t5/t5-small",
    use_fast=True,
)


def read_lines(path):
    # Remove line endings while preserving other whitespace.
    return path.read_text(encoding="utf-8").splitlines()


def compute_stats(texts):
    # Tokenize complete texts without padding or special tokens.
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
    )
    sequences = encoded["input_ids"]

    # Count token occurrences for the mean length.
    mean_length = sum(len(ids) for ids in sequences) / len(sequences)

    # Count distinct token IDs observed in this collection.
    vocabulary = {
        token_id
        for ids in sequences
        for token_id in ids
    }

    return mean_length, len(vocabulary)


for split in ("train", "dev"):
    questions = read_lines(DATA_DIR / f"{split}.nl")
    queries = read_lines(DATA_DIR / f"{split}.sql")

    # Verify that questions and SQL queries are paired by line.
    assert len(questions) == len(queries), "Mismatched example counts."
    assert questions, "The dataset is empty."
    assert all(text.strip() for text in questions), "Empty question found."
    assert all(text.strip() for text in queries), "Empty SQL query found."

    question_length, question_vocab = compute_stats(questions)
    sql_length, sql_vocab = compute_stats(queries)

    print(f"\nSplit: {split}")
    print(f"Number of examples: {len(questions)}")
    print(f"Mean sentence length: {question_length:.3f}")
    print(f"Mean SQL query length: {sql_length:.3f}")
    print(f"Vocabulary size (natural language): {question_vocab}")
    print(f"Vocabulary size (SQL): {sql_vocab}")