import re
from transformers import T5TokenizerFast


def build_tokenizer():
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

    # Preserve SQL operators as ordinary tokens.
    tokenizer.add_tokens(["<", "<="], special_tokens=False)

    return tokenizer


def decode_sql(tokenizer, generated_ids):
    queries = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    normalized_queries = []

    for query in queries:
        # Preserve quoted strings and identifiers.
        parts = re.split(
            r"""('(?:''|[^'])*'|"(?:""|[^"])*")""",
            query,
        )

        # Normalize spacing around added operators outside quotes.
        for index in range(0, len(parts), 2):
            parts[index] = re.sub(
                r"\s*(<=|<>|<)\s*",
                r" \1 ",
                parts[index],
            )

        normalized_queries.append("".join(parts).strip())

    return normalized_queries