import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    import re

    # Define adjacent letter keys within each QWERTY keyboard row.
    keyboard_rows = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
    neighbors = {}

    for row in keyboard_rows:
        for i, char in enumerate(row):
            candidates = ""
            if i > 0:
                candidates += row[i - 1]
            if i < len(row) - 1:
                candidates += row[i + 1]
            neighbors[char] = candidates

    def introduce_typo(match):
        word = match.group(0)

        # Select words of at least four letters with probability 0.35.
        if len(word) < 4 or random.random() >= 0.35:
            return word

        # Replace one randomly selected letter with an adjacent key.
        index = random.randrange(len(word))
        original_char = word[index]
        replacement = random.choice(neighbors[original_char.lower()])

        # Preserve the case of the original letter.
        if original_char.isupper():
            replacement = replacement.upper()

        return word[:index] + replacement + word[index + 1:]

    # Update the text while preserving punctuation, spacing, and labels.
    example["text"] = re.sub(r"[A-Za-z]+", introduce_typo, example["text"])

    ##### YOUR CODE ENDS HERE ######

    return example
