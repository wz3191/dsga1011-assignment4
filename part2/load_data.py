import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output. 
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.split = split
        self.tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")

        # Use the same decoder start token during training and generation.
        self.start_token_id = self.tokenizer.convert_tokens_to_ids("<extra_id_0>")

        self.examples = self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        # Read questions without changing their text.
        with open(
            os.path.join(data_folder, f"{split}.nl"),
            encoding="utf-8",
        ) as file:
            questions = file.read().splitlines()

        # Test SQL targets are not publicly available.
        queries = None
        if split != "test":
            with open(
                os.path.join(data_folder, f"{split}.sql"),
                encoding="utf-8",
            ) as file:
                queries = file.read().splitlines()

            if len(questions) != len(queries):
                raise ValueError("Question and SQL counts do not match.")

        examples = []

        for index, question in enumerate(questions):
            # Include EOS, but do not pad or truncate individual examples.
            encoder_ids = torch.tensor(
                tokenizer.encode(
                    question,
                    add_special_tokens=True,
                    truncation=False,
                ),
                dtype=torch.long,
            )

            initial_decoder_inputs = torch.tensor(
                [self.start_token_id],
                dtype=torch.long,
            )

            if split == "test":
                examples.append(
                    (encoder_ids, initial_decoder_inputs)
                )
                continue

            # Tokenize the SQL and append EOS.
            target_ids = tokenizer.encode(
                queries[index],
                add_special_tokens=True,
                truncation=False,
            )

            # Shift the target sequence right to construct decoder inputs.
            decoder_inputs = torch.tensor(
                [self.start_token_id] + target_ids[:-1],
                dtype=torch.long,
            )
            decoder_targets = torch.tensor(
                target_ids,
                dtype=torch.long,
            )

            examples.append(
                (
                    encoder_ids,
                    decoder_inputs,
                    decoder_targets,
                    initial_decoder_inputs,
                )
            )

        return examples
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # Group the corresponding fields from all examples.
    encoder_sequences, decoder_sequences, target_sequences, starts = zip(
        *batch
    )

    # Pad encoder inputs to the longest question in this batch.
    encoder_ids = pad_sequence(
        encoder_sequences,
        batch_first=True,
        padding_value=PAD_IDX,
    )
    encoder_mask = (encoder_ids != PAD_IDX).long()

    # Pad decoder inputs and targets to the longest SQL in this batch.
    decoder_inputs = pad_sequence(
        decoder_sequences,
        batch_first=True,
        padding_value=PAD_IDX,
    )
    decoder_targets = pad_sequence(
        target_sequences,
        batch_first=True,
        padding_value=PAD_IDX,
    )

    # Each example supplies one decoder start token.
    initial_decoder_inputs = torch.stack(starts)

    return (
        encoder_ids,
        encoder_mask,
        decoder_inputs,
        decoder_targets,
        initial_decoder_inputs,
    )

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns: 
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    # Test examples contain only encoder inputs and a decoder start token.
    encoder_sequences, starts = zip(*batch)

    encoder_ids = pad_sequence(
        encoder_sequences,
        batch_first=True,
        padding_value=PAD_IDX,
    )
    encoder_mask = (encoder_ids != PAD_IDX).long()
    initial_decoder_inputs = torch.stack(starts)

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split):
    data_folder = 'data'
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size):
    train_loader = get_dataloader(batch_size, "train")
    dev_loader = get_dataloader(test_batch_size, "dev")
    test_loader = get_dataloader(test_batch_size, "test")
    
    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x