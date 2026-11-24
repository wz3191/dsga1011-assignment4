# DS-GA 1011 Assignment 4

Wenshan Zhang (wz3191)

Based on the [course starter code](https://github.com/gregdurrett/nyu-dsga1011-a4).

- `part1/`: BERT training, typo transformation, and data augmentation.
- `part2/`: T5 training, evaluation, and test generation.
- `experiments/tokenizer_fix/`: code for the tokenizer experiment reported in Q6.

Use the setup instructions in each part's README and obtain the data from the course repository.

The final model uses learning rate `1e-4` and beam search with four beams.
[Model checkpoint](https://drive.google.com/drive/folders/1FjvmzQhq-PaMB6C09AFjqRof1zkbUR1v).
Download all three checkpoint files into one directory. Use the original `google-t5/t5-small` tokenizer.

From `part2/`, generate test outputs with:

```bash
python final_test_generate.py generate --checkpoint /path/to/model \
  --beams 4 --batch_size 8 --max_new_tokens 512 \
  --output results/t5_ft_experiment_test.sql
python final_test_records.py
```

The report and the six required output files are submitted separately on Gradescope.
