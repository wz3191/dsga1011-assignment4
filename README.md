# DS-GA 1011 Assignment 4

Wenshan Zhang (wz3191). Work in progress: final test outputs and report links will be added.

Based on the course starter repository: https://github.com/gregdurrett/nyu-dsga1011-a4

## Contents

- `part1/`: BERT sentiment classification, keyboard typos, and data augmentation.
- `part2/`: T5 fine-tuning with the original tokenizer, evaluation, and experiment launchers.
- `experiments/tokenizer_fix/`: the separate tokenizer-fix variant, including its decoding helper.

The Part 2 main source is restored from the successful learning-rate run, not the tokenizer-fix variant. Model weights, datasets, generated outputs, and backup archives are not committed. Obtain course data and ground-truth records from the starter repository and place them in the corresponding part directories. Preserve the original data file ordering.

## Environment

Use separate Python environments for Part 1 and Part 2. Each part contains its requirements and the original setup README. Part 2 was run with Python 3.11 and Transformers 4.40.0.

## Part 2 training

Run from `part2/` with the course data and ground-truth records present:

```bash
python -u train_t5.py --finetune --experiment_name lr1e4 \
  --learning_rate 1e-4 --weight_decay 0 \
  --batch_size 8 --test_batch_size 8 --scheduler_type linear \
  --num_warmup_epochs 0 --max_n_epochs 20 --patience_epochs 5
```

Training uses seed 42 and selects the checkpoint with the highest development Record F1. This direct entry point retains the original SQL execution helper. The `lr_experiment.py` wrapper isolates SQL execution in a subprocess to clean up lingering queries after the original helper returns; metric formulas and its 120-second timeout remain unchanged.

The wrapper expects a local `baseline_backup.zip` containing the original `train_t5.py`, `load_data.py`, `t5_utils.py`, and `utils.py` at the archive root. These are the four main source files in `part2/`; the full model backup is not needed to reconstruct the source archive. The wrapper creates a separate `runs/lr1e4_JOBID/` directory.

Slurm scripts contain NYU account and filesystem paths for this run. Adjust those paths before using another checkout. Submit from an initialized environment with inherited environment-retrieval flags removed:

```bash
env -u SLURM_GET_USER_ENV -u SBATCH_GET_USER_ENV -u SBATCH_EXPORT -u SLURM_EXPORT_ENV \
  sbatch --export=ALL lr_experiment.slurm
```

## Decoding comparison

`beam_eval.py generate` loads a saved checkpoint without training, uses the original T5 tokenizer, and generates development SQL. Use `--checkpoint` and `--output` to identify the model and avoid overwriting another experiment. `--beams 4` is the beam-search configuration; `--beams 1` is greedy. The maximum number of new tokens is 512 and the batch size is 8.

`beam_eval.py score --sql PATH1 PATH2` scores development predictions using the course metrics in isolated worker processes. It requires `data/dev.sql`, the database, and `records/ground_truth_dev.pkl`. This script currently handles development evaluation only; final test inference remains to be finalized.

## Verified development results

Scores are percentages. All experiments use one random seed.

| Configuration | SQL EM | Record EM | Record F1 |
|---|---:|---:|---:|
| Original baseline, LR 5e-5, greedy | 37.12 | 51.93 | 60.71 |
| Tokenizer fix, LR 5e-5, greedy | 38.41 | 52.15 | 60.68 |
| Original baseline, LR 5e-5, beam=4 | 38.41 | 53.22 | 62.55 |
| Original tokenizer, LR 1e-4, greedy | 54.94 | 69.31 | 74.46 |

The LR=1e-4 plus beam=4 evaluation is pending. Final selection will use development performance, not test-set tuning.

## Remaining submission materials

- Final configuration and its test SQL/record files.
- Final report and error analysis for the selected configuration.
- Google Drive link to the selected model checkpoint.
- Repository access for course staff if this repository is private.
