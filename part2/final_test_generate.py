"""Evaluate decoding choices without training or changing the official metrics."""
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys
import traceback


def generate(args):
    import torch
    from tqdm import tqdm
    from transformers import T5ForConditionalGeneration, T5TokenizerFast

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run the full job on CPU.")
    torch.manual_seed(42)
    torch.set_num_threads(4)
    # Use the original tokenizer, not the tokenizer-fix experiment.
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint)
    start_id = tokenizer.convert_tokens_to_ids("<extra_id_0>")
    assert model.config.decoder_start_token_id == start_id
    assert model.generation_config.decoder_start_token_id == start_id
    model.to(args.device).eval()
    questions = Path("data/test.nl").read_text().splitlines()
    if args.limit:
        questions = questions[:args.limit]
    predictions = []
    print(f"Checkpoint: {args.checkpoint}; beams: {args.beams}; examples: {len(questions)}", flush=True)
    with torch.inference_mode():
        for offset in tqdm(range(0, len(questions), args.batch_size), desc="Generating"):
            batch = questions[offset:offset + args.batch_size]
            inputs = tokenizer(batch, padding=True, truncation=False, return_tensors="pt")
            inputs = {k: v.to(args.device) for k, v in inputs.items()}
            starts = torch.full((len(batch), 1), start_id, dtype=torch.long, device=args.device)
            ids = model.generate(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
                decoder_input_ids=starts, do_sample=False, num_beams=args.beams,
                max_new_tokens=args.max_new_tokens,
            )
            # Match the original baseline's decoding and whitespace handling.
            predictions.extend(q.strip() for q in tokenizer.batch_decode(
                ids.cpu().tolist(), skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ))
    assert len(predictions) == len(questions)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(predictions) + "\n")
    print(f"Saved {len(predictions)} queries to {output}", flush=True)


def score_worker(sql_path, record_path, connection):
    try:
        # Import the course's unchanged scoring functions from the working directory.
        sys.path.insert(0, str(Path.cwd()))
        from utils import save_queries_and_records, compute_metrics
        queries = Path(sql_path).read_text().splitlines()
        expected = len(Path("data/dev.sql").read_text().splitlines())
        assert len(queries) == expected, (len(queries), expected)
        Path(record_path).parent.mkdir(parents=True, exist_ok=True)
        save_queries_and_records(queries, sql_path, record_path)
        sql_em, record_em, f1, errors = compute_metrics(
            "data/dev.sql", sql_path, "records/ground_truth_dev.pkl", record_path,
        )
        result = dict(sql_file=sql_path, examples=expected, sql_em=float(sql_em),
                      record_em=float(record_em), record_f1=float(f1),
                      execution_errors=sum(bool(e) for e in errors),
                      timeouts=sum("timed out" in str(e).lower() for e in errors))
        # Files are closed before notifying the parent that scoring is complete.
        connection.send((True, result))
    except BaseException:
        connection.send((False, traceback.format_exc()))
    finally:
        connection.close()


def score(args):
    # Work on copies so the original experiment outputs remain unchanged.
    destination = Path("results/beam_ablation")
    destination.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    all_results = []
    for source in args.sql:
        source = Path(source)
        target = destination / source.name
        if source.resolve() == target.resolve():
            raise ValueError("Use source SQL outside results/beam_ablation.")
        target.write_bytes(source.read_bytes())
        records = Path("records/beam_ablation") / (source.stem + ".pkl")
        receiver, sender = ctx.Pipe(duplex=False)
        process = ctx.Process(target=score_worker, args=(str(target), str(records), sender))
        process.start()
        sender.close()
        try:
            # This is a process watchdog, not a replacement for the course's SQL timeout.
            if not receiver.poll(300):
                raise TimeoutError("Scoring did not finish in 300 seconds; no score accepted.")
            success, result = receiver.recv()
            if not success:
                raise RuntimeError(result)
            print(json.dumps(result, indent=2), flush=True)
            all_results.append(result)
        finally:
            receiver.close()
            process.join(timeout=2)
            # Stop lingering SQL threads only after receiving the completed results.
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
    (destination / "metrics.json").write_text(json.dumps(all_results, indent=2) + "\n")
    print("Scoring completed; metrics saved to results/beam_ablation/metrics.json", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--checkpoint", default="checkpoints/ft_experiments/baseline/best")
    gen.add_argument("--beams", type=int, default=4)
    gen.add_argument("--batch_size", type=int, default=8)
    gen.add_argument("--max_new_tokens", type=int, default=512)
    gen.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    gen.add_argument("--limit", type=int, default=0)
    gen.add_argument("--output", default="results/t5_ft_baseline_beam4_dev.sql")
    scorer = sub.add_parser("score")
    scorer.add_argument("--sql", nargs="+", default=[
        "results/t5_ft_baseline_dev.sql", "results/t5_ft_baseline_beam4_dev.sql",
    ])
    args = parser.parse_args()
    generate(args) if args.mode == "generate" else score(args)


if __name__ == "__main__":
    main()
