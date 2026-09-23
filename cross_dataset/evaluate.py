"""Independent PandaSet/DDAD evaluation; run with python -m cross_dataset.evaluate."""

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import subprocess
import traceback


def parser():
    result = argparse.ArgumentParser(description="OmniScene PandaSet/DDAD all_6 evaluation")
    result.add_argument("--py-config", required=True)
    result.add_argument("--load-from")
    result.add_argument("--output-dir")
    result.add_argument("--split", choices=("train", "val", "test"))
    result.add_argument("--max-samples", type=int, help="Explicit diagnostic limit; never labeled a complete split")
    result.add_argument("--source-dataset")
    result.add_argument("--source-config")
    result.add_argument("--source-metadata", help="Optional hash-bound source_metadata.json")
    return result


def code_record():
    from .datasets.common import PROJECT_ROOT
    from .checkpoint import sha256_file
    paths = []
    for directory in ("cross_dataset", "configs/PandaSet", "configs/DDAD", "configs/ZeroShot"):
        paths += sorted((PROJECT_ROOT / directory).rglob("*.py"))
    return {"frozen_baseline": "a8318b4da99879c625a60154d111a510e5ac5a91",
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True),
            "new_file_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256_file(p) for p in paths}}


def main(args):
    # These imports do not construct the original model or initialize CUDA.
    from .configuration import load_config, source_declaration, output_path
    from .checkpoint import resolve_checkpoint, checkpoint_record, load_model_weights
    from .datasets import build_dataset, DATASETS
    from .datasets.common import CAMERA_TYPES, PROTOCOL, project_path
    from .evaluation import EvaluationShard, batch_metrics, summarize, write_json, write_results, save_artifacts

    cfg = load_config(args.py_config)
    declaration = source_declaration(cfg, args.source_dataset, args.source_config)
    requested_weights = args.load_from or cfg.get("load_from")
    weights = resolve_checkpoint(requested_weights)
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    split = args.split or cfg.get("split", "test")
    dataset = build_dataset(cfg.dataset_params, split)
    count = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    expected_tokens = dataset.bin_tokens[:count]
    destination = output_path(args.output_dir or cfg.output_dir,
                              protected=(project_path(requested_weights), dataset.processed_root))
    cfg.output_dir = str(destination)
    cfg.load_from = str(weights)
    cfg.split = split
    cfg.source_dataset = declaration["dataset"]
    cfg.source_config = declaration["config"]

    import torch
    from accelerate import Accelerator
    from accelerate.utils import gather_object, send_to_device, set_seed
    from torch.utils.data import DataLoader

    accelerator = Accelerator(mixed_precision=cfg.mixed_precision)
    errors = []
    manifest = None
    if accelerator.is_main_process:
        try:
            # Never mix this run with an older evaluation or a training directory.
            destination.mkdir(parents=True, exist_ok=True)
            if any(destination.iterdir()):
                raise ValueError(f"Evaluation requires an empty output directory: {destination}")
            cfg.dump(str(destination / "resolved_config.py"))
            checkpoint = checkpoint_record(weights, declaration, cfg.resolution, args.source_metadata)
            manifest = {
                "status": "running", "complete_split": False,
                "zero_shot": cfg.get("zero_shot", False), "checkpoint": checkpoint,
                "target_dataset": DATASETS[cfg.dataset_params.dataset_name].dataset_name,
                "split": split, "split_count": len(dataset), "selected_count": count,
                "expected_tokens": expected_tokens, "actual_tokens": [],
                "max_samples": args.max_samples, "processed_root": str(dataset.processed_root.resolve()),
                "resolution": list(cfg.resolution), "camera_order": list(CAMERA_TYPES),
                "input_views": 6, "output_views": 6, "protocol": PROTOCOL,
                "metric_group": "all_6", "aggregation": "view mean per bin, then equal-weight unique-bin mean",
                "compute_pcc": cfg.eval_args.compute_pcc, "pcc_reference": "Metric3D-v2 metric depth",
                "world_size": accelerator.num_processes, "code": code_record(),
                "resolved_config_sha256": hashlib.sha256((destination / "resolved_config.py").read_bytes()).hexdigest(),
            }
            write_json(destination / "evaluation_manifest.json", manifest)
        except Exception:
            errors.append(traceback.format_exc())
    errors = gather_object(errors)
    if errors:
        raise RuntimeError("Evaluation setup failed:\n" + "\n".join(errors))

    local_records, local_error = [], None
    try:
        if cfg.seed is not None:
            set_seed(cfg.seed + accelerator.local_process_index)
        # Intentionally unchanged builder and model, including their CUDA allocations.
        from builder import builder as model_builder
        model = model_builder.build(deepcopy(cfg.model)).to(accelerator.device)
        load_model_weights(model, weights)
        # Inference needs no DDP gradient synchronization. Sharding is explicit below
        # so padding is distinguishable from an unexpected duplicate token.
        model = accelerator.prepare_model(model, evaluation_mode=True)
        model.eval()
        shard = EvaluationShard(dataset, count, cfg.dataset_params.batch_size_test,
                                accelerator.process_index, accelerator.num_processes)
        dataloader = DataLoader(shard, batch_size=cfg.dataset_params.batch_size_test,
                                num_workers=cfg.dataset_params.num_workers_test, shuffle=False)
        with torch.no_grad():
            for batch_index, batch in enumerate(dataloader):
                batch = send_to_device(batch, accelerator.device)
                preds, gts, tokens = accelerator.unwrap_model(model).forward_test(batch)
                records = batch_metrics(preds, gts, tokens, batch["_evaluation_index"],
                                        batch["_evaluation_padding"], cfg.eval_args.compute_pcc)
                if cfg.eval_args.save_vis or cfg.eval_args.save_ply:
                    save_artifacts(destination, preds, gts, records, cfg.eval_args.save_vis, cfg.eval_args.save_ply)
                local_records.extend(records)
                if batch_index % cfg.print_freq == 0:
                    accelerator.print(f"all_6 {split}: batch {batch_index + 1}/{len(dataloader)}")
    except Exception:
        local_error = traceback.format_exc()
    results = gather_object([{"rank": accelerator.process_index, "records": local_records, "error": local_error}])
    errors = [r["error"] for r in results if r["error"]]
    finalization_errors = []
    if accelerator.is_main_process:
        try:
            all_records = [record for result in results for record in result["records"]]
            manifest["actual_tokens"] = sorted({r["bin_token"] for r in all_records if not r["padding"]})
            if errors:
                raise RuntimeError("\n".join(errors))
            summary, ordered = summarize(all_records, expected_tokens, len(dataset), args.max_samples)
            write_results(destination, summary, ordered)
            manifest.update({k: summary[k] for k in ("status", "complete_split", "evaluated_count", "padding_records_discarded")})
            manifest["checkpoint"]["strict_model_load"] = "passed"
            write_json(destination / "evaluation_manifest.json", manifest)
            accelerator.print(summary)
        except Exception:
            error = traceback.format_exc()
            manifest.update(status="failed", complete_split=False, error=error)
            write_json(destination / "evaluation_manifest.json", manifest)
            write_json(destination / "evaluation_summary.json", {"status": "failed", "complete_split": False, "error": error})
            finalization_errors.append(error)
    finalization_errors = gather_object(finalization_errors)
    if finalization_errors:
        raise RuntimeError("Evaluation failed:\n" + "\n".join(finalization_errors))
    accelerator.end_training()


if __name__ == "__main__":
    main(parser().parse_args())
