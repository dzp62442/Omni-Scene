"""Target-only all_6 metrics, explicit distributed padding and coverage checks."""

import csv
import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset


class EvaluationShard(Dataset):
    """Sequential global batches, with labeled padding instead of implicit repeats.

    Only evaluation uses this wrapper. Each rank gets the same number of batches;
    padded items never write artifacts or contribute to the final average.
    """

    def __init__(self, dataset, count, batch_size=1, rank=0, world_size=1):
        if not 0 < count <= len(dataset) or batch_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid evaluation shard parameters")
        self.dataset = dataset
        self.count = count
        stride = batch_size * world_size
        self.positions = [start + rank * batch_size + b
                          for start in range(0, count, stride) for b in range(batch_size)]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, item):
        position = self.positions[item]
        index = position % self.count
        sample = dict(self.dataset[index])
        sample["_evaluation_index"] = index
        sample["_evaluation_padding"] = position >= self.count
        return sample


def batch_metrics(preds, gts, tokens, indices, padding, compute_pcc=False, metrics=None):
    if metrics is None:
        # Original implementations, unchanged; LPIPS is initialized only at runtime.
        from tools.metrics import compute_psnr, compute_ssim, compute_lpips, compute_pcc as pcc
        metrics = {"psnr": compute_psnr, "ssim": compute_ssim, "lpips": compute_lpips, "pcc": pcc}
    predicted, target = preds["img"], gts["img"]
    if predicted.ndim != 5 or predicted.shape != target.shape or predicted.shape[1:3] != (6, 3):
        raise ValueError("Expected matching [B, 6, 3, H, W] RGB predictions and targets")
    batch_size = predicted.shape[0]
    if len(tokens) != batch_size or len(indices) != batch_size or len(padding) != batch_size:
        raise ValueError("Token/index/padding count differs from prediction batch size")
    for label, tensor in (("prediction", predicted), ("target", target),
                          ("depth", preds["depth"]), ("gaussian", preds["gaussian"])):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite {label}")
    if preds["depth"].shape != (batch_size, 6, *predicted.shape[-2:]):
        raise ValueError("Expected [B, 6, H, W] predicted depth")
    values = {name: metrics[name](target.flatten(0, 1), predicted.flatten(0, 1)).reshape(batch_size, 6)
              for name in ("psnr", "ssim", "lpips")}
    records = []
    for b, token in enumerate(tokens):
        record = {"bin_token": token, "index": int(indices[b]), "padding": bool(padding[b]),
                  "per_view": {name: value[b].detach().cpu().tolist() for name, value in values.items()},
                  "all_6": {name: float(value[b].mean()) for name, value in values.items()}}
        if compute_pcc:
            record["all_6"]["pcc"] = float(metrics["pcc"](gts["depth_m"][b], preds["depth"][b]))
        if not all(math.isfinite(v) for v in record["all_6"].values()):
            raise ValueError(f"Non-finite metrics for {token}")
        records.append(record)
    return records


def summarize(records, expected_tokens, full_count, max_samples=None):
    if not expected_tokens or len(set(expected_tokens)) != len(expected_tokens):
        raise ValueError("Expected tokens must be unique and nonempty")
    unique, padding_count = {}, 0
    metric_names = None
    for record in records:
        index = record["index"]
        if not 0 <= index < len(expected_tokens) or record["bin_token"] != expected_tokens[index]:
            raise ValueError(f"Unexpected evaluation sample: {record['bin_token']}")
        names = set(record["all_6"])
        if not {"psnr", "ssim", "lpips"} <= names or (metric_names is not None and names != metric_names):
            raise ValueError("Inconsistent or incomplete metric fields")
        metric_names = names
        if not all(math.isfinite(v) for v in record["all_6"].values()):
            raise ValueError(f"Non-finite metrics for {record['bin_token']}")
        if record["padding"]:
            padding_count += 1
            continue
        if index in unique:
            raise ValueError(f"Unexpected non-padding duplicate: {record['bin_token']}")
        unique[index] = record
    missing = [token for i, token in enumerate(expected_tokens) if i not in unique]
    if missing:
        raise ValueError(f"Missing {len(missing)} bins; first missing: {missing[0]}")
    ordered = [unique[i] for i in range(len(expected_tokens))]
    # An explicit --max-samples is always a diagnostic/limited run, even >= N.
    summary = {"status": "limited" if max_samples is not None else "complete",
               "complete_split": max_samples is None and len(ordered) == full_count,
               "metric_group": "all_6", "split_count": full_count,
               "selected_count": len(expected_tokens), "evaluated_count": len(ordered),
               "padding_records_discarded": padding_count, "max_samples": max_samples,
               "all_6": {name: math.fsum(r["all_6"][name] for r in ordered) / len(ordered)
                         for name in sorted(metric_names)}}
    return summary, ordered


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_results(directory, summary, records):
    directory = Path(directory)
    names = list(summary["all_6"])
    temporary = directory / "per_bin_metrics.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["bin_token", *names])
        writer.writeheader()
        for record in records:
            writer.writerow({"bin_token": record["bin_token"], **record["all_6"]})
    temporary.replace(directory / "per_bin_metrics.csv")
    write_json(directory / "per_view_metrics.json", records)
    write_json(directory / "evaluation_summary.json", summary)


def save_artifacts(directory, preds, gts, records, save_vis=False, save_ply=False):
    for b, record in enumerate(records):
        if record["padding"]:
            continue
        token = record["bin_token"]
        path = Path(directory) / "visualizations" / token
        # Each real index belongs to exactly one rank, including small tail batches.
        path.mkdir(parents=True, exist_ok=False)
        if save_ply:
            from .ply import save_ply as export_ply
            export_ply(preds["gaussian"][b], str(path / f"{token}.ply"), crop_range=None)
        if save_vis:
            import imageio.v2 as imageio
            import numpy as np
            from einops import rearrange
            from tools.visualization import depths_to_colors
            target = rearrange(gts["img"][b], "v c h w -> c h (v w)")
            predicted = rearrange(preds["img"][b], "v c h w -> c h (v w)")
            rgb = torch.cat([target, predicted], dim=1).permute(1, 2, 0)
            rgb = (rgb.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            depth = depths_to_colors(preds["depth"][b].clamp(0, 140))
            imageio.imwrite(path / f"{token}.png", np.concatenate([rgb, depth], axis=0))
