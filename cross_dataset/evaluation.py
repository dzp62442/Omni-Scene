"""Target-only temporal18 metrics, explicit distributed padding and coverage checks."""

import csv
import json
import math
from pathlib import Path

import numpy as np
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


def view_groups(views):
    if views == 18:
        return {"all_18": slice(0, 18), "novel_12": slice(0, 12), "input_6": slice(12, 18)}
    if views == 6:
        return {"all_6": slice(0, 6)}
    raise ValueError("Only 18 target views or explicit 6-view diagnostics are supported")


def batch_metrics(preds, gts, tokens, indices, padding, compute_pcc=False, metrics=None,
                  expected_views=18, eval_mask=None, mask_cfg=None, identity=None,
                  scene_ids=None, view_metadata=None):
    groups = view_groups(expected_views)
    predicted, target = preds["img"], gts["img"]
    if predicted.ndim != 5 or predicted.shape != target.shape or predicted.shape[1:3] != (expected_views, 3):
        raise ValueError(f"Expected matching [B, {expected_views}, 3, H, W] RGB predictions and targets")
    batch_size, views, _, height, width = predicted.shape
    if len(tokens) != batch_size or len(indices) != batch_size or len(padding) != batch_size:
        raise ValueError("Token/index/padding count differs from prediction batch size")
    for label, tensor in (("prediction", predicted), ("target", target),
                          ("depth", preds["depth"]), ("reference depth", gts["depth_m"]),
                          ("gaussian", preds["gaussian"])):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Non-finite {label}")
    depth_shape = (batch_size, views, height, width)
    if preds["depth"].shape != depth_shape or gts["depth_m"].shape != depth_shape:
        raise ValueError("Expected B/V/H/W-matched predicted and reference depth")
    if (eval_mask is None) != (mask_cfg is None):
        raise ValueError("Evaluation mask and configuration must be provided together")
    identity = dict(identity or {})
    identity.setdefault("pixel_protocol", "full_image")
    identity.setdefault("mask_manifest_sha256", "")
    if eval_mask is not None:
        if (eval_mask.shape != depth_shape or eval_mask.dtype != torch.bool or eval_mask.device != predicted.device
                or not eval_mask[:, -6:].all()):
            raise ValueError("Expected boolean target masks with six all-valid input views")
        if identity["pixel_protocol"] != mask_cfg["pixel_protocol"] or not identity["mask_manifest_sha256"]:
            raise ValueError("Masked evaluation requires matching pixel protocol and manifest identity")
    elif identity["pixel_protocol"] != "full_image" or identity["mask_manifest_sha256"]:
        raise ValueError("Full-image evaluation cannot carry mask identity")
    flat_gt, flat_pred = target.flatten(0, 1), predicted.flatten(0, 1)
    if metrics is None:
        from .metrics import compute_image_metrics, compute_eval_pcc
        flat_mask = None if eval_mask is None else eval_mask.flatten(0, 1)
        values = compute_image_metrics(flat_gt, flat_pred, flat_mask, mask_cfg)
        pcc = compute_eval_pcc
    else:
        if eval_mask is not None:
            raise ValueError("Masked metrics must use the reference-compatible metric implementation")
        values = {name: metrics[name](flat_gt, flat_pred) for name in ("psnr", "ssim", "lpips")}
        pcc = lambda gt, pred, mask: metrics["pcc"](gt, pred)
    values = {name: value.reshape(batch_size, views) for name, value in values.items()}
    records = []
    for b, token in enumerate(tokens):
        per_view = {name: value[b].detach().cpu().tolist() for name, value in values.items()}
        per_view["valid_pixels"] = ([height * width] * views if eval_mask is None else
                                    eval_mask[b].sum((1, 2)).cpu().tolist())
        if view_metadata is not None:
            per_view.update({key: [v[b] for v in value] for key, value in view_metadata.items()})
        for group, selection in groups.items():
            record = dict(bin_token=str(token), scene_id="" if scene_ids is None else str(scene_ids[b]),
                          index=int(indices[b]), padding=bool(padding[b]), stage="final", view_group=group,
                          view_count=selection.stop-selection.start, per_view=per_view, **identity)
            record.update({name: value[b, selection].double().mean().item() for name, value in values.items()})
            if compute_pcc:
                record["pcc"] = pcc(gts["depth_m"][b, selection].contiguous(),
                                    preds["depth"][b, selection].contiguous(),
                                    None if eval_mask is None else eval_mask[b, selection]).item()
            if not all(math.isfinite(record[name]) for name in ("psnr", "ssim", "lpips", *(["pcc"] if compute_pcc else []))):
                raise ValueError(f"Non-finite metrics for {token}/{group}")
            records.append(record)
    return records


def summarize(records, expected_tokens, full_count, max_samples=None, expected_views=18):
    groups = view_groups(expected_views)
    if not expected_tokens or len(set(expected_tokens)) != len(expected_tokens):
        raise ValueError("Expected tokens must be unique and nonempty")
    if full_count < len(expected_tokens) or (max_samples is None and full_count != len(expected_tokens)):
        raise ValueError("Selected coverage is inconsistent with full split")
    unique, protocol_identity = {}, None
    metric_names = None
    identity_fields = ("pixel_protocol", "mask_manifest_sha256", "selection_sha256", "manifest_sha256")
    for record in records:
        index, group = record["index"], record["view_group"]
        if not 0 <= index < len(expected_tokens) or record["bin_token"] != expected_tokens[index]:
            raise ValueError(f"Unexpected evaluation sample: {record['bin_token']}")
        if record["stage"] != "final" or group not in groups or record["view_count"] != groups[group].stop-groups[group].start:
            raise ValueError("Unexpected stage/view group/view count")
        current_identity = tuple(record.get(k, "") for k in identity_fields)
        pixel_protocol, mask_sha = current_identity[:2]
        if pixel_protocol not in ("full_image", "ddad_ego_novel12_v1") or ((pixel_protocol == "full_image") != (mask_sha == "")):
            raise ValueError("Invalid evaluation pixel protocol or mask identity")
        if protocol_identity is not None and protocol_identity != current_identity:
            raise ValueError("Mixed selection, manifest, pixel protocols or mask identities")
        protocol_identity = current_identity
        names = {key for key in ("psnr", "ssim", "lpips", "pcc") if key in record}
        if not {"psnr", "ssim", "lpips"} <= names or (metric_names is not None and names != metric_names):
            raise ValueError("Inconsistent or incomplete metric fields")
        metric_names = names
        if not all(math.isfinite(record[name]) for name in names):
            raise ValueError(f"Non-finite metrics for {record['bin_token']}")
        if record["padding"]:
            continue
        key = (index, group)
        if key in unique:
            raise ValueError(f"Unexpected non-padding duplicate: {record['bin_token']}/{group}")
        unique[key] = record
    ordered = []
    for i, token in enumerate(expected_tokens):
        for group in groups:
            if (i, group) not in unique:
                raise ValueError(f"Missing bin/group: {token}/{group}")
            ordered.append(unique[i, group])
    identity = dict(zip(identity_fields, protocol_identity))
    summary = dict(status="limited" if max_samples is not None else "complete",
                   complete_split=max_samples is None, split_count=full_count,
                   selected_count=len(expected_tokens), evaluated_count=len(expected_tokens),
                   padding_records_discarded=sum(r["padding"] for r in records),
                   max_samples=max_samples, metric_groups=[f"final/{group}" for group in groups], **identity)
    for group in groups:
        summary[f"final/{group}"] = dict(num_bins=len(expected_tokens), **identity,
            **{name: float(np.asarray([unique[i, group][name] for i in range(len(expected_tokens))], dtype=np.float64).mean())
               for name in sorted(metric_names)})
    return summary, ordered


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_results(directory, summary, records):
    directory = Path(directory)
    temporary = directory / "per_bin_metrics.csv.tmp"
    keys = [key for key in records[0] if key not in ("per_view", "padding", "index")]
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in keys})
    temporary.replace(directory / "per_bin_metrics.csv")
    bins = {record["bin_token"]: dict(bin_token=record["bin_token"], scene_id=record["scene_id"],
                pixel_protocol=record["pixel_protocol"], mask_manifest_sha256=record["mask_manifest_sha256"],
                **record["per_view"]) for record in records}
    write_json(directory / "per_view_metrics.json", list(bins.values()))
    write_json(directory / "evaluation_summary.json", summary)


def save_artifacts(directory, preds, gts, records, save_vis=False, save_ply=False):
    # Group rows share one bin. Iterate once per local batch item, including padding.
    bins = list({(r["index"], r["padding"]): r for r in records}.values())
    for b, record in enumerate(bins):
        if record["padding"]:
            continue
        token = record["bin_token"]
        path = Path(directory) / "visualizations" / token
        path.mkdir(parents=True, exist_ok=False)
        if save_ply:
            from .ply import save_ply as export_ply
            export_ply(preds["gaussian"][b], str(path / f"{token}.ply"), crop_range=None)
        if save_vis:
            import imageio.v2 as imageio
            from einops import rearrange
            from tools.visualization import depths_to_colors
            views = preds["img"].shape[1]
            layouts = {"input": list(range(6))} if views == 6 else {
                "before": list(range(0, 12, 2)), "after": list(range(1, 12, 2)), "input": list(range(12, 18))}
            for label, selection in layouts.items():
                target = rearrange(gts["img"][b, selection], "v c h w -> c h (v w)")
                predicted = rearrange(preds["img"][b, selection], "v c h w -> c h (v w)")
                rgb = torch.cat([target, predicted], dim=1).permute(1, 2, 0)
                rgb = (rgb.detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
                depth = depths_to_colors(preds["depth"][b, selection].clamp(0, 140))
                imageio.imwrite(path / f"{token}_{label}.png", np.concatenate([rgb, depth], axis=0))
