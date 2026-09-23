"""Small CPU oracle export. Run from the documented SVF-GS checkout.

Only reference loaders and metrics are imported; no preprocessing or full model.
Production modules never import SVF-GS. Results are restricted to /tmp.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(Path("/tmp")):
        raise ValueError("Development oracle artifacts must stay in /tmp")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if commit != "af39b31d984ba128020282764265fa38d31ce767":
        raise ValueError("Run from the documented SVF-GS reference checkout")
    output.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(2)
    with patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("GPU forbidden")), \
         patch("torch.hub.download_url_to_file", side_effect=AssertionError("Offline oracle: missing local weights")):
        from configs.build_config import build_zero_shot_config, build_eval_mask_config
        from data.ddad_dataset import DDADDataset
        from data.pandaset_dataset import PandaSetDataset
        from tools.metrics import compute_image_metrics
        from tools.ablation_metrics import render_records
        from tools.ablation_results import summarize_records
        mask_cfg = build_eval_mask_config()
        metadata = dict(reference_commit=commit, atol=1e-6, rtol=1e-5,
                        packages={name: importlib.metadata.version(name) for name in
                                  ("torch", "torchvision", "lpips", "numpy", "scipy", "scikit-image", "torchmetrics")},
                        data=[], metrics=[], cuda_initialized=False)
        for name, cls in (("pandaset", PandaSetDataset), ("ddad", DDADDataset)):
            cfg = SimpleNamespace(temporal_cfg=build_zero_shot_config(name), only_input=False,
                                  eval_mask_cfg=mask_cfg if name == "ddad" else None)
            for resolution in ((112, 200), (224, 400)):
                dataset = cls(cfg, list(resolution), "test")
                metadata["data"].append(dict(dataset=name, resolution=list(resolution), tokens=dataset.bin_tokens,
                                             val_tokens=cls(cfg, list(resolution), "val").bin_tokens))
                for index in (0, len(dataset)//2, len(dataset)-1):
                    torch.save(dataset[index], output / f"{name}_{resolution[0]}_{index}.pt")
            print(f"Exported {name}: first/middle/last, two resolutions", flush=True)
        weights = Path(torch.hub.get_dir()) / "checkpoints/vgg16-397923af.pth"
        metadata["vgg_sha256"] = hashlib.sha256(weights.read_bytes()).hexdigest()
        for height, width in ((112, 200), (224, 400)):
            torch.manual_seed(height)
            rows, packs = [], []
            for index in range(2):
                gt = torch.rand(1, 18, 3, height, width)
                pred = (gt + torch.randn_like(gt)*.07).clamp(0, 1)
                depth = torch.rand(1, 18, height, width)*80+1
                pred_depth = depth * 1.1 + torch.randn_like(depth)
                mask = torch.ones(1, 18, height, width, dtype=torch.bool)
                for view in range(2, 12):
                    mask[:, view, height - (view+index+1)*height//20:] = False
                data = dict(bin_token=[f"bin_{index}"], scene_id=[f"scene_{index}"], output_imgs=gt,
                            output_depths=depth, output_eval_masks=mask, n_base=torch.tensor([10]),
                            n_layer=torch.tensor([0]), n_total=torch.tensor([10]))
                render = dict(image=pred, depth=pred_depth.unsqueeze(2))
                image_metrics = {k: v.reshape(1, 18) for k, v in compute_image_metrics(
                    gt.flatten(0, 1), pred.flatten(0, 1), mask.flatten(0, 1), mask_cfg).items()}
                records = render_records(data, render, "final", image_metrics, mask_cfg, "fixture-mask-sha")
                rows.extend(records)
                packs.append(dict(gt=gt, pred=pred, depth=depth, pred_depth=pred_depth, mask=mask,
                                  image_metrics=image_metrics, records=records))
            filename = f"metrics_{height}.pt"
            torch.save(dict(packs=packs, mask_cfg=mask_cfg,
                            summary=summarize_records(rows, ["bin_0", "bin_1"])), output / filename)
            metadata["metrics"].append(filename)
            print(f"Exported real VGG metrics: 2 bins x 18 views at {height}x{width}", flush=True)
        if torch.cuda.is_initialized():
            raise AssertionError("Unexpected CUDA initialization")
        (output/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")


if __name__ == "__main__":
    main()
