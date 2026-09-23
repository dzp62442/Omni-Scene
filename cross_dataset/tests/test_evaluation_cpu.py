from copy import deepcopy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
from torch.utils.data import Dataset, DataLoader

from cross_dataset.checkpoint import resolve_checkpoint, load_model_weights, checkpoint_record
from cross_dataset.evaluation import EvaluationShard, batch_metrics, summarize, write_results, save_artifacts
from cross_dataset.tests.cpu_only import CPUOnlyTest


class SyntheticDataset(Dataset):
    def __len__(self): return 5
    def __getitem__(self, index): return {"bin_token": f"bin_{index}", "value": index}


def synthetic_records(batch):
    target = torch.zeros(len(batch["value"]), 6, 3, 12, 12)
    predicted = target + (batch["value"].float() + 1)[:, None, None, None, None] / 10
    pred = {"img": predicted, "depth": torch.ones(len(target), 6, 12, 12),
            "gaussian": torch.ones(len(target), 2, 14)}
    gt = {"img": target, "depth_m": pred["depth"] * 2}
    from tools.metrics import compute_psnr, compute_ssim
    metrics = {"psnr": compute_psnr, "ssim": compute_ssim,
               "lpips": lambda a, b: (a - b).square().flatten(1).mean(1),
               "pcc": lambda a, b: torch.tensor(0.25)}
    return batch_metrics(pred, gt, batch["bin_token"], batch["_evaluation_index"],
                         batch["_evaluation_padding"], compute_pcc=True, metrics=metrics)


class EvaluationTests(CPUOnlyTest):
    def test_entrypoint_missing_weights_fails_before_model_or_accelerator(self):
        import sys
        from cross_dataset.evaluate import main, parser
        args = parser().parse_args(["--py-config", "configs/ZeroShot/omni_gs_nusc_to_ddad_r50_112x200.py"])
        with self.assertRaisesRegex(ValueError, "Explicit"):
            main(args)
        self.assertNotIn("builder.builder", sys.modules)

    def test_batch_size_world_size_padding_and_equal_bin_aggregation(self):
        expected = [f"bin_{i}" for i in range(5)]
        baseline = None
        for world in (1, 2, 3, 8):
            for batch_size in (1, 2, 4):
                records = []
                for rank in range(world):
                    shard = EvaluationShard(SyntheticDataset(), 5, batch_size, rank, world)
                    for batch in DataLoader(shard, batch_size=batch_size):
                        records.extend(synthetic_records(batch))
                summary, ordered = summarize(records, expected, 5)
                self.assertEqual(summary["status"], "complete")
                self.assertTrue(summary["complete_split"])
                self.assertEqual(len(ordered), 5)
                self.assertAlmostEqual(summary["all_6"]["lpips"], (1 + 4 + 9 + 16 + 25) / 500, places=6)
                self.assertEqual(summary["all_6"]["pcc"], 0.25)
                if baseline is None:
                    baseline = summary["all_6"]
                self.assertEqual(summary["all_6"], baseline)
        with tempfile.TemporaryDirectory() as tmp:
            write_results(tmp, summary, ordered)
            self.assertEqual(len((Path(tmp) / "per_bin_metrics.csv").read_text().splitlines()), 6)
            self.assertEqual(json.loads((Path(tmp) / "evaluation_summary.json").read_text())["evaluated_count"], 5)

    def test_coverage_duplicate_invalid_and_limited_status(self):
        batch = next(iter(DataLoader(EvaluationShard(SyntheticDataset(), 5, 5), batch_size=5)))
        records = synthetic_records(batch)
        expected = [f"bin_{i}" for i in range(5)]
        for bad, message in ((records[:-1], "Missing"), (records + records[:1], "duplicate")):
            with self.assertRaisesRegex(ValueError, message):
                summarize(bad, expected, 5)
        bad = deepcopy(records)
        bad[0]["all_6"]["psnr"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            summarize(bad, expected, 5)
        bad = deepcopy(records)
        bad[0]["bin_token"] = "foreign"
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            summarize(bad, expected, 5)
        summary, _ = summarize(records, expected, 5, max_samples=100)
        self.assertFalse(summary["complete_split"])
        self.assertEqual(summary["status"], "limited")

    def test_artifact_indexing_and_padding_no_write(self):
        records = [{"bin_token": "a", "padding": False}, {"bin_token": "b", "padding": False},
                   {"bin_token": "a", "padding": True}]
        gaussians = torch.stack([torch.full((2, 14), i) for i in (1., 2., 3.)])
        with tempfile.TemporaryDirectory() as tmp, patch("cross_dataset.ply.save_ply") as export:
            save_artifacts(tmp, {"gaussian": gaussians}, {}, records, save_ply=True)
            self.assertEqual(export.call_count, 2)
            self.assertTrue(torch.equal(export.call_args_list[0].args[0], gaussians[0]))
            self.assertTrue(torch.equal(export.call_args_list[1].args[0], gaussians[1]))
            self.assertEqual(sorted(p.name for p in (Path(tmp) / "visualizations").iterdir()), ["a", "b"])


class CheckpointTests(CPUOnlyTest):
    def test_strict_weights_shared_aliases_and_no_optimizer(self):
        from safetensors.torch import save_model

        class Tied(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.first = torch.nn.Linear(3, 2)
                self.second = torch.nn.Linear(3, 2)
                self.second.weight = self.first.weight

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            expected, actual = Tied(), Tied()
            save_model(expected, str(path / "model.safetensors"))
            (path / "optimizer.bin").write_bytes(b"this must never be opened")
            load_model_weights(actual, path)
            for key, value in expected.state_dict().items():
                torch.testing.assert_close(actual.state_dict()[key], value)
            with self.assertRaises(RuntimeError):
                load_model_weights(torch.nn.Linear(4, 4), path)
            torch.save(expected.state_dict(), path / "pytorch_model.bin")
            load_model_weights(actual, path / "pytorch_model.bin")
            declaration = {"dataset": "nuScenes", "config": None}
            record = checkpoint_record(path, declaration, [112, 200])
            self.assertEqual(record["source"]["verification"], "unverified_legacy_checkpoint")
            metadata = {"checkpoint_sha256": record["sha256"], "source_dataset": "nuScenes", "resolution": [112, 200]}
            (path / "source_metadata.json").write_text(json.dumps(metadata))
            self.assertEqual(checkpoint_record(path, declaration, [112, 200])["source"]["verification"], "hash_bound_metadata_provided")
            with self.assertRaisesRegex(ValueError, "resolution"):
                checkpoint_record(path, declaration, [224, 400])
        with self.assertRaisesRegex(ValueError, "Explicit"):
            resolve_checkpoint(None)
        with self.assertRaises(FileNotFoundError):
            resolve_checkpoint("missing-source-model.bin")
