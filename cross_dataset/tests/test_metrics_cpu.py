"""Masked metric semantics and external SVF-GS golden comparisons, CPU only."""

from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import torch
from skimage.metrics import structural_similarity

from cross_dataset import metrics
from cross_dataset.datasets.assets import build_eval_mask_config
from cross_dataset.evaluation import batch_metrics, summarize, write_results
from cross_dataset.tests.cpu_only import CPUOnlyTest
from cross_dataset.tests.test_data_cpu import reference_fixture


class MetricTests(CPUOnlyTest):
    def test_svf_gs_golden_views_bins_and_groups(self):
        root = reference_fixture()
        if not (root / "metadata.json").is_file():
            self.skipTest("Export fixed SVF-GS CPU metric reference under /tmp first")
        metadata = json.loads((root / "metadata.json").read_text())
        self.assertEqual(metadata["reference_commit"], "af39b31d984ba128020282764265fa38d31ce767")
        report = dict(atol=metadata["atol"], rtol=metadata["rtol"], reference_commit=metadata["reference_commit"],
                      reference_packages=metadata["packages"], dtype="float32",
                      local_packages={name: importlib.metadata.version(name) for name in metadata["packages"]}, errors={})
        for filename in metadata["metrics"]:
            oracle = torch.load(root / filename, map_location="cpu")
            records = []
            for index, pack in enumerate(oracle["packs"]):
                preds = dict(img=pack["pred"], depth=pack["pred_depth"], gaussian=torch.ones(1, 2, 14))
                gts = dict(img=pack["gt"], depth_m=pack["depth"])
                with patch("torch.hub.download_url_to_file", side_effect=AssertionError("Offline weights only")):
                    rows = batch_metrics(preds, gts, [f"bin_{index}"], [index], [False], compute_pcc=True,
                        eval_mask=pack["mask"], mask_cfg=oracle["mask_cfg"],
                        identity=dict(pixel_protocol="ddad_ego_novel12_v1", mask_manifest_sha256="fixture-mask-sha"))
                for name, expected in pack["image_metrics"].items():
                    actual = torch.tensor(rows[0]["per_view"][name])
                    error = float((actual-expected[0]).abs().max())
                    report["errors"][f"{filename}/bin{index}/{name}/per_view"] = error
                    torch.testing.assert_close(actual, expected[0], atol=metadata["atol"], rtol=metadata["rtol"])
                for actual, expected in zip(rows, pack["records"]):
                    self.assertEqual(actual["view_group"], expected["view_group"])
                    for key in ("psnr", "ssim", "lpips", "pcc"):
                        self.assertTrue(np.isclose(actual[key], expected[key], atol=metadata["atol"], rtol=metadata["rtol"]),
                                        (filename, index, actual["view_group"], key, actual[key], expected[key]))
                        report["errors"][f"{filename}/bin{index}/{actual['view_group']}/{key}"] = abs(actual[key]-expected[key])
                records.extend(rows)
            summary, ordered = summarize(records, ["bin_0", "bin_1"], 2)
            for group, expected in oracle["summary"].items():
                for key in ("psnr", "ssim", "lpips", "pcc"):
                    self.assertTrue(np.isclose(summary[group][key], expected[key], atol=metadata["atol"], rtol=metadata["rtol"]))
                    report["errors"][f"{filename}/{group}/{key}"] = abs(summary[group][key]-expected[key])
            for key in ("psnr", "ssim", "lpips"):
                self.assertAlmostEqual(summary['final/all_18'][key],
                    (12*summary['final/novel_12'][key]+6*summary['final/input_6'][key])/18, places=12)
            with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
                write_results(tmp, summary, ordered)
                self.assertEqual(len((Path(tmp)/"per_bin_metrics.csv").read_text().splitlines()), 7)
                self.assertEqual(len(json.loads((Path(tmp)/"per_view_metrics.json").read_text())), 2)
            for key, value in (("pixel_protocol", "full_image"), ("mask_manifest_sha256", "wrong"),
                               ("selection_sha256", "wrong")):
                bad = deepcopy(records); bad[0][key] = value
                with self.assertRaisesRegex(ValueError, "Mixed|Invalid"):
                    summarize(bad, ["bin_0", "bin_1"], 2)
        report["cuda_initialized"] = torch.cuda.is_initialized()
        (root.parent/"metric-comparison.json").write_text(json.dumps(report, indent=2)+"\n")

    def test_real_lpips_full_mask_invariance_and_valid_changes(self):
        torch.manual_seed(20)
        gt = torch.rand(2, 3, 48, 64)
        pred = (gt + .07 * torch.randn_like(gt)).clip(0, 1)
        mask = torch.ones(2, 48, 64, dtype=torch.bool); mask[0, 31:] = False
        cfg = build_eval_mask_config()
        with patch("torch.hub.download_url_to_file", side_effect=AssertionError("Offline weights only")):
            plain = metrics.compute_image_metrics(gt, pred)
            complete = metrics.compute_image_metrics(gt, pred, torch.ones_like(mask), cfg)
            masked = metrics.compute_image_metrics(gt, pred, mask, cfg)
            changed = pred.clone(); changed[0, :, 31:] = 1-changed[0, :, 31:]
            ignored = metrics.compute_image_metrics(gt, changed, mask, cfg)
            changed[0, :, :12] = 1-changed[0, :, :12]
            valid = metrics.compute_image_metrics(gt, changed, mask, cfg)
        for key in ("psnr", "ssim", "lpips"):
            self.assertTrue(torch.equal(complete[key], plain[key]))
            self.assertEqual(masked[key][1], plain[key][1])
            self.assertTrue(torch.equal(masked[key], ignored[key]))
            self.assertFalse(torch.isclose(masked[key][0], valid[key][0]))

    def test_psnr_ssim_support_and_degenerate_inputs(self):
        cfg = build_eval_mask_config()
        torch.manual_seed(6)
        gt = torch.rand(1, 3, 32, 40)*.5; pred = gt+.1
        mask = torch.ones(1, 32, 40, dtype=torch.bool); mask[:, 22:] = False
        # Real LPIPS is covered separately; these checks isolate window/denominator errors.
        with patch.object(metrics, "compute_lpips", lambda a,b: (a-b).square().mean((1,2,3))), \
             patch.object(metrics, "get_spatial_lpips", lambda *a: lambda gt,pred,**kw: (gt-pred).square().mean(1,keepdim=True)):
            values = metrics.compute_image_metrics(gt, pred, mask, cfg)
            self.assertAlmostEqual(values['psnr'].item(), 20., places=5)
            _, image = structural_similarity(gt[0].numpy(), pred[0].numpy(), win_size=11, gaussian_weights=True,
                                              channel_axis=0, data_range=1., full=True)
            centers = np.zeros((32,40), dtype=bool); centers[5:17,5:-5] = True
            self.assertEqual(values['ssim'].item(), float(image[:,centers].mean()))
            pred[:, :, 22:] = 0
            changed = metrics.compute_image_metrics(gt, pred, mask, cfg)
            self.assertTrue(torch.equal(changed['ssim'], values['ssim']))
            for area, message in ((torch.zeros_like(mask), 'no valid pixels'), (mask.float(), 'boolean')):
                with self.assertRaisesRegex(ValueError, message):
                    metrics.compute_image_metrics(gt,pred,area,cfg)
            area = torch.zeros_like(mask); area[:, :5] = True
            with self.assertRaisesRegex(ValueError, 'no valid SSIM windows'):
                metrics.compute_image_metrics(gt,pred,area,cfg)
            with self.assertRaisesRegex(ValueError, 'provided together'):
                metrics.compute_image_metrics(gt,pred,mask)
        depth = torch.arange(20, dtype=torch.float32).reshape(1,4,5)
        area = torch.ones_like(depth,dtype=torch.bool); area[:,-1] = False
        pred_depth = depth*2+1; pred_depth[:,-1] = -100
        self.assertAlmostEqual(metrics.compute_eval_pcc(depth,pred_depth,area).item(),1.,places=6)
        for bad in (torch.ones_like(depth), depth*float('nan')):
            with self.assertRaisesRegex(ValueError,'nonconstant'):
                metrics.compute_eval_pcc(depth,bad,area)
