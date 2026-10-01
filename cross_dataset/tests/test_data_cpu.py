"""Small real-asset CPU checks; reference tensors are exported under /tmp."""

import ast
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from cross_dataset.datasets import PandaSetDataset, DDADDataset
from cross_dataset.datasets.common import PROJECT_ROOT
from cross_dataset.datasets.assets import build_eval_mask_config, safe_asset_path
from cross_dataset.tests.cpu_only import CPUOnlyTest


def reference_fixture():
    path = Path(os.environ.get("OMNISCENE_REFERENCE_FIXTURE", "/tmp/omniscene-temporal18/reference")).resolve()
    if not path.is_relative_to(Path("/tmp")):
        raise ValueError("CPU reference fixtures and comparison reports must stay in /tmp")
    return path


class RealDataTests(CPUOnlyTest):
    def test_reference_values_splits_and_projection(self):
        root = reference_fixture()
        if not (root / "metadata.json").is_file():
            self.skipTest("Export the fixed SVF-GS CPU reference to /tmp first")
        metadata = json.loads((root / "metadata.json").read_text())
        self.assertEqual(metadata["reference_commit"], "af39b31d984ba128020282764265fa38d31ce767")
        max_errors = {}
        for entry in metadata["data"]:
            name, resolution = entry["dataset"], entry["resolution"]
            cls = PandaSetDataset if name == "pandaset" else DDADDataset
            mask_cfg = build_eval_mask_config() if name == "ddad" else None
            dataset = cls(resolution, "test", eval_mask_cfg=mask_cfg)
            self.assertEqual(dataset.bin_tokens, entry["tokens"])
            self.assertEqual(len(dataset), 264 if name == "pandaset" else 324)
            self.assertEqual(cls(resolution, "val").bin_tokens, entry["val_tokens"])
            for index in (0, len(dataset)//2, len(dataset)-1):
                actual = dataset[index]
                expected = torch.load(root / f"{name}_{resolution[0]}_{index}.pt", map_location="cpu")
                for section in ("inputs", "inputs_pix", "inputs_vol", "outputs"):
                    for key, tensor in actual[section].items():
                        reference = expected[section][key]
                        if name == "ddad":
                            # The fixed SVF-GS oracle stores the raw forward/left/up
                            # frame. Compare DDAD geometry after a change of basis;
                            # RGB, intrinsics, depth, masks and PandaSet stay exact.
                            if key == "c2w":
                                reference = torch.stack((-reference[..., 1, :], reference[..., 0, :],
                                                         reference[..., 2, :], reference[..., 3, :]), dim=-2)
                            elif key in ("rays_o", "rays_d"):
                                reference = torch.stack((-reference[..., 1], reference[..., 0],
                                                         reference[..., 2]), dim=-1)
                            elif key == "w2i":
                                reference = torch.stack((-reference[..., 1], reference[..., 0],
                                                         reference[..., 2], reference[..., 3]), dim=-1)
                        error = (tensor.float()-reference.float()).abs().max().item()
                        label = f"{name}/{resolution[0]}/{section}.{key}"
                        max_errors[label] = max(max_errors.get(label, 0), error)
                        atol = 1e-6
                        if name == "ddad" and key == "w2i":
                            # Re-solving inv(A @ T) in float32 can differ from
                            # permuting the already rounded inv(T). Bound rounding
                            # by four ulps at the projection matrix's scale.
                            atol = 4 * torch.finfo(reference.dtype).eps * reference.abs().max().item()
                        torch.testing.assert_close(tensor, reference, atol=atol, rtol=1e-6)
                self.assertNotIn("mask", actual["outputs"])
                torch.testing.assert_close(actual["outputs"]["rgb"][12:], actual["inputs"]["rgb"], atol=0, rtol=0)
                for field in ("depth_m", "conf_m", "c2w", "rays_o", "rays_d"):
                    torch.testing.assert_close(actual["outputs"][field][12:], actual["inputs_pix"][field], atol=0, rtol=0)
                self.check_projection(actual, resolution)
        (root.parent / "loader-comparison.json").write_text(json.dumps(max_errors, indent=2)+"\n")

    def test_prepared_file_inventory_and_missing_train(self):
        counts = {}
        for cls in (PandaSetDataset, DDADDataset):
            with self.assertRaisesRegex(FileNotFoundError, "train temporal18"):
                cls(split="train")
            dataset = cls(split="test")
            paths = set()
            for token in dataset.bin_tokens:
                paths.update(dataset.required_files(token))
            self.assertEqual([str(p) for p in paths if not p.is_file()], [])
            counts[dataset.dataset_name] = {"bins": len(dataset), "required_files": len(paths)}
        print("Read-only file inventory:", json.dumps(counts))

    def check_projection(self, sample, resolution):
        pix = sample["inputs_pix"]
        # Metric depth is optical-axis z, not normalized-ray distance.
        world = pix["rays_o"] + pix["rays_d"] * pix["depth_m"].unsqueeze(-1)
        world_h = torch.cat([world, torch.ones_like(world[..., :1])], dim=-1)
        projected = torch.einsum("vij,vhwj->vhwi", sample["inputs_vol"]["w2i"], world_h)
        uv = projected[..., :2] / projected[..., 2:3]
        h, w = resolution
        u, v = torch.meshgrid(torch.arange(w) + 0.5, torch.arange(h) + 0.5, indexing="xy")
        expected = torch.stack([u, v], -1).expand_as(uv)
        torch.testing.assert_close(uv, expected, atol=2e-3, rtol=0)


    def test_batch_and_working_directory_independence(self):
        dataset = DDADDataset((112, 200), "val")
        previous = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
                self.assertEqual(batch["inputs"]["rgb"].shape, (2, 6, 3, 112, 200))
                self.assertEqual(batch["bin_token"], dataset.bin_tokens[:2])
        finally:
            os.chdir(previous)


    def test_original_get_data_contract_without_constructing_model(self):
        # Execute the unchanged get_data/plucker methods on a tiny CPU holder.
        # Do not import or instantiate OmniGaussian or replace its CUDA behavior.
        tree = ast.parse((PROJECT_ROOT / "model/omni_gs.py").read_text())
        original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "OmniGaussian")
        methods = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in ("get_data", "plucker_embedder")]
        namespace = {"torch": torch}
        exec(compile(ast.Module(body=methods, type_ignores=[]), "original-data-contract", "exec"), namespace)
        holder_type = type("CPUDataHolder", (), {"device": torch.device("cpu"), "dtype": torch.float32,
                                               "get_data": namespace["get_data"],
                                               "plucker_embedder": namespace["plucker_embedder"]})
        for dataset in (PandaSetDataset((112, 200), "val"), DDADDataset((112, 200), "val")):
            batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
            data = holder_type().get_data(batch)
            self.assertEqual(data["imgs"].shape, (2, 6, 3, 112, 200))
            self.assertEqual(data["pluckers"].shape, (2, 6, 6, 112, 200))
            self.assertEqual(data["output_positions"].shape, (2, 18, 112, 200, 3))
            self.assertEqual(data["img_metas"][0]["lidar2img"].shape, (6, 4, 4))
            self.assertTrue(torch.equal(data["output_depths"], data["output_depths_m"]))



class InvalidDataTests(CPUOnlyTest):
    def test_split_flags_paths_and_missing_assets(self):
        for split in ("total", "mini-test", "demo"):
            with self.assertRaisesRegex(ValueError, "split must"):
                PandaSetDataset(split=split)
        for kwargs in ({"only_input": 1}, {"use_first": True}, {"use_center": False}):
            with self.assertRaisesRegex(ValueError, "six center inputs"):
                DDADDataset(**kwargs)
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "prepare them in SVF-GS"):
                DDADDataset(processed_root=tmp, split="test")
            for value in ("../escape", "/etc/passwd"):
                with self.assertRaisesRegex(ValueError, "relative"):
                    safe_asset_path(tmp, value)

    def test_reject_changed_selection_incomplete_manifest_and_bins(self):
        source = PROJECT_ROOT / "data/DDAD/processed"
        for kind in ("selection", "manifest", "bins"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(dir="/tmp") as tmp:
                root = Path(tmp)
                for prefix in ("selection", "manifest", "bins"):
                    payload = json.loads((source/f"{prefix}_test.json").read_text())
                    if prefix == kind:
                        if kind == "selection": payload["selection_sha256"] = "wrong"
                        if kind == "manifest": payload["complete"] = False
                        if kind == "bins": payload["bins"] = payload["bins"][::-1]
                    (root/f"{prefix}_test.json").write_text(json.dumps(payload))
                with self.assertRaises(ValueError):
                    DDADDataset(split="test", processed_root=root)

    def test_ego_mask_switch_and_corruptions(self):
        from cross_dataset.datasets import ego_mask
        cfg = build_eval_mask_config()
        plain = DDADDataset((112, 200), "val")[0]
        dataset = DDADDataset((112, 200), "val", eval_mask_cfg=cfg)
        enabled = dataset[0]
        for key, value in plain["outputs"].items():
            self.assertTrue(torch.equal(value, enabled["outputs"][key]))
        self.assertNotIn("eval_mask", plain["outputs"])
        area = enabled["outputs"]["eval_mask"]
        self.assertEqual(area.shape, (18, 112, 200))
        self.assertTrue(area[12:].all() and area[:2].all())
        self.assertFalse(area[2:12].all())
        diagnostic = DDADDataset((112, 200), "val", only_input=True, eval_mask_cfg=cfg)[0]
        self.assertEqual(diagnostic["outputs"]["eval_mask"].shape, (6, 112, 200))
        self.assertTrue(diagnostic["outputs"]["eval_mask"].all())
        with self.assertRaisesRegex(ValueError, "only supported"):
            PandaSetDataset(split="val", eval_mask_cfg=cfg)
        original = ego_mask.read_json
        for kind in ("selection", "missing", "sha", "camera", "intrinsic", "crop", "scene", "encoding"):
            def corrupt(path):
                payload = original(path)
                if Path(path).name != "manifest.json":
                    return payload
                variant = next(iter(payload["variants"].values()))
                first = next(iter(payload["masks"].values()))
                if kind == "selection": payload["selection_sha256"] = "wrong"
                if kind == "missing": first["path"] = "absent.png"
                if kind == "sha": first["sha256"] = "wrong"
                if kind == "camera": variant["camera"] = "invalid"
                if kind == "intrinsic": variant["intrinsic_raw"][0][2] += 1
                if kind == "crop":
                    for v in payload["variants"].values(): v["pixel_transform"][0][2] += 1
                if kind == "scene": payload["scene_camera_to_variant"].clear()
                if kind == "encoding": payload["mask_values"]["0"] = "valid"
                return payload
            with self.subTest(kind=kind), patch.object(ego_mask, "read_json", side_effect=corrupt):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    DDADDataset((112, 200), "val", eval_mask_cfg=cfg)[0]
