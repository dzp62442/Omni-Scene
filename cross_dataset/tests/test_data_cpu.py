"""Real artifacts and independently executed SVF-GS reference definitions, CPU only."""

import ast
import copy
import json
import os
import os.path as osp
from pathlib import Path
import pickle as pkl
import tempfile
from types import SimpleNamespace

import numpy as np
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

from cross_dataset.datasets import PandaSetDataset, DDADDataset
from cross_dataset.datasets.common import PROJECT_ROOT, CAMERA_TYPES
from cross_dataset.tests.cpu_only import CPUOnlyTest

REFERENCE = PROJECT_ROOT.parent / "SVF-GS"


def definitions(path, names, namespace):
    """Execute only reference definitions; do not import its model/CUDA packages."""
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {n.name for n in selected} == set(names)
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


def reference_class(target):
    namespace = dict(np=np, torch=torch, PIL=PIL, Image=Image, Path=Path, json=json,
                     os=os, osp=osp, copy=copy, pkl=pkl, Dataset=Dataset)
    definitions(REFERENCE / "model/utils/image.py", ["HWC3"], namespace)
    definitions(REFERENCE / "model/utils/ops.py", ["get_ray_directions", "get_rays"], namespace)
    definitions(REFERENCE / "data/transforms/loading.py", ["load_info"], namespace)
    lower = target.lower()
    definitions(REFERENCE / f"data/transforms/{lower}_loading.py", [f"load_{lower}_conditions"], namespace)
    definitions(REFERENCE / f"data/{lower}_dataset.py", [f"{target}Dataset"], namespace)
    return namespace[f"{target}Dataset"]


class RealDataTests(CPUOnlyTest):
    def test_splits_reference_values_and_projection(self):
        max_errors = {}
        for target, cls in (("PandaSet", PandaSetDataset), ("DDAD", DDADDataset)):
            if not (PROJECT_ROOT / "data" / target / "processed").exists() or not REFERENCE.exists():
                self.skipTest("Real datasets and SVF-GS reference checkout are required")
            reference = reference_class(target)
            params = SimpleNamespace(**{f"{target.lower()}_processed_root": str(PROJECT_ROOT / "data" / target / "processed")})
            for resolution in ((112, 200), (224, 400)):
                for split in ("train", "val", "test"):
                    dataset = cls(resolution, split)
                    original = reference(params, resolution=list(resolution), split=split, load_rel_depth=False)
                    self.assertEqual(dataset.bin_tokens, original.bin_tokens)
                    self.assertEqual(len(dataset), {"PandaSet": {"train": 3120, "val": 10, "test": 3120},
                                                   "DDAD": {"train": 1265, "val": 10, "test": 395}}[target][split])
                    for index in (0, len(dataset) // 2, len(dataset) - 1):
                        actual, expected = dataset[index], original[index]
                        for section in ("inputs", "inputs_pix", "inputs_vol", "outputs"):
                            for key, tensor in actual[section].items():
                                error = (tensor - expected[section][key]).abs().max().item()
                                label = f"{target}/{section}.{key}"
                                max_errors[label] = max(max_errors.get(label, 0), error)
                                torch.testing.assert_close(tensor, expected[section][key], rtol=1e-6, atol=1e-6)
                        self.assertNotIn("mask", actual["outputs"])
                        self.assertIs(actual["outputs"]["rgb"], actual["inputs"]["rgb"])
                        self.assertIs(actual["outputs"]["depth"], actual["inputs_pix"]["depth_m"])
                        self.check_projection(actual, resolution)
        print("Reference max absolute differences:", json.dumps(max_errors, sort_keys=True))

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

    def test_full_file_inventory_and_overlap(self):
        counts = {}
        for cls in (PandaSetDataset, DDADDataset):
            train, test = cls(split="train"), cls(split="test")
            tokens = sorted(set(train.bin_tokens) | set(test.bin_tokens))
            overlap = len(set(train.bin_tokens) & set(test.bin_tokens))
            self.assertEqual(overlap, 3120 if cls is PandaSetDataset else 0)
            paths = set()
            for token in tokens:
                paths.update(train.required_files(token))
            missing = [str(path) for path in paths if not path.is_file()]
            self.assertEqual(missing, [])
            counts[train.dataset_name] = {"unique_bins": len(tokens), "required_files": len(paths), "train_test_overlap": overlap}
        print("Full artifact inventory:", json.dumps(counts))

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
            self.assertEqual(data["output_positions"].shape, (2, 6, 112, 200, 3))
            self.assertEqual(data["img_metas"][0]["lidar2img"].shape, (6, 4, 4))
            self.assertTrue(torch.equal(data["output_depths"], data["output_depths_m"]))


class InvalidDataTests(CPUOnlyTest):
    def test_invalid_split_and_temporal_flags(self):
        for split in ("total", "mini-test", "demo"):
            with self.assertRaisesRegex(ValueError, "split must"):
                PandaSetDataset(split=split)
        for kwargs in ({"only_input": False}, {"use_first": True}, {"use_center": False}):
            with self.assertRaisesRegex(ValueError, "center-only"):
                DDADDataset(**kwargs)

    def test_mask_free_fixture_and_actionable_missing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bin_infos").mkdir()
            (root / "bins_train.json").write_text(json.dumps({"bins": ["sample"]}))
            sensors = {}
            for index, camera in enumerate(CAMERA_TYPES):
                relative = Path("scene") / camera
                image = root / "images_small" / relative / "00.jpg"
                param = root / "params_small" / relative / "00.json"
                depth = root / "dptm_small" / relative / "00_dpt.npy"
                for path in (image, param, depth):
                    path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (48, 32), color=(20 + index, 40, 60)).save(image)
                param.write_text(json.dumps({"camera_intrinsic": [[30, 0, 24], [0, 31, 16], [0, 0, 1]]}))
                np.save(depth, np.full((32, 48), 7.0, dtype=np.float32))
                np.save(depth.with_name("00_conf.npy"), np.ones((32, 48), dtype=np.float32))
                sensors[camera] = [dict(data_path=f"/old/project/data/DDAD/processed/images_small/{relative}/00.jpg",
                                       sensor2lidar_transform=np.eye(4), sensor2lidar_rotation=np.eye(3),
                                       sensor2lidar_translation=np.zeros(3))]
            (root / "bin_infos/sample.pkl").write_bytes(pkl.dumps({"sensor_info": sensors}))
            dataset = DDADDataset((16, 24), processed_root=root)
            sample = dataset[0]
            self.assertNotIn("mask", sample["outputs"])
            self.assertTrue(torch.all(sample["outputs"]["depth"] == 7))
            depth.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "DDAD/train/sample.*00_dpt.npy"):
                dataset[0]
