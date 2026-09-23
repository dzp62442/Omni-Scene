"""Target-only 6-input/18-target loading of completed temporal18 artifacts."""

import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .assets import (PROTOCOL, REFERENCE_COMMIT, build_zero_shot_config, centered_crop,
                     file_digest, read_json, read_selection, safe_asset_path, validate_protocol)
from .ego_mask import DDADEgoMasks

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMERA_TYPES = tuple(build_zero_shot_config("ddad")["camera_order"])
SPLITS = ("train", "val", "test")


def project_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_temporal_conditions(infos, reso, processed_root, cfg):
    images, depths, confs, intrinsics, paths = [], [], [], [], []
    source_h, source_w = cfg["image_hw"]
    height, width = reso
    for info in infos:
        path = safe_asset_path(processed_root, info["data_path"])
        param = read_json(safe_asset_path(processed_root, info["intrinsic_path"]))
        metadata = read_json(safe_asset_path(processed_root, info["depth_meta_path"]))
        if metadata["synthetic"] or metadata["reference"] != cfg["depth_reference"]:
            raise ValueError("Expected real Metric3D assets")
        if metadata["model"] != info["depth_model"]:
            raise ValueError("Depth identity differs from the completed dataset")
        if (param["selection_sha256"] != info["selection_sha256"] or
                metadata["selection_sha256"] != info["selection_sha256"] or
                param["asset_id"] != info["asset_id"]):
            raise ValueError("Asset selection identity mismatch")
        if param["image_sha256"] != metadata["image_sha256"] or file_digest(path) != param["image_sha256"]:
            raise ValueError("RGB/depth identity mismatch")
        with Image.open(path) as source:
            if source.size != (source_w, source_h):
                raise ValueError(f"Unexpected image shape: {path}")
            image = source.convert("RGB")
            if (height, width) != (source_h, source_w):
                image = image.resize((width, height), Image.Resampling.BILINEAR)
            images.append(np.asarray(image, dtype=np.float32) / 255.0)
        k = np.asarray(param["camera_intrinsic"], dtype=np.float32).copy()
        affine, expected_k = centered_crop(info["intrinsic"], param["source_hw"], cfg["image_hw"])
        if not np.allclose(k, expected_k, atol=1e-5) or not np.allclose(param["pixel_transform"], affine, atol=1e-6):
            raise ValueError("Processed intrinsics no longer match the crop")
        if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
            raise ValueError("Invalid camera intrinsics")
        if not np.allclose(k[:2, 2], [source_w/2, source_h/2], atol=1e-4):
            raise ValueError("Expected principal-centered processed images")
        k[0] *= width/source_w
        k[1] *= height/source_h
        intrinsics.append(k)
        for field, collection in (("depth_path", depths), ("confidence_path", confs)):
            array_path = safe_asset_path(processed_root, info[field])
            array = np.load(array_path, allow_pickle=False)
            if array.shape != (source_h, source_w) or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"Invalid depth/confidence: {array_path}")
            if file_digest(array_path) != metadata[field + "_sha256"]:
                raise ValueError(f"Changed depth/confidence: {array_path}")
            if field == "depth_path" and (array.min() < 0 or array.max() <= 0 or array.max() > cfg["depth_max_m"]):
                raise ValueError("Invalid metric depth range")
            if (height, width) != (source_h, source_w):
                array = np.asarray(Image.fromarray(array).resize((width, height), Image.Resampling.BILINEAR)).copy()
            collection.append(array)
        paths.append(str(path))
    rgb = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
    depth = torch.from_numpy(np.stack(depths))
    confidence = torch.from_numpy(np.stack(confs))
    return dict(rgb=rgb, depth=depth, depth_m=depth, conf_m=confidence,
                ck=torch.from_numpy(np.stack(intrinsics)), paths=paths)


def camera_tensors(infos, intrinsics, reso):
    """与 OmniScene 一致的 OpenGL c2w、非归一化射线和列向量投影。"""
    transforms = torch.from_numpy(np.stack([np.asarray(i["sensor2lidar_transform"], dtype=np.float32) for i in infos]))
    if transforms.shape[1:] != (4, 4) or not torch.isfinite(transforms).all():
        raise ValueError("Invalid sensor transform")
    flip = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0]))
    c2w = transforms @ flip
    height, width = reso
    xx, yy = torch.meshgrid(torch.arange(width, dtype=torch.float32)+0.5,
                           torch.arange(height, dtype=torch.float32)+0.5, indexing="xy")
    fx, fy, cx, cy = (intrinsics[:, a, b] for a, b in ((0, 0), (1, 1), (0, 2), (1, 2)))
    directions = torch.stack(((xx[None]-cx[:, None, None])/fx[:, None, None],
                              -(yy[None]-cy[:, None, None])/fy[:, None, None],
                              -torch.ones(len(infos), height, width)), dim=-1)
    rays_d = torch.einsum("vij,vhwj->vhwi", c2w[:, :3, :3], directions)
    rays_o = c2w[:, None, None, :3, 3].expand_as(rays_d)
    viewpad = torch.eye(4).expand(len(infos), -1, -1).clone()
    viewpad[:, :3, :3] = intrinsics
    return dict(c2w=c2w, rays_o=rays_o, rays_d=rays_d, fovx=2*torch.atan(cx/fx), fovy=2*torch.atan(cy/fy),
                fx=fx, fy=fy, cx=cx, cy=cy, w2i=viewpad @ torch.linalg.inv(transforms))



class TemporalDataset(Dataset):
    camera_types = CAMERA_TYPES
    dataset_name = None
    depth_directory = None

    def __init__(self, resolution=(224, 400), split="train", processed_root=None,
                 use_center=True, use_first=False, use_last=False, only_input=False,
                 temporal_cfg=None, eval_mask_cfg=None):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if not use_center or use_first or use_last or not isinstance(only_input, bool):
            raise ValueError("Temporal datasets require six center inputs and a boolean only_input")
        if len(resolution) != 2 or any(isinstance(v, bool) or int(v) != v or v <= 0 for v in resolution):
            raise ValueError(f"Invalid [height, width]: {resolution}")
        self.reso = tuple(int(v) for v in resolution)
        self.split, self.only_input = split, only_input
        self.cfg = dict(temporal_cfg or build_zero_shot_config(self.dataset_name.lower()))
        validate_protocol(self.cfg)
        if self.cfg["dataset"] != self.dataset_name.lower():
            raise ValueError("Dataset configuration mismatch")
        self.processed_root = project_path(processed_root or self.cfg["processed_root"])
        self.cfg["processed_root"] = str(self.processed_root)
        storage_split = "train" if split == "train" else "test"
        for prefix in ("selection", "manifest", "bins"):
            path = self.processed_root / f"{prefix}_{storage_split}.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing completed {storage_split} temporal18 assets: {path}; "
                                        "prepare them in SVF-GS; test is never a train fallback")
        self.selection = read_selection(self.processed_root, storage_split, self.cfg)
        self.manifest_path = self.processed_root / f"manifest_{storage_split}.json"
        self.manifest = read_json(self.manifest_path)
        if (self.manifest["complete"] is not True or self.manifest["schema"] != PROTOCOL
                or self.manifest["selection_sha256"] != self.selection["selection_sha256"]
                or self.manifest["protocol"] != self.selection["protocol"]):
            raise ValueError("Temporal preprocessing incomplete or inconsistent")
        payload = read_json(self.processed_root / f"bins_{storage_split}.json")
        tokens = payload["bins"]
        if (not tokens or len(set(tokens)) != len(tokens)
                or payload["selection_sha256"] != self.selection["selection_sha256"]
                or tokens != [row["bin_token"] for row in self.selection["bins"]]):
            raise ValueError("Published bin coverage mismatch")
        if any(not isinstance(t, str) or Path(t).name != t or t in (".", "..") for t in tokens):
            raise ValueError("Invalid bin token")
        self.rows = {row["bin_token"]: row for row in self.selection["bins"]}
        self.bin_tokens = tokens if split != "val" else [
            tokens[i] for i in np.linspace(0, len(tokens)-1, min(10, len(tokens)), dtype=int)]
        self.eval_mask_cfg, self.ego_masks = eval_mask_cfg, None
        if eval_mask_cfg is not None:
            if self.dataset_name != "DDAD" or split == "train":
                raise ValueError("Ego masks are only supported for DDAD independent evaluation")
            self.ego_masks = DDADEgoMasks(self.processed_root, self.selection, self.reso, eval_mask_cfg)

    def __len__(self):
        return len(self.bin_tokens)

    def evaluation_metadata(self):
        result = dict(protocol=self.manifest["protocol"], reference_commit=REFERENCE_COMMIT,
                      selection_sha256=self.selection["selection_sha256"],
                      manifest_sha256=file_digest(self.manifest_path), depth_model=self.manifest["depth_model"],
                      camera_order=list(self.camera_types), camera_map=self.cfg["camera_map"],
                      view_order="center camera order" if self.only_input else "before/after per camera, then center cameras",
                      input_views=6, output_views=6 if self.only_input else 18,
                      pixel_protocol="full_image", mask_manifest_sha256="")
        if self.ego_masks is not None:
            result.update(self.ego_masks.metadata())
        return result

    def read_info(self, token):
        path = self.processed_root / "bin_infos" / f"{token}.pkl"
        if file_digest(path) != self.manifest["bin_info_sha256"][token]:
            raise ValueError(f"Bin identity mismatch: {token}")
        with path.open("rb") as handle:
            info = pickle.load(handle)
        if (info["bin_token"] != token or info["schema"] != PROTOCOL
                or info["selection_sha256"] != self.selection["selection_sha256"]):
            raise ValueError("Bin schema/selection mismatch")
        row = self.rows[token]
        for key in ("scene_id", "frame_ids", "frame_indices", "frame_gaps", "distance_m"):
            if info[key] != row[key]:
                raise ValueError(f"Bin temporal identity mismatch: {key}")
        center, novel = [], []
        reference_inverse = np.linalg.inv(np.asarray(row["center_lidar_pose"], dtype=np.float64))
        for camera in self.camera_types:
            sensors = info["sensor_info"][camera]
            if len(sensors) != 3:
                raise ValueError("Expected [center, before, after] for each camera")
            for sensor, asset_id in zip(sensors, row["sensor_assets"][camera]):
                if sensor["camera"] != self.cfg["camera_map"][camera] or sensor["asset_id"] != asset_id:
                    raise ValueError("Camera/time semantics do not match selection")
                asset = self.selection["assets"][asset_id]
                for key in ("data_path", "intrinsic_path", "depth_path", "confidence_path", "depth_meta_path"):
                    if sensor[key] != asset[key]:
                        raise ValueError(f"Asset path differs from selection: {key}")
                transform = np.asarray(sensor["sensor2lidar_transform"])
                expected = (reference_inverse @ np.asarray(asset["world_pose"], dtype=np.float64)).astype(np.float32)
                if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                        or not np.allclose(transform, expected, rtol=1e-6, atol=1e-5)
                        or not np.allclose(transform[:3, :3], sensor["sensor2lidar_rotation"], atol=1e-6)
                        or not np.allclose(transform[:3, 3], sensor["sensor2lidar_translation"], atol=1e-6)):
                    raise ValueError("Inconsistent pose in center lidar coordinates")
                sensor["selection_sha256"] = self.selection["selection_sha256"]
                sensor["depth_model"] = self.manifest["depth_model"]
            center.append(sensors[0])
            novel.extend(sensors[1:])
        return info, center, novel

    def required_files(self, token):
        yield self.processed_root / "bin_infos" / f"{token}.pkl"
        _, center, novel = self.read_info(token)
        for sensor in center + novel:
            for key in ("data_path", "intrinsic_path", "depth_path", "confidence_path", "depth_meta_path"):
                yield safe_asset_path(self.processed_root, sensor[key])

    def __getitem__(self, index):
        token = self.bin_tokens[index]
        try:
            return self._load(token)
        except (OSError, KeyError, ValueError) as exc:
            raise type(exc)(f"{self.dataset_name}/{self.split}/{token}: {exc}") from exc

    def _load(self, token):
        info, center, novel = self.read_info(token)
        inputs = load_temporal_conditions(center, self.reso, self.processed_root, self.cfg)
        input_cameras = camera_tensors(center, inputs["ck"], self.reso)
        if self.only_input:
            outputs, cameras, sensors = inputs, input_cameras, center
        else:
            outputs = load_temporal_conditions(novel, self.reso, self.processed_root, self.cfg)
            cameras = camera_tensors(novel, outputs["ck"], self.reso)
            outputs = {k: outputs[k]+inputs[k] if k == "paths" else torch.cat((outputs[k], inputs[k])) for k in outputs}
            cameras = {k: torch.cat((cameras[k], input_cameras[k])) for k in cameras}
            sensors = novel + center
        input_pix = {k: inputs[k] for k in ("depth_m", "conf_m", "ck")}
        input_pix.update({k: input_cameras[k] for k in ("c2w", "fx", "fy", "cx", "cy", "rays_o", "rays_d")})
        output = {k: outputs[k] for k in ("rgb", "depth", "depth_m", "conf_m")}
        output.update({k: cameras[k] for k in ("c2w", "fovx", "fovy", "rays_o", "rays_d")})
        if self.ego_masks is not None:
            input_mask = torch.ones((6, *self.reso), dtype=torch.bool)
            output["eval_mask"] = input_mask if self.only_input else torch.cat((self.ego_masks.load(info["scene_id"], novel), input_mask))
        result = dict(bin_token=token, scene_id=info["scene_id"], inputs=dict(rgb=inputs["rgb"]),
                      inputs_pix=input_pix, inputs_vol=dict(w2i=input_cameras["w2i"]), outputs=output,
                      view_assets=[s["asset_id"] for s in sensors], view_cameras=[s["camera"] for s in sensors],
                      view_times=["center"]*6 if self.only_input else ["before", "after"]*6 + ["center"]*6)
        if self.ego_masks is not None:
            result["eval_mask_manifest_sha256"] = self.ego_masks.manifest_sha256
        return result
