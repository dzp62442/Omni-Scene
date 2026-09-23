"""Read-only contracts for prepared SVF-GS temporal18 assets.

Reference: SVF-GS af39b31. No selection, preprocessing, downloads or raw SDKs.
The crop matrix calculation only validates metadata; it never writes images.
"""

import hashlib
import json
import math
import os.path as osp
from pathlib import Path

import numpy as np

PROTOCOL = "svfgs_temporal18_v1"
REFERENCE_COMMIT = "af39b31d984ba128020282764265fa38d31ce767"

def build_zero_shot_config(dataset: str) -> dict:
    """只读资产的预期协议；路径仅用于来源校验，不加载深度网络。"""
    if dataset not in ("pandaset", "ddad"):
        raise ValueError(f"Unsupported zero-shot dataset: {dataset}")
    cameras = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    names = (["front_camera", "front_right_camera", "front_left_camera", "back_camera", "left_camera", "right_camera"]
             if dataset == "pandaset" else ["CAMERA_01", "CAMERA_06", "CAMERA_05", "CAMERA_09", "CAMERA_07", "CAMERA_08"])
    root = "data/PandaSet" if dataset == "pandaset" else "data/DDAD"
    return dict(schema="svfgs_temporal18_v1", dataset=dataset, center_stride=10, center_offset=0,
                target_side_m=1.6, min_side_m=0.1, image_hw=[224, 400], camera_order=cameras,
                camera_map=dict(zip(cameras, names)), mini_size=100, demo_size=10,
                raw_root=osp.join(root, "raw" if dataset == "pandaset" else "ddad_train_val"),
                processed_root=osp.join(root, "processed"), depth_dir="dptm" if dataset == "pandaset" else "dptm_small",
                crop_method="principal_center_float_v1", depth_max_m=300.0, quaternion_atol=1e-4,
                depth_config="model/Metric3D/mono/configs/HourglassDecoder/vit.raft5.large.py",
                depth_checkpoint="model/Metric3D/weight/metric_depth_vit_large_800k.pth",
                depth_reference="metric3d_v2", output_tag="novel18_s10_d1p6_min0p1")


def build_eval_mask_config() -> dict:
    """已处理的 DDAD 自车掩码与指标协议；独立于冻结的数据预处理清单。"""
    return dict(schema="svfgs_ddad_ego_mask_v1", manifest_path="ego_masks/vidar_v1/manifest.json",
                source_commit="0d84851ce4d86a9f132f8027898ff981e751db79",
                pixel_protocol="ddad_ego_novel12_v1", output_suffix="_ego_novel12_v1",
                image_hw=[224, 400], resize="PIL_NEAREST", novel_views=12, input_views=6,
                invalid_value=0, valid_value=255, geometry_atol=1e-6,
                ssim_win_size=11, ssim_sigma=1.5, ssim_use_sample_covariance=True,
                lpips_net="vgg", lpips_normalize=True,
                lpips_rule="gt_fill_invalid_spatial_valid_mean", pcc_rule="valid_group_flatten")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_protocol(cfg):
    required = ("schema", "dataset", "center_stride", "center_offset", "target_side_m", "min_side_m", "image_hw",
                "camera_order", "camera_map", "mini_size", "demo_size", "raw_root", "processed_root", "depth_dir",
                "crop_method", "depth_max_m", "quaternion_atol", "depth_config", "depth_checkpoint", "depth_reference", "output_tag")
    missing = set(required) - set(cfg)
    if missing:
        raise ValueError(f"Missing temporal configuration: {sorted(missing)}")
    if cfg["dataset"] not in ("pandaset", "ddad") or cfg["schema"] != "svfgs_temporal18_v1":
        raise ValueError("Unsupported temporal dataset/schema")
    for key in ("center_stride", "mini_size", "demo_size"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], int) or cfg[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not isinstance(cfg["center_offset"], int) or not 0 <= cfg["center_offset"] < cfg["center_stride"]:
        raise ValueError("Invalid center_offset")
    if not 0 < cfg["min_side_m"] <= cfg["target_side_m"] or not math.isfinite(cfg["target_side_m"]):
        raise ValueError("Invalid temporal distances")
    if len(cfg["image_hw"]) != 2 or any(not isinstance(v, int) or v <= 0 for v in cfg["image_hw"]):
        raise ValueError("Invalid image_hw")
    if len(cfg["camera_order"]) != 6 or len(set(cfg["camera_order"])) != 6:
        raise ValueError("Expected six distinct cameras")
    if set(cfg["camera_map"]) != set(cfg["camera_order"]) or len(set(cfg["camera_map"].values())) != 6:
        raise ValueError("Invalid camera_map")
    if cfg["crop_method"] != "principal_center_float_v1" or cfg["depth_reference"] != "metric3d_v2":
        raise ValueError("Unsupported crop/depth protocol")


def centered_crop(intrinsic, source_hw, target_hw):
    """返回原图像素→目标像素的浮点仿射与 K；两者使用同一映射。"""
    k = np.asarray(intrinsic, dtype=np.float64)
    height, width = source_hw
    target_h, target_w = target_hw
    cx, cy = k[0, 2], k[1, 2]
    half_h = min(cy, height-1-cy, min(cx, width-1-cx) * target_h / target_w)
    if not np.isfinite(k).all() or half_h <= 0:
        raise ValueError("Principal point must be strictly inside the source image")
    half_w = half_h * target_w / target_h
    sx, sy = target_w / (2*half_w), target_h / (2*half_h)
    affine = np.array([[sx, 0, target_w/2-sx*cx], [0, sy, target_h/2-sy*cy], [0, 0, 1]], dtype=np.float64)
    return affine, affine @ k


def safe_asset_path(root, relative):
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Expected a processed-relative asset path: {relative}")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Asset escapes processed root: {relative}")
    return path


def read_selection(root, split, cfg):
    selection = read_json(Path(root) / f"selection_{split}.json")
    saved = selection["selection_sha256"]
    payload = {k: v for k, v in selection.items() if k != "selection_sha256"}
    if digest(payload) != saved:
        raise ValueError("Selection checksum mismatch")
    expected = {k: v for k, v in cfg.items() if k not in ("raw_root", "processed_root")}
    if selection["protocol"] != expected or selection["dataset"] != cfg["dataset"] or selection["split"] != split:
        raise ValueError("Selection does not match the configured dataset/protocol/split")
    return selection
