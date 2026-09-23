"""读取已经处理好的 DDAD 共享自车掩码；不下载或生成预处理资产。"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .assets import centered_crop, file_digest, read_json, safe_asset_path


class DDADEgoMasks:
    def __init__(self, processed_root, selection, resolution, cfg):
        self.cfg = dict(cfg)
        self.processed_root = Path(processed_root)
        self.resolution = tuple(resolution)
        path = safe_asset_path(self.processed_root, cfg["manifest_path"])
        self.root = path.parent
        self.manifest = read_json(path)
        self.manifest_sha256 = file_digest(path)
        self.cache = {}
        manifest = self.manifest
        if (selection["dataset"] != "ddad" or manifest["dataset"] != "ddad" or
                manifest["schema"] != cfg["schema"] or manifest["source_commit"] != cfg["source_commit"] or
                manifest["selection_sha256"] != selection["selection_sha256"]):
            raise ValueError("DDAD ego mask dataset/selection/source identity mismatch")
        if manifest["selection_file_sha256"] != file_digest(self.processed_root / f"selection_{selection['split']}.json"):
            raise ValueError("DDAD ego mask selection file changed")
        if (cfg["resize"] != "PIL_NEAREST" or manifest["transform"]["interpolation"] != "INTER_NEAREST" or
                manifest["transform"]["name"] != selection["protocol"]["crop_method"] or
                manifest["transform"]["image_hw"] != cfg["image_hw"] or
                cfg["image_hw"] != selection["protocol"]["image_hw"] or
                cfg["invalid_value"] != 0 or cfg["valid_value"] != 255 or
                manifest["mask_values"] != {"0": "exclude ego-vehicle region", "255": "valid region"}):
            raise ValueError("DDAD ego mask geometry/encoding protocol mismatch")
        cameras = [selection["protocol"]["camera_map"][c] for c in selection["protocol"]["camera_order"]]
        if manifest["camera_order"] != cameras:
            raise ValueError("DDAD ego mask camera order mismatch")
        for entries in (manifest["source_masks"], manifest.get("cleaned_source_masks", {}), manifest["masks"]):
            for entry in entries.values():
                if file_digest(safe_asset_path(self.root, entry["path"])) != entry["sha256"]:
                    raise ValueError("DDAD ego mask file checksum mismatch")
        # 在启动时核对完整引用范围，避免晚到某个 bin 才发现缺少场景或标定。
        for asset in selection["assets"].values():
            scene, camera, _ = asset["asset_id"].split("/")
            variant = self._variant(scene, camera)
            if not np.allclose(variant["intrinsic_raw"], asset["intrinsic"], rtol=0, atol=cfg["geometry_atol"]):
                raise ValueError(f"DDAD ego mask raw intrinsics mismatch: {scene}/{camera}")

    def _variant(self, scene, camera):
        manifest = self.manifest
        try:
            variant = manifest["variants"][manifest["scene_camera_to_variant"][scene][camera]]
            source = manifest["source_masks"][camera]
            manifest["masks"][variant["mask_id"]]
        except KeyError as exc:
            raise ValueError(f"Missing DDAD ego mask mapping: {scene}/{camera}") from exc
        if variant["camera"] != camera or variant["source_sha256"] != source["sha256"]:
            raise ValueError(f"DDAD ego mask camera/source mismatch: {scene}/{camera}")
        return variant

    def load(self, scene, infos):
        masks = []
        for info in infos:
            camera = info["camera"]
            if info["asset_id"].split("/")[:2] != [scene, camera]:
                raise ValueError("DDAD ego mask target scene/camera mismatch")
            variant = self._variant(scene, camera)
            param = read_json(safe_asset_path(self.processed_root, info["intrinsic_path"]))
            if (param["asset_id"] != info["asset_id"] or
                    param["selection_sha256"] != self.manifest["selection_sha256"] or
                    param["source_hw"] != variant["source_hw"] or param["image_hw"] != self.cfg["image_hw"]):
                raise ValueError("DDAD ego mask target asset/shape mismatch")
            affine, intrinsic = centered_crop(info["intrinsic"], param["source_hw"], param["image_hw"])
            pairs = [(variant["intrinsic_raw"], info["intrinsic"]), (variant["pixel_transform"], affine),
                     (variant["camera_intrinsic"], intrinsic), (param["pixel_transform"], affine),
                     (param["camera_intrinsic"], intrinsic)]
            if any(not np.allclose(a, b, rtol=0, atol=self.cfg["geometry_atol"]) for a, b in pairs):
                raise ValueError("DDAD ego mask target crop/intrinsics mismatch")
            key = (variant["mask_id"], *self.resolution)
            if key not in self.cache:
                entry = self.manifest["masks"][variant["mask_id"]]
                path = safe_asset_path(self.root, entry["path"])
                if file_digest(path) != entry["sha256"]:
                    raise ValueError("DDAD ego mask changed after initialization")
                with Image.open(path) as source:
                    pixels = np.asarray(source)
                    if (source.mode != "L" or list(pixels.shape) != self.cfg["image_hw"] or
                            not np.isin(pixels, [self.cfg["invalid_value"], self.cfg["valid_value"]]).all()):
                        raise ValueError("DDAD ego mask must be a binary image of the configured size")
                    resized = source.resize(self.resolution[::-1], Image.Resampling.NEAREST)
                    valid = torch.from_numpy(np.asarray(resized) == self.cfg["valid_value"])
                if not valid.any():
                    raise ValueError("DDAD ego mask has no valid pixels")
                self.cache[key] = valid
            masks.append(self.cache[key])
        return torch.stack(masks)

    def metadata(self):
        return dict(pixel_protocol=self.cfg["pixel_protocol"], mask_manifest_sha256=self.manifest_sha256,
                    source_commit=self.manifest["source_commit"], settings=deepcopy(self.cfg),
                    manifest_path=self.cfg["manifest_path"], asset_revision=self.manifest.get("asset_revision"),
                    template_quality=self.manifest["quality_note"])
