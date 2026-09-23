"""Read existing SVF-GS single-frame artifacts without importing its runtime.

Geometry and interpolation intentionally follow the reference loaders. No raw
SDK, dynamic mask, relative depth, or on-the-fly preprocessing is required.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMERA_TYPES = (
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
)
PROTOCOL = "svfgs_single_frame_v1"
SPLITS = ("train", "val", "test")


def project_path(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def hwc3(image):
    if image.ndim == 2:
        image = image[:, :, None]
    if image.shape[2] == 1:
        return np.concatenate([image] * 3, axis=2)
    if image.shape[2] == 4:
        color = image[:, :, :3].astype(np.float32)
        alpha = image[:, :, 3:4].astype(np.float32) / 255.0
        return (color * alpha + 255.0 * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
    if image.shape[2] != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    return image


class SingleFrameDataset(Dataset):
    camera_types = CAMERA_TYPES
    dataset_name = None
    depth_directory = None

    def __init__(self, resolution=(224, 400), split="train", processed_root=None,
                 use_center=True, use_first=False, use_last=False, only_input=True):
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if not use_center or use_first or use_last or not only_input:
            raise ValueError("PandaSet/DDAD require center-only, six-input/six-output reconstruction")
        if len(resolution) != 2 or any(int(v) != v or v <= 0 for v in resolution):
            raise ValueError(f"Invalid [height, width]: {resolution}")
        self.reso = tuple(int(v) for v in resolution)
        self.split = split
        self.only_input = True
        self.processed_root = project_path(
            processed_root or f"data/{self.dataset_name}/processed"
        )
        index = self.processed_root / ("bins_train.json" if split == "train" else "bins_test.json")
        with index.open(encoding="utf-8") as f:
            tokens = json.load(f)["bins"]
        if not tokens or len(set(tokens)) != len(tokens):
            raise ValueError(f"Empty or duplicate index: {index}")
        if any(not isinstance(t, str) or Path(t).name != t or t in (".", "..") for t in tokens):
            raise ValueError(f"Invalid bin token in {index}")
        if split == "val":
            tokens = [tokens[i] for i in np.linspace(0, len(tokens) - 1, 10, dtype=int)]
        self.bin_tokens = tokens

    def __len__(self):
        return len(self.bin_tokens)

    def image_path(self, stored_path):
        # Older PKLs include another project's relative or absolute prefix.
        # Always bind processed artifacts to this dataset's configured root.
        path = Path(stored_path)
        parts = path.parts
        if "images_small" not in parts:
            raise ValueError(f"Expected an images_small artifact, got {stored_path!r}")
        relative = Path(*parts[parts.index("images_small"):])
        if ".." in relative.parts:
            raise ValueError(f"Invalid artifact path: {stored_path!r}")
        return self.processed_root / relative

    def frame_paths(self, sensor):
        image = self.image_path(sensor["data_path"])
        camera = image.parent.name
        sequence = image.parent.parent.name
        if sequence == "camera" and self.dataset_name == "PandaSet":
            sequence = image.parent.parent.parent.name
        stem = image.stem
        return (
            image,
            self.processed_root / "params_small" / sequence / camera / f"{stem}.json",
            self.processed_root / self.depth_directory / sequence / camera / f"{stem}_dpt.npy",
            self.processed_root / self.depth_directory / sequence / camera / f"{stem}_conf.npy",
        )

    def read_sensors(self, token):
        path = self.processed_root / "bin_infos" / f"{token}.pkl"
        with path.open("rb") as f:
            info = pickle.load(f)
        return [info["sensor_info"][camera][0] for camera in self.camera_types]

    def required_files(self, token):
        """Metadata-only inventory for a full CPU audit (no image/depth decoding)."""
        yield self.processed_root / "bin_infos" / f"{token}.pkl"
        for sensor in self.read_sensors(token):
            yield from self.frame_paths(sensor)

    def __getitem__(self, index):
        token = self.bin_tokens[index]
        try:
            return self._load(token)
        except (OSError, KeyError, ValueError) as exc:
            raise type(exc)(f"{self.dataset_name}/{self.split}/{token}: {exc}") from exc

    def _load(self, token):
        height, width = self.reso
        images, depths, confs, intrinsics, c2ws, w2cs = [], [], [], [], [], []
        for sensor in self.read_sensors(token):
            image_path, param_path, depth_path, conf_path = self.frame_paths(sensor)
            with param_path.open(encoding="utf-8") as f:
                ck = np.array(json.load(f)["camera_intrinsic"])
            with Image.open(image_path) as image:
                source_shape = (image.height, image.width)
                if ck.shape != (3, 3) or not np.isfinite(ck).all():
                    raise ValueError(f"Invalid intrinsics: {param_path}")
                # Existing renderer uses symmetric FOV. Reject incompatible data,
                # rather than changing the original renderer or silently recentering.
                expected = np.array([[ck[0, 0], 0, image.width / 2],
                                     [0, ck[1, 1], image.height / 2], [0, 0, 1]])
                if min(ck[0, 0], ck[1, 1]) <= 0 or not np.allclose(ck, expected, atol=1e-4, rtol=0):
                    raise ValueError(f"Renderer requires centered, zero-skew intrinsics: {param_path}")
                resize = source_shape != self.reso
                if resize:
                    sy, sx = height / image.height, width / image.width
                    ck = np.array([[ck[0, 0] * sx, 0, ck[0, 2] * sx],
                                   [0, ck[1, 1] * sy, ck[1, 2] * sy], [0, 0, 1]])
                    image = image.resize((width, height))  # SVF-GS PIL default
                images.append(hwc3(np.array(image)))
            intrinsics.append(ck)
            for path, destination in ((depth_path, depths), (conf_path, confs)):
                value = np.load(path, allow_pickle=False).astype(np.float32)
                if value.shape != source_shape or not np.isfinite(value).all():
                    raise ValueError(f"Invalid shape or non-finite condition: {path}")
                if resize:
                    value = np.array(Image.fromarray(value).resize((width, height), Image.Resampling.BILINEAR))
                destination.append(value)

            transform = np.asarray(sensor["sensor2lidar_transform"])
            rotation = np.asarray(sensor["sensor2lidar_rotation"])
            translation = np.asarray(sensor["sensor2lidar_translation"])
            if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                    or not np.allclose(transform[:3, :3], rotation, atol=1e-5)
                    or not np.allclose(transform[:3, 3], translation, atol=1e-5)
                    or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)):
                raise ValueError(f"Inconsistent pose: {image_path}")
            # Match reference load_info, including its transposed w2c storage.
            inverse_rotation = np.linalg.inv(rotation)
            inverse_translation = translation @ inverse_rotation.T
            w2c = np.eye(4)
            w2c[:3, :3] = inverse_rotation.T
            w2c[3, :3] = -inverse_translation
            if not np.allclose(w2c.T @ transform, np.eye(4), atol=1e-5):
                raise ValueError(f"Pose inverse mismatch: {image_path}")
            c2ws.append(transform @ np.diag([1, -1, -1, 1]))
            w2cs.append(w2c)

        rgb = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255.0
        depth = torch.from_numpy(np.stack(depths))
        conf = torch.from_numpy(np.stack(confs))
        ck = torch.from_numpy(np.stack(intrinsics)).float()
        c2w = torch.from_numpy(np.stack(c2ws)).float()
        w2c = torch.from_numpy(np.stack(w2cs)).float()
        fx, fy, cx, cy = ck[:, 0, 0], ck[:, 1, 1], ck[:, 0, 2], ck[:, 1, 2]
        u, v = torch.meshgrid(torch.arange(width, dtype=torch.float32) + 0.5,
                              torch.arange(height, dtype=torch.float32) + 0.5, indexing="xy")
        directions = torch.stack([
            torch.stack([(u - x) / f, -(v - y) / g, -torch.ones_like(u)], dim=-1)
            for f, g, x, y in zip(fx, fy, cx, cy)
        ])
        rays_d = (directions[:, :, :, None, :] * c2w[:, None, None, :3, :3]).sum(-1)
        rays_o = c2w[:, None, None, :3, 3].expand_as(rays_d)
        k4 = torch.eye(4).repeat(6, 1, 1)
        k4[:, :3, :3] = ck
        w2i = torch.stack([k @ pose.T for k, pose in zip(k4, w2c)])
        return {
            "bin_token": token,
            "inputs": {"rgb": rgb},
            "inputs_pix": {"depth_m": depth, "conf_m": conf, "ck": ck, "c2w": c2w,
                           "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                           "rays_o": rays_o, "rays_d": rays_d},
            "inputs_vol": {"w2i": w2i},
            "outputs": {"rgb": rgb, "depth": depth, "depth_m": depth, "conf_m": conf,
                        "c2w": c2w, "fovx": 2 * torch.atan(cx / fx), "fovy": 2 * torch.atan(cy / fy),
                        "rays_o": rays_o, "rays_d": rays_d},
        }
