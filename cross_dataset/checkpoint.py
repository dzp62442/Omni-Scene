"""Model-only strict checkpoint loading for target evaluation."""

import hashlib
import json
from pathlib import Path

import torch

from .datasets.common import project_path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_checkpoint(path):
    if not path:
        raise ValueError("Explicit --load-from or config.load_from is required")
    path = project_path(path)
    if path.is_dir():
        candidates = [path / name for name in ("model.safetensors", "pytorch_model.bin")]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin in {candidates[0].parent}")
    if not path.is_file():
        raise FileNotFoundError(f"Model weights not found: {path}")
    if path.suffix not in (".safetensors", ".bin", ".pt", ".pth"):
        raise ValueError(f"Unsupported model weights: {path}")
    return path.resolve()


def load_model_weights(model, path):
    path = resolve_checkpoint(path)
    if path.suffix == ".safetensors":
        # load_model understands shared parameter aliases saved by Accelerate.
        from safetensors.torch import load_model
        load_model(model, str(path), strict=True, device="cpu")
    else:
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError("Expected a model state_dict; optimizer/training states are not loaded")
        model.load_state_dict(state, strict=True)


def checkpoint_record(path, declaration, resolution, metadata_path=None):
    path = resolve_checkpoint(path)
    record = {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
              "source": dict(declaration), "source_metadata": None,
              "provenance_note": "Strict parameter loading does not verify the training dataset or resolution."}
    if declaration.get("config"):
        record["source"]["config_sha256"] = sha256_file(declaration["config"])
    sidecar = project_path(metadata_path) if metadata_path else path.parent / "source_metadata.json"
    if metadata_path or sidecar.is_file():
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        # Only hash-bound, explicit metadata is accepted as provenance evidence.
        for key, expected in (("checkpoint_sha256", record["sha256"]),
                              ("source_dataset", declaration["dataset"]),
                              ("resolution", list(resolution))):
            if metadata.get(key) != expected:
                raise ValueError(f"Source metadata mismatch/missing field: {key}")
        record["source_metadata"] = {"path": str(sidecar.resolve()), "sha256": sha256_file(sidecar),
                                     "contents": metadata}
        record["source"]["verification"] = "hash_bound_metadata_provided"
    else:
        record["source"]["verification"] = "unverified_legacy_checkpoint"
    return record
