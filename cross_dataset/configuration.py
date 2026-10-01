"""Checks for the new target entrypoints; original configs remain read-only."""

from copy import deepcopy
from pathlib import Path

from mmengine.config import Config

from .datasets import DATASETS
from .datasets.common import PROJECT_ROOT, PROTOCOL, project_path
from .datasets.assets import build_eval_mask_config, validate_protocol


def model_signature(cfg):
    model = deepcopy(dict(cfg.model))
    # Dataset identity/loader options are not learned parameters. Spatial range is.
    params = model.pop("dataset_params")
    model["pc_range"] = params["pc_range"]
    return model


def load_config(path, training=False):
    cfg = Config.fromfile(str(project_path(path)))
    params = cfg.dataset_params
    if params.dataset_name not in DATASETS:
        raise ValueError("cross_dataset accepts only PandaSetDataset or DDADDataset")
    if cfg.get("protocol") != PROTOCOL:
        raise ValueError(f"Expected protocol={PROTOCOL!r}")
    if training and cfg.get("zero_shot", False):
        raise ValueError("A zero-shot evaluation config cannot be used for training")
    if (not isinstance(params.only_input, bool) or not params.use_center
            or params.use_first or params.use_last):
        raise ValueError("Temporal18 requires six center inputs; only_input selects six diagnostic targets")
    validate_protocol(params.temporal_cfg)
    if params.temporal_cfg.dataset != DATASETS[params.dataset_name].dataset_name.lower():
        raise ValueError("Temporal configuration dataset mismatch")
    if dict(cfg.model.dataset_params) != dict(params):
        raise ValueError("model.dataset_params and dataset_params must agree")
    resolution = list(params.resolution)
    for label, actual in (
        ("resolution", cfg.resolution), ("camera_args", cfg.camera_args.resolution),
        ("model.camera_args", cfg.model.camera_args.resolution),
        ("loss_args", cfg.loss_args.perceptual_resolution),
        ("model.loss_args", cfg.model.loss_args.perceptual_resolution),
    ):
        if list(actual) != resolution:
            raise ValueError(f"Resolution mismatch in {label}; choose the matching inherited config")
    if cfg.num_cams != 6 or cfg.model.pixel_gs.num_cams != 6:
        raise ValueError("Target datasets require six cameras")
    if cfg.model.loss_args != cfg.loss_args or cfg.model.dataset_params.pc_range != cfg.point_cloud_range:
        raise ValueError("Inconsistent nested loss/spatial configuration")
    # Shared target configs enable masks only for independent DDAD evaluation.
    evaluation_mask_config(cfg, "train" if training else cfg.get("split", "test"),
                           enabled=False if training else None, training=training)
    if training:
        # Fail before constructing Accelerator/model; never substitute test assets.
        from .datasets import build_dataset
        build_dataset(params, "train")
    return cfg


def evaluation_mask_config(cfg, split, enabled=None, training=False):
    enabled = cfg.eval_args.get("eval_use_ego_mask", False) if enabled is None else enabled
    if not isinstance(enabled, bool):
        raise ValueError("eval_use_ego_mask must be boolean")
    if enabled and (training or split == "train" or cfg.dataset_params.dataset_name != "DDADDataset"):
        raise ValueError("Ego masks require DDAD independent evaluation with split=test/val")
    cfg.eval_args.eval_use_ego_mask = enabled
    if not enabled:
        return None
    mask_cfg = dict(cfg.eval_args.eval_mask_cfg)
    if mask_cfg != build_eval_mask_config():
        raise ValueError("Ego mask metric configuration must match SVF-GS ddad_ego_novel12_v1")
    return mask_cfg


def evaluation_output_directory(cfg):
    tag = cfg.dataset_params.temporal_cfg.output_tag
    if cfg.dataset_params.only_input:
        tag += "_input6"
    if cfg.eval_args.eval_use_ego_mask:
        tag += cfg.eval_args.eval_mask_cfg.output_suffix
    return str(Path(cfg.output_dir) / tag)


def source_declaration(cfg, source_dataset=None, source_config=None):
    dataset = source_dataset or cfg.get("source_dataset")
    config_path = source_config or cfg.get("source_config")
    target = DATASETS[cfg.dataset_params.dataset_name].dataset_name
    if cfg.get("zero_shot", False):
        if not dataset or not config_path:
            raise ValueError("Zero-shot evaluation requires source_dataset and source_config")
        if dataset.casefold().removesuffix("dataset") == target.casefold():
            raise ValueError("Source and target datasets must differ in zero-shot evaluation")
    result = {"dataset": dataset, "config": None, "verification": "declared_only"}
    if config_path:
        path = project_path(config_path).resolve(strict=True)
        source = Config.fromfile(str(path))
        if model_signature(source) != model_signature(cfg):
            raise ValueError("Source/target model, resolution or spatial range differs")
        declared_name = (dataset or "").casefold().removesuffix("dataset")
        if dataset and declared_name != source.dataset_params.dataset_name.casefold().removesuffix("dataset"):
            raise ValueError("Source dataset declaration disagrees with source config")
        result["config"] = str(path)
    return result


def output_path(path, protected=()):
    destination = project_path(path).resolve()
    # Protect both local symlink entrances and the underlying shared datasets.
    roots = [PROJECT_ROOT / "data", PROJECT_ROOT / "checkpoints"]
    roots += [PROJECT_ROOT / "data" / name for name in ("nuScenes", "PandaSet", "DDAD")]
    roots += [Path(p) for p in protected]
    for root in roots:
        root = root.resolve()
        if destination == root or root in destination.parents or destination in root.parents:
            raise ValueError(f"Output overlaps protected data/checkpoint path: {destination}")
    return destination


def training_directory(cfg, requested=None):
    path = output_path(requested or cfg.work_dir)
    if path.exists() and any(path.iterdir()):
        # Resume only a directory that already contains a target config snapshot.
        # This is not a new optimizer/checkpoint format or data-position tracker.
        snapshots = list(path.glob("*.py"))
        if not snapshots:
            raise ValueError(f"Nonempty workdir lacks a target config snapshot: {path}")
        for snapshot in snapshots:
            previous = load_config(snapshot, training=True)
            if (previous.dataset_params.dataset_name != cfg.dataset_params.dataset_name
                    or previous.exp_name != cfg.exp_name
                    or model_signature(previous) != model_signature(cfg)):
                raise ValueError(f"Workdir belongs to a different experiment: {path}")
    return path
