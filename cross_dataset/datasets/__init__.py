from .ddad import DDADDataset
from .pandaset import PandaSetDataset

DATASETS = {"PandaSetDataset": PandaSetDataset, "DDADDataset": DDADDataset}


def build_dataset(params, split):
    name = params["dataset_name"]
    if name not in DATASETS:
        raise ValueError(f"This entrypoint only supports PandaSet/DDAD, got {name!r}")
    return DATASETS[name](
        resolution=params["resolution"], split=split,
        processed_root=params.get("processed_root"),
        use_center=params.get("use_center", True),
        use_first=params.get("use_first", False),
        use_last=params.get("use_last", False),
        only_input=params.get("only_input", True),
    )
