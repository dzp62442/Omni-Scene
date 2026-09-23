from .common import SingleFrameDataset


class DDADDataset(SingleFrameDataset):
    dataset_name = "DDAD"
    depth_directory = "dptm_small"
