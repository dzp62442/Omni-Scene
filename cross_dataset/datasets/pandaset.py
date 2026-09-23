from .common import SingleFrameDataset


class PandaSetDataset(SingleFrameDataset):
    dataset_name = "PandaSet"
    depth_directory = "dptm"
