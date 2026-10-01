import numpy as np

from .common import TemporalDataset


class DDADDataset(TemporalDataset):
    dataset_name = "DDAD"
    depth_directory = "dptm_small"

    # Prepared DDAD: +X forward, +Y left, +Z up.
    # nuScenes-trained model: +X right, +Y forward, +Z up.
    # This changes the shared reference frame, not the camera-local axes.
    reference_to_model = (
        (0., -1., 0., 0.),
        (1., 0., 0., 0.),
        (0., 0., 1., 0.),
        (0., 0., 0., 1.),
    )

    def read_info(self, token):
        # Validate the immutable prepared poses in their original frame first.
        info, center, novel = super().read_info(token)
        for sensor in center + novel:
            pose = np.asarray(sensor["sensor2lidar_transform"])
            pose = np.asarray(self.reference_to_model, dtype=pose.dtype) @ pose
            sensor["sensor2lidar_transform"] = pose
            sensor["sensor2lidar_rotation"] = pose[:3, :3].copy()
            sensor["sensor2lidar_translation"] = pose[:3, 3].copy()
        # The unchanged camera_tensors() now derives c2w, rays and w2i from the
        # aligned poses for both inputs and targets, including only_input mode.
        return info, center, novel

    def evaluation_metadata(self):
        return {
            **super().evaluation_metadata(),
            "camera_frame": "nuscenes_axes_x_right_y_forward_z_up",
            "reference_to_model": [list(row) for row in self.reference_to_model],
        }
