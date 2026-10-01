"""DDAD reference-frame regression tests; no model or GPU is constructed."""

import pickle

import numpy as np
import torch

from cross_dataset.datasets import DDADDataset, PandaSetDataset
from cross_dataset.datasets.assets import build_eval_mask_config, file_digest
from cross_dataset.datasets.common import PROJECT_ROOT, TemporalDataset
from cross_dataset.tests.cpu_only import CPUOnlyTest


class PreparedDDAD(TemporalDataset):
    """The unchanged shared loader exposes the prepared frame for comparison."""
    dataset_name = "DDAD"
    depth_directory = "dptm_small"


def change_axes(value):
    # A physical forward displacement must become +Y, left must become -X.
    return torch.stack((-value[..., 1], value[..., 0], value[..., 2]), dim=-1)


class DDADAxesTests(CPUOnlyTest):
    def test_optical_axes_match_nuscenes_and_all_sensor_poses_are_aligned(self):
        import json
        root = PROJECT_ROOT / "data/nuScenes/interp_12Hz_trainval"
        token = json.loads((root / "bins_val_3.2m.json").read_text())["bins"][0]
        with (root / "bin_infos_3.2m" / f"{token}.pkl").open("rb") as handle:
            nusc = pickle.load(handle)
        raw = PreparedDDAD(split="test")
        aligned = DDADDataset(split="test")
        # Both camera rotations AND centers must change at every target time.
        for index in (0, len(raw)//2, len(raw)-1):
            token = raw.bin_tokens[index]
            path = raw.processed_root / "bin_infos" / f"{token}.pkl"
            digest = file_digest(path)
            _, raw_center, raw_novel = raw.read_info(token)
            info, center, novel = aligned.read_info(token)
            _, repeated_center, repeated_novel = aligned.read_info(token)
            for old, new, repeated in zip(raw_center+raw_novel, center+novel, repeated_center+repeated_novel):
                expected = np.stack((-old['sensor2lidar_transform'][1], old['sensor2lidar_transform'][0],
                                     old['sensor2lidar_transform'][2], old['sensor2lidar_transform'][3]))
                np.testing.assert_array_equal(new['sensor2lidar_transform'], expected)
                np.testing.assert_array_equal(new['sensor2lidar_rotation'], expected[:3,:3])
                np.testing.assert_array_equal(new['sensor2lidar_translation'], expected[:3,3])
                np.testing.assert_array_equal(repeated['sensor2lidar_transform'], expected)
                np.testing.assert_allclose(np.linalg.det(expected[:3,:3]), 1., atol=1e-6)
            for camera, sensor in zip(aligned.camera_types, center):
                self.assertIs(info['sensor_info'][camera][0], sensor)
            self.assertEqual(file_digest(path), digest)
        # Anchor the direction to actual training data, not just projection
        # invariance (which also passes in the incorrectly rotated world).
        _, raw_center, _ = raw.read_info(raw.bin_tokens[0])
        _, center, _ = aligned.read_info(aligned.bin_tokens[0])
        for slot, camera in enumerate(aligned.camera_types):
            train_axis = nusc['sensor_info'][camera][0]['sensor2lidar_transform'][:3,2]
            fixed_axis = center[slot]['sensor2lidar_transform'][:3,2]
            raw_axis = raw_center[slot]['sensor2lidar_transform'][:3,2]
            self.assertGreater(float(np.dot(train_axis, fixed_axis)), .95, camera)
            self.assertLess(abs(float(np.dot(train_axis, raw_axis))), .3, camera)

    def test_geometry_projection_relative_poses_and_unchanged_image_fields(self):
        for resolution in ((112,200), (224,400)):
            with self.subTest(resolution=resolution):
                mask_cfg = build_eval_mask_config()
                raw_ds = PreparedDDAD(resolution, "val", eval_mask_cfg=mask_cfg)
                fixed_ds = DDADDataset(resolution, "val", eval_mask_cfg=mask_cfg)
                before, after = raw_ds[0], fixed_ds[0]
                self.assertEqual(before.keys(), after.keys())
                for section in ("inputs", "inputs_pix", "outputs"):
                    for key, value in before[section].items():
                        actual = after[section][key]
                        if key == "c2w":
                            expected = value.clone()
                            expected[:,0] = -value[:,1]; expected[:,1] = value[:,0]
                            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        elif key in ("rays_o", "rays_d"):
                            torch.testing.assert_close(actual, change_axes(value), rtol=0, atol=0)
                        else:
                            self.assertTrue(torch.equal(actual,value), (section,key))
                for key in ('bin_token','scene_id','view_assets','view_cameras','view_times','eval_mask_manifest_sha256'):
                    self.assertEqual(before[key],after[key])
                for section, depth_key in (("inputs_pix", "depth_m"), ("outputs", "depth_m")):
                    old, new = before[section], after[section]
                    old_points = old['rays_o'] + old['rays_d']*old[depth_key][...,None]
                    new_points = new['rays_o'] + new['rays_d']*new[depth_key][...,None]
                    torch.testing.assert_close(new_points, change_axes(old_points), rtol=0, atol=0)
                    # Verify the geometric identity in float64; changing the
                    # subtraction order in a float32 cross product changes rounding.
                    old_moment = torch.cross(old['rays_o'].double(),old['rays_d'].double(),dim=-1)
                    new_moment = torch.cross(new['rays_o'].double(),new['rays_d'].double(),dim=-1)
                    torch.testing.assert_close(new_moment, change_axes(old_moment), rtol=1e-12, atol=1e-12)
                # All 6x18 relative camera poses and metric baselines are preserved.
                old_relative = torch.linalg.inv(before['inputs_pix']['c2w'].double())[:,None] @ before['outputs']['c2w'].double()[None]
                new_relative = torch.linalg.inv(after['inputs_pix']['c2w'].double())[:,None] @ after['outputs']['c2w'].double()[None]
                torch.testing.assert_close(new_relative,old_relative,rtol=1e-10,atol=1e-10)
                torch.testing.assert_close(torch.cdist(after['outputs']['c2w'][:,:3,3].double(),after['outputs']['c2w'][:,:3,3].double()),
                                           torch.cdist(before['outputs']['c2w'][:,:3,3].double(),before['outputs']['c2w'][:,:3,3].double()),rtol=1e-10,atol=1e-10)
                # Project each changed point using the correspondingly changed w2i.
                pix = after['inputs_pix']
                points = pix['rays_o']+pix['rays_d']*pix['depth_m'][...,None]
                homogeneous = torch.cat((points,torch.ones_like(points[...,:1])),dim=-1)
                projected = torch.einsum('vij,vhwj->vhwi',after['inputs_vol']['w2i'],homogeneous)
                uv = projected[...,:2]/projected[...,2:3]
                h,w = resolution
                x,y = torch.meshgrid(torch.arange(w)+.5,torch.arange(h)+.5,indexing='xy')
                torch.testing.assert_close(uv,torch.stack((x,y),-1).expand_as(uv),atol=2e-3,rtol=0)
                # Pixel protocol, prepared hashes and selected samples do not change.
                meta = fixed_ds.evaluation_metadata()
                self.assertEqual(meta.pop('camera_frame'),'nuscenes_axes_x_right_y_forward_z_up')
                self.assertEqual(meta.pop('reference_to_model'),[list(row) for row in DDADDataset.reference_to_model])
                self.assertEqual(meta,raw_ds.evaluation_metadata())

    def test_mask_switch_input_only_and_pandaset_unchanged(self):
        masked = DDADDataset((112,200),'val',eval_mask_cfg=build_eval_mask_config())[0]
        plain = DDADDataset((112,200),'val')[0]
        inputs_only = DDADDataset((112,200),'val',only_input=True)[0]
        for key,value in plain['outputs'].items():
            self.assertTrue(torch.equal(masked['outputs'][key],value),key)
            torch.testing.assert_close(inputs_only['outputs'][key],value[12:],atol=0,rtol=0)
        self.assertNotIn('eval_mask',plain['outputs'])
        panda = PandaSetDataset((112,200),'val')
        self.assertIs(type(panda).read_info,TemporalDataset.read_info)
        self.assertNotIn('reference_to_model',panda.evaluation_metadata())
