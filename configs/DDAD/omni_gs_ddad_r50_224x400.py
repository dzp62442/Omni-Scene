# Target-only overrides; original OmniScene settings are inherited unchanged.
_base_ = ['../OmniScene/omni_gs_nusc_novelview_r50_224x400.py']

exp_name = 'omni_gs_ddad_r50_224x400'
work_dir = 'workdirs/omni_gs_ddad_r50_224x400'
output_dir = 'outputs/omni_gs_ddad_r50_224x400'
protocol = 'svfgs_temporal18_v1'
zero_shot = False
split = 'test'
load_from = None

dataset_params = dict(
    dataset_name='DDADDataset',
    processed_root='data/DDAD/processed',
    only_input=False,
    temporal_cfg={'schema': 'svfgs_temporal18_v1',
     'dataset': 'ddad',
     'center_stride': 10,
     'center_offset': 0,
     'target_side_m': 1.6,
     'min_side_m': 0.1,
     'image_hw': [224, 400],
     'camera_order': ['CAM_FRONT',
                      'CAM_FRONT_RIGHT',
                      'CAM_FRONT_LEFT',
                      'CAM_BACK',
                      'CAM_BACK_LEFT',
                      'CAM_BACK_RIGHT'],
     'camera_map': {'CAM_FRONT': 'CAMERA_01',
                    'CAM_FRONT_RIGHT': 'CAMERA_06',
                    'CAM_FRONT_LEFT': 'CAMERA_05',
                    'CAM_BACK': 'CAMERA_09',
                    'CAM_BACK_LEFT': 'CAMERA_07',
                    'CAM_BACK_RIGHT': 'CAMERA_08'},
     'mini_size': 100,
     'demo_size': 10,
     'raw_root': 'data/DDAD/ddad_train_val',
     'processed_root': 'data/DDAD/processed',
     'depth_dir': 'dptm_small',
     'crop_method': 'principal_center_float_v1',
     'depth_max_m': 300.0,
     'quaternion_atol': 0.0001,
     'depth_config': 'model/Metric3D/mono/configs/HourglassDecoder/vit.raft5.large.py',
     'depth_checkpoint': 'model/Metric3D/weight/metric_depth_vit_large_800k.pth',
     'depth_reference': 'metric3d_v2',
     'output_tag': 'novel18_s10_d1p6_min0p1'},
)
model = dict(dataset_params=dataset_params)
eval_args = dict(compute_pcc=False, eval_use_ego_mask=False,
                 eval_mask_cfg={'schema': 'svfgs_ddad_ego_mask_v1',
                  'manifest_path': 'ego_masks/vidar_v1/manifest.json',
                  'source_commit': '0d84851ce4d86a9f132f8027898ff981e751db79',
                  'pixel_protocol': 'ddad_ego_novel12_v1',
                  'output_suffix': '_ego_novel12_v1',
                  'image_hw': [224, 400],
                  'resize': 'PIL_NEAREST',
                  'novel_views': 12,
                  'input_views': 6,
                  'invalid_value': 0,
                  'valid_value': 255,
                  'geometry_atol': 1e-06,
                  'ssim_win_size': 11,
                  'ssim_sigma': 1.5,
                  'ssim_use_sample_covariance': True,
                  'lpips_net': 'vgg',
                  'lpips_normalize': True,
                  'lpips_rule': 'gt_fill_invalid_spatial_valid_mean',
                  'pcc_rule': 'valid_group_flatten'})
