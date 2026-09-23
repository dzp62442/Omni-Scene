# Target-only overrides; original OmniScene settings are inherited unchanged.
_base_ = ['../OmniScene/omni_gs_nusc_novelview_r50_224x400.py']

exp_name = 'omni_gs_pandaset_r50_224x400'
work_dir = 'workdirs/omni_gs_pandaset_r50_224x400'
output_dir = 'outputs/omni_gs_pandaset_r50_224x400'
protocol = 'svfgs_single_frame_v1'
zero_shot = False
split = 'test'
load_from = None

dataset_params = dict(
    dataset_name='PandaSetDataset',
    processed_root='data/PandaSet/processed',
    only_input=True,
)
model = dict(dataset_params=dataset_params)
eval_args = dict(compute_pcc=False)
