# Evaluation only: supply explicit source model weights.
_base_ = ['../PandaSet/omni_gs_pandaset_r50_224x400.py']

exp_name = 'omni_gs_nusc_to_pandaset_r50_224x400'
work_dir = 'workdirs/omni_gs_nusc_to_pandaset_r50_224x400'
output_dir = 'outputs/zero_shot/nusc_to_pandaset_r50_224x400'
zero_shot = True
source_dataset = 'nuScenes'
source_config = 'configs/OmniScene/omni_gs_nusc_novelview_r50_224x400.py'
load_from = None
split = 'test'
