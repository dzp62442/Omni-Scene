# Evaluation only: supply explicit source model weights.
_base_ = ['../DDAD/omni_gs_ddad_r50_112x200.py']

exp_name = 'omni_gs_nusc_to_ddad_r50_112x200'
work_dir = 'workdirs/omni_gs_nusc_to_ddad_r50_112x200'
output_dir = 'outputs/zero_shot/nusc_to_ddad_r50_112x200'
zero_shot = True
source_dataset = 'nuScenes'
source_config = 'configs/OmniScene/omni_gs_nusc_novelview_r50_112x200.py'
load_from = None
split = 'test'
