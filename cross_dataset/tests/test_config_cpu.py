from copy import deepcopy
from pathlib import Path
import tempfile

from mmengine.config import Config

from cross_dataset.configuration import load_config, source_declaration, output_path, training_directory
from cross_dataset.datasets.common import PROJECT_ROOT
from cross_dataset.tests.check_baseline_unchanged import check
from cross_dataset.tests.cpu_only import CPUOnlyTest


class ConfigTests(CPUOnlyTest):
    def test_frozen_boundary(self):
        self.assertEqual(check()["status"], "unchanged")

    def test_eight_configs_inherit_every_original_training_and_model_setting(self):
        for target in ("PandaSet", "DDAD"):
            for resolution in ("112x200", "224x400"):
                original = Config.fromfile(str(PROJECT_ROOT / f"configs/OmniScene/omni_gs_nusc_novelview_r50_{resolution}.py")).to_dict()
                full_path = f"configs/{target}/omni_gs_{target.lower()}_r50_{resolution}.py"
                full = load_config(full_path)
                zero = load_config(f"configs/ZeroShot/omni_gs_nusc_to_{target.lower()}_r50_{resolution}.py")
                for cfg in (full, zero):
                    actual = deepcopy(cfg.to_dict())
                    for scope in (actual["dataset_params"], actual["model"]["dataset_params"]):
                        self.assertEqual(scope.pop("processed_root"), f"data/{target}/processed")
                        self.assertFalse(scope.pop("only_input"))
                        scope.pop("temporal_cfg")
                        scope["dataset_name"] = "nuScenesDataset"
                    self.assertFalse(actual["eval_args"].pop("compute_pcc"))
                    self.assertFalse(actual["eval_args"].pop("eval_use_ego_mask"))
                    actual["eval_args"].pop("eval_mask_cfg")
                    actual["exp_name"] = original["exp_name"]
                    actual["output_dir"] = original["output_dir"]
                    # All original keys, including optimizer/loss/model/scheduler,
                    # must match after only the authorized target overrides.
                    self.assertEqual({k: actual[k] for k in original}, original)
                    self.assertEqual(set(actual) - set(original),
                                     {"work_dir", "protocol", "zero_shot", "split", "load_from"}
                                     | ({"source_dataset", "source_config"} if cfg.zero_shot else set()))
                self.assertEqual(source_declaration(zero)["dataset"], "nuScenes")

    def test_reject_wrong_entrypoint_and_inconsistent_resolution(self):
        with self.assertRaisesRegex(ValueError, "only PandaSet"):
            load_config("configs/OmniScene/omni_gs_nusc_novelview_r50_112x200.py")
        with self.assertRaisesRegex(ValueError, "zero-shot"):
            load_config("configs/ZeroShot/omni_gs_nusc_to_ddad_r50_112x200.py", training=True)
        cfg = load_config("configs/DDAD/omni_gs_ddad_r50_112x200.py")
        cfg.resolution = [224, 400]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mismatch.py"
            cfg.dump(str(path))
            with self.assertRaisesRegex(ValueError, "Resolution mismatch"):
                load_config(path)

    def test_source_and_destination_guards(self):
        cfg = load_config("configs/ZeroShot/omni_gs_nusc_to_pandaset_r50_112x200.py")
        with self.assertRaisesRegex(ValueError, "must differ"):
            source_declaration(cfg, "PandaSet")
        with self.assertRaisesRegex(ValueError, "differs"):
            source_declaration(cfg, source_config="configs/OmniScene/omni_gs_nusc_novelview_r50_224x400.py")
        for path in ("data/PandaSet/processed/run", "data/nuScenes/run", "checkpoints/new", "."):
            with self.assertRaisesRegex(ValueError, "overlaps"):
                output_path(path)
        cfg = load_config("configs/PandaSet/omni_gs_pandaset_r50_112x200.py")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(training_directory(cfg, tmp), Path(tmp))
            (Path(tmp) / "original.log").write_text("already occupied")
            with self.assertRaisesRegex(ValueError, "lacks a target"):
                training_directory(cfg, tmp)
            cfg.dump(str(Path(tmp) / "target.py"))
            with self.assertRaisesRegex(FileNotFoundError, "train temporal18"):
                training_directory(cfg, tmp)

    def test_training_assets_and_mask_mode_preflight(self):
        from cross_dataset.configuration import evaluation_mask_config, evaluation_output_directory
        for target in ("PandaSet", "DDAD"):
            path = f"configs/{target}/omni_gs_{target.lower()}_r50_112x200.py"
            with self.assertRaisesRegex(FileNotFoundError, "train temporal18"):
                load_config(path, training=True)
        cfg = load_config("configs/DDAD/omni_gs_ddad_r50_112x200.py")
        plain = evaluation_output_directory(cfg)
        evaluation_mask_config(cfg, "val", True)
        self.assertEqual(evaluation_output_directory(cfg), plain + "_ego_novel12_v1")
        with self.assertRaisesRegex(ValueError, "independent evaluation"):
            evaluation_mask_config(cfg, "train", True)
        cfg = load_config("configs/PandaSet/omni_gs_pandaset_r50_112x200.py")
        with self.assertRaisesRegex(ValueError, "independent evaluation"):
            evaluation_mask_config(cfg, "test", True)
