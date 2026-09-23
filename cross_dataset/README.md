# PandaSet / DDAD 零样本泛化评估

加载 `workdirs` 中 reference 实验的 nuScenes 预训练 OmniScene 权重，在 PandaSet、DDAD 上直接评估，不进行目标域训练或微调。使用独立入口 `cross_dataset.evaluate`，原 OmniScene 数据加载、训练及推理入口保持不变。

本文命令使用 GPU，当前仅记录运行方式，尚未执行；等待明确允许使用 GPU 后再运行。

## 环境、数据与源权重

先激活环境并进入项目根目录，后续命令均在该目录执行：

```bash
conda activate omniscene
cd /home/dzp62442/Projects/Omni-Scene
```

- 数据入口：`data/PandaSet/processed`、`data/DDAD/processed`，复用现有软连接及预处理产物。
- 源实验：`workdirs/[reference]omni_gs_nusc_novelview_r50_112x200`。
- 源权重：上述目录的 `checkpoint-100000/model.safetensors`；向 `--load-from` 传入检查点目录即可，只读取模型权重，不恢复优化器状态。
- 源配置：上述目录保存的 `omni_gs_nusc_novelview_r50_112x200.py`，通过 `--source-config` 显式记录。
- 评估分辨率：**112×200**，与 reference 保存配置一致。224×400 实验应另行使用对应分辨率的源配置和权重。

命令中的 reference 路径包含方括号，必须保留引号，避免 shell 将其当作通配符。`PYTHONNOUSERSITE=1` 避免用户级 Python 包覆盖 Conda 环境；`CUDA_VISIBLE_DEVICES=0` 表示使用物理 GPU 0，可按获准使用的设备修改。下面使用 `python -m` 启动单进程评估，入口内部仍使用 Accelerate 管理设备。

## PandaSet：完整 test 评估

评估 SVF-GS 对应划分的全部 **3120 个 bin**：

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python -m cross_dataset.evaluate \
  --py-config configs/ZeroShot/omni_gs_nusc_to_pandaset_r50_112x200.py \
  --load-from 'workdirs/[reference]omni_gs_nusc_novelview_r50_112x200/checkpoint-100000' \
  --source-dataset nuScenes \
  --source-config 'workdirs/[reference]omni_gs_nusc_novelview_r50_112x200/omni_gs_nusc_novelview_r50_112x200.py' \
  --split test \
  --output-dir outputs/zero_shot/reference_nusc_to_pandaset_r50_112x200/test/checkpoint-100000
```

## DDAD：完整 test 评估

评估 SVF-GS 对应划分的全部 **395 个 bin**：

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python -m cross_dataset.evaluate \
  --py-config configs/ZeroShot/omni_gs_nusc_to_ddad_r50_112x200.py \
  --load-from 'workdirs/[reference]omni_gs_nusc_novelview_r50_112x200/checkpoint-100000' \
  --source-dataset nuScenes \
  --source-config 'workdirs/[reference]omni_gs_nusc_novelview_r50_112x200/omni_gs_nusc_novelview_r50_112x200.py' \
  --split test \
  --output-dir outputs/zero_shot/reference_nusc_to_ddad_r50_112x200/test/checkpoint-100000
```

## 单样本检查与重复运行

首次 GPU 验证时，可以在任一命令中增加 `--max-samples 1`，并将输出目录换为独立的诊断目录，例如：

```text
--max-samples 1
--output-dir outputs/zero_shot/reference_nusc_to_ddad_r50_112x200/smoke_test/checkpoint-100000
```

上面两行为需要加到评估命令中的参数，不是独立 shell 命令。限量运行记录为 `limited`，不能作为完整 test 结果。正式全量评估应去掉 `--max-samples`，使用前述完整 test 命令。

输出目录必须不存在或为空。重复运行时选择新的目录，例如在目录名后加 `_run2`；不要将结果写入 reference 源实验目录。

## 结果与协议

两个目标数据集均为同一帧六路输入、重建相同六路输出，报告 **`all_6` 输入视角重建**，不称为新视角生成。沿用 SVF-GS 的 `train/val/test` 索引；上述命令固定使用 `test`，不使用 mini 划分。

默认记录 PSNR、SSIM、LPIPS，先在每个 bin 内平均六路视角，再对唯一 bin 等权平均。PCC 默认关闭；需要启用时使用 Metric3D-v2 尺度深度。数据加载不依赖动态物体掩码。

每次评估在指定输出目录生成：

| 文件 | 内容 |
| --- | --- |
| `resolved_config.py` | 本次实际配置、源权重及输出路径 |
| `evaluation_manifest.json` | 源配置与权重摘要、目标数据、split、token 覆盖、协议及完成状态 |
| `evaluation_summary.json` | `all_6` 汇总指标、样本数量及完成状态 |
| `per_bin_metrics.csv` | 每个唯一 bin 的指标 |
| `per_view_metrics.json` | 各 bin 的逐视角指标 |

完整评估应同时满足 `status="complete"`、`complete_split=true`，且 `evaluated_count` 为 PandaSet 的 3120 或 DDAD 的 395。当前默认不保存可视化和 PLY。

当前 reference 检查点未提供 `source_metadata.json`，因此 manifest 中会记录 `unverified_legacy_checkpoint`。保存的源配置已通过静态模型设置一致性核对；完整原样模型的严格加载与真实前向仍待 GPU 验证。
