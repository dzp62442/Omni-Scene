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

评估 SVF-GS 对应划分的全部 **264 个 bin**：

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python -m cross_dataset.evaluate \
  --py-config configs/ZeroShot/omni_gs_nusc_to_pandaset_r50_112x200.py \
  --load-from 'workdirs/[reference]omni_gs_nusc_novelview_r50_112x200/checkpoint-100000' \
  --source-dataset nuScenes \
  --source-config 'workdirs/[reference]omni_gs_nusc_novelview_r50_112x200/omni_gs_nusc_novelview_r50_112x200.py' \
  --split test \
  --output-dir outputs/zero_shot/reference_nusc_to_pandaset_r50_112x200/test/checkpoint-100000/novel18_s10_d1p6_min0p1
```

## DDAD：完整 test，手动关闭自车掩码

评估 SVF-GS 对应划分的全部 **324 个 bin**，用 `--no-eval-use-ego-mask` 覆写为全图评估：

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python -m cross_dataset.evaluate \
  --py-config configs/ZeroShot/omni_gs_nusc_to_ddad_r50_112x200.py \
  --load-from 'workdirs/[reference]omni_gs_nusc_novelview_r50_112x200/checkpoint-100000' \
  --source-dataset nuScenes \
  --source-config 'workdirs/[reference]omni_gs_nusc_novelview_r50_112x200/omni_gs_nusc_novelview_r50_112x200.py' \
  --split test \
  --no-eval-use-ego-mask \
  --output-dir outputs/zero_shot/reference_nusc_to_ddad_r50_112x200/test/checkpoint-100000/novel18_s10_d1p6_min0p1
```

## DDAD：完整 test，默认使用自车掩码

DDAD 独立评估默认启用自车掩码（112×200、224×400 均生效），无需额外开关。仍评估相同 324 个 bin：`novel_12` 按有效像素计算，`input_6` 保持全图，`all_18` 对 18 个视角等权平均。训练与 PandaSet 保持不使用自车掩码。

```bash
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python -m cross_dataset.evaluate \
  --py-config configs/ZeroShot/omni_gs_nusc_to_ddad_r50_112x200.py \
  --load-from 'workdirs/[reference]omni_gs_nusc_novelview_r50_112x200/checkpoint-100000' \
  --source-dataset nuScenes \
  --source-config 'workdirs/[reference]omni_gs_nusc_novelview_r50_112x200/omni_gs_nusc_novelview_r50_112x200.py' \
  --split test \
  --output-dir outputs/zero_shot/reference_nusc_to_ddad_r50_112x200/test/checkpoint-100000/novel18_s10_d1p6_min0p1_ego_novel12_v1
```

仅读取 `data/DDAD/processed/ego_masks/vidar_v1` 中 SVF-GS 已处理的模板，不生成或修改掩码。masked PSNR、SSIM、LPIPS 及分组平均与 SVF-GS `af39b31` 对齐；其中 SSIM 排除跨越无效像素的窗口，LPIPS 在指标内部用 GT 填充无效区，再平均有效区的空间距离图。全白视角仍使用原全图函数，不丢弃这些视角。

## 单样本检查与重复运行

用户明确授权后的首次 GPU 验证，只跑一个样本，产物放到 `/tmp`。可以在任一命令中增加 `--max-samples 1`，并将输出目录换为独立的诊断目录，例如：

```text
--max-samples 1
--output-dir /tmp/omniscene-ddad-smoke-novel18-run1/novel18_s10_d1p6_min0p1
```

上面两行为需要加到评估命令中的参数，不是独立 shell 命令。限量运行记录为 `limited`，不能作为完整 test 结果。正式全量评估应去掉 `--max-samples`，使用前述完整 test 命令。

输出目录必须不存在或为空。重复运行时选择新的目录，例如在目录名后加 `_run2`；不要将结果写入 reference 源实验目录。

## 结果与协议

两个数据集默认 **中央 6 路输入 → 前后帧 12 路与中央 6 路共 18 路输出**，输出顺序是按相机交错的 before/after，然后中央六路。保存三组 `final/all_18`、`final/novel_12`、`final/input_6`，正式前馈结果采用 `all_18`。显式 `only_input=True` 配置仅用于新清单上的 `final/all_6` 诊断。

本项目只接受 `train/val/test`：test 对应 SVF-GS Dataset 的完整 test；val 是 test 均匀抽取的 10 个 bin。新预处理产物尚无独立 train 清单，训练入口会在构造模型前报错，不能用 test 代替。八份目标配置保留原 OmniScene 模型、损失、优化器、调度和训练循环设置。

默认记录 PSNR、SSIM、LPIPS，先逐视角计算，再在每个 bin 的组内以 float64 等权平均，最后对唯一 bin 等权平均；不能按有效像素数对视角加权。PCC 默认关闭；需要启用时使用 Metric3D-v2 尺度深度。数据加载不依赖动态物体掩码。

每次评估在指定输出目录生成：

| 文件 | 内容 |
| --- | --- |
| `resolved_config.py` | 本次实际配置、源权重及输出路径 |
| `evaluation_manifest.json` | 源配置与权重摘要、目标数据、split、token 覆盖、协议及完成状态 |
| `evaluation_summary.json` | 三组汇总指标、样本数量及完成状态 |
| `per_bin_metrics.csv` | 每个唯一 bin 的三组指标（共 3N 行） |
| `per_view_metrics.json` | 各 bin 的逐视角指标 |

完整评估应同时满足 `status="complete"`、`complete_split=true`，且 `evaluated_count` 为 PandaSet 的 264 或 DDAD 的 324。默认不保存可视化和 PLY。开启保存时，每个 bin 只导出一次 PLY，RGB/深度按 before、after、input 三组六相机布局保存，保留实际渲染结果。

manifest 与各组结果记录 selection/manifest 摘要、`pixel_protocol`、`mask_manifest_sha256`。全图和掩码实验使用不同空目录；掩码版本、选帧或像素协议不同的记录不能混合汇总。省略 `--output-dir` 时，入口按最终 only_input/掩码开关追加协议目录；显式路径优先。

当前 reference 检查点未提供 `source_metadata.json`，因此 manifest 中会记录 `unverified_legacy_checkpoint`。保存的源配置已通过静态模型设置一致性核对；完整原样模型的严格加载与真实前向仍待 GPU 验证。

## 小规模 CPU 开发验证

调试产物统一放在 `/tmp`，不运行完整模型、不初始化 CUDA。配置/汇总/训练调用等轻量检查：

```bash
mkdir -p /tmp/omniscene-temporal18
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 \
  python -m unittest discover -s cross_dataset/tests -p 'test_*cpu.py' -v \
  > /tmp/omniscene-temporal18/cpu-tests.log 2>&1
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
  python -m cross_dataset.tests.check_baseline_unchanged
```

真实数据对照使用两数据集各首/中/末三个 bin、两种分辨率；指标对照使用每分辨率两个合成 bin，运行真实 VGG LPIPS。测试只消费固定 SVF-GS 版本独立导出的夹具：

```bash
# 从固定版本的 SVF-GS 根目录，使用其 svfgs 环境，仅导出 CPU 对照数据。
# 不导入或执行预处理；结果必须在 /tmp。
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 PYTHONPATH=. \
  python /home/dzp62442/Projects/Omni-Scene/cross_dataset/tests/export_reference_cpu.py \
  --output /tmp/omniscene-temporal18/reference
```

返回本项目与 `omniscene` 环境后运行上述测试。可通过 `OMNISCENE_REFERENCE_FIXTURE` 指定另一个 `/tmp` 夹具目录；未准备参考夹具时相应对照项会 skip，不能当作数值一致性已通过。LPIPS 只用已有本地权重，测试禁止自动下载。CPU 对照不代表完整原样模型的 GPU 前向或正式测试集指标已通过。
