# PandaSet 与 DDAD 零样本泛化及训练评估接入方案

> 修订日期：2026-09-18。本版替代此前包含“公共训练流程修正”的方案，以用户最新要求为准。
>
> **最高优先级原则：原有 OmniScene 的数据加载、训练、推理流程及其代码、配置绝对不可改动。只新增 PandaSet/DDAD 的数据加载、完整实验配置与必要的训练、推理适配。**
>
> 当前已按本版完成独立实现与 CPU 验证，详见第 14 节。此前撤回实现的测试结果未用于本版验收。原有运行文件保持冻结，目标数据入口统一为 `data/PandaSet`、`data/DDAD`；本次未使用 GPU，完整原样模型的严格加载、前向与训练验证仍等待用户明确允许。

## 1. 不可突破的开发边界

### 1.1 原有实现全部冻结

冻结基线为当前提交 `a8318b4da99879c625a60154d111a510e5ac5a91`。实施前再次确认工作区状态，以该基线的文件内容作为核验依据。

| 范围 | 冻结内容 |
| --- | --- |
| 原有数据层 | `data/dataloader.py`、`data/transforms/loading.py` 及其余原有数据代码 |
| 原有执行入口 | 根目录 `train.py`、`evaluate.py`、`demo.py` 及原有启动脚本 |
| 原有配置 | `configs/OmniScene/`、其 `_base_` 文件、`accelerate_config.yaml` |
| 模型与渲染 | 整个原有 `model/`，包括优化器构建、损失、验证方法、encoder、decoder、渲染器和几何工具 |
| 构建与公共工具 | `builder/`、原有 `tools/`、包初始化文件及依赖配置 |
| 其他原有文件 | README 等保持原样；根目录 `.gitignore` 仅允许第 1.2 节明确授权的两条数据入口忽略规则 |

约束是**原有运行代码、配置与执行路径不变**。`.gitignore` 的两条已授权规则是文件冻结范围中的明确例外，不构成修改原数据代码或运行流程的授权。即使认为改动能修复问题、简化代码、改善兼容性，或抽样输出相同，也不能修改冻结文件。具体包括：

- 不在原 `train.py`、`evaluate.py`、`data/dataloader.py` 中增加新数据集分支。
- 不调整 nuScenes 路径解析、数据划分、导入方式、batch 字段或几何处理。
- 不统一原优化器配置来源，不修改调度器、停止条件、计步、验证、保存及恢复行为。
- 不把原模型函数搬到新文件再让原模型反向导入；不重构公共工具。
- 不为 CPU 模型测试修改原模型中的 CUDA 分配或设备迁移行为。
- 不通过 monkey patch、替换模块、运行时注入 Dataset 类或改写原注册项间接改变原有流程。
- 不为本次接入升级或重装共享环境依赖。

发现原代码问题时，只记录现状及影响，不能将处理原问题并入数据集接入。若需求无法在新增适配层解决，应说明具体冲突，不自行放宽冻结边界。

### 1.2 允许的工作

只允许新增第 9 节列出的目标数据集专用文件，并修改本文。数据软连接统一使用 `data/PandaSet`、`data/DDAD`；按用户明确要求，根目录 `.gitignore` 仅新增 `/data/PandaSet`、`/data/DDAD` 两条精确规则，不忽略整个 `data/`，也不改变其他规则。新模块可以调用原模型和已有计算接口，但原入口不能导入或依赖新模块。

本次实验以零样本推理为重点：加载其他数据集上预训练的 **OmniScene 权重**，在 PandaSet/DDAD 上评估，不在目标数据上优化模型。训练仅补齐与原 OmniScene 设置一致的配置及必要的数据接口适配，不开发新的通用训练框架。

## 2. 需求与当前基线

### 2.1 已确认需求

| 要求 | 本版约定 |
| --- | --- |
| 原有功能 | OmniScene 风格 nuScenes 的数据加载、训练、推理保持原代码原样 |
| 数据入口 | `data/PandaSet`、`data/DDAD` 软连接，与原 `data/nuScenes` 同目录 |
| 预处理 | 复用已有 RGB、内外参、Metric3D-v2 尺度深度及置信度 |
| 完整配置 | 每个目标数据集提供 112×200、224×400 配置，继承对应原 nuScenes 配置 |
| 输入输出 | 同一帧六路输入，重建相同六路目标图像 |
| 动态物体 | 不生成、不加载动态掩码，全部视为静态数据 |
| PCC | 无需正式报告；需要计算时直接使用已有 Metric3D-v2 尺度深度 |
| 数据划分 | 新数据集只支持 `train`、`val`、`test`，逐项对齐 SVF-GS 对应 Dataset split |
| 评估协议 | `all_6` 输入视角重建，不称为新视角生成质量 |
| GPU | 用户明确允许后，才进行模型加载、前向或训练的 GPU 测试 |

### 2.2 当前状态核对

- 当前原有运行代码和配置与冻结基线无差异；`.gitignore` 仅新增已授权的两条数据入口规则。
- 两个目标数据软连接已移至 `data/PandaSet`、`data/DDAD`，指向原共享数据；原 `data/nuScenes` 保持不变，旧的空 `datasets/` 目录已移除。
- 新实现仅位于 `cross_dataset/`、`configs/PandaSet/`、`configs/DDAD/` 和 `configs/ZeroShot/`；`docs/` 中在本文记录本次实际验收结果。
- 本机存在 `checkpoints/checkpoint-100000/model.safetensors`，已在 CPU 上读取元信息并计算 SHA-256。本版尚未构造原样模型做严格加载验证；该文件无来源元数据，真实训练源域和分辨率不能仅凭文件名或参数形状认定。

原模型消费 `inputs`、`inputs_pix`、`inputs_vol`、`outputs`。`forward_test()` 与渲染器按实际目标相机数处理输出，从接口上可承接六路目标，因此首先通过新 Dataset 满足已有模型接口，不修改网络。

### 2.3 两条互不接入的执行路径

```text
原有路径：原 train.py / evaluate.py / demo.py
            → 原 nuScenes Dataset
            → 原模型、损失、优化器、渲染器
            → 原有行为与输出

新增路径：cross_dataset.train / cross_dataset.evaluate
            → 新 PandaSet / DDAD Dataset
            → 调用原样 OmniScene 模型
            → 目标域训练产物 / all_6 零样本评估结果
```

新增入口只接受 PandaSet/DDAD；误传 nuScenes 配置时明确报错，避免把新入口变成原流程的替代实现。

## 3. 数据路径与预处理复用

### 3.1 数据目录

```text
data/                              # 原有数据代码原样保留，此处仅展示软连接
├─ nuScenes -> /home/B_UserData/dongzhipeng/Datasets/dataset_omniscene
├─ PandaSet -> /home/B_UserData/dongzhipeng/Datasets/PandaSet
│  └─ processed/
│     ├─ bins_train.json
│     ├─ bins_test.json
│     ├─ bin_infos/<bin_token>.pkl
│     ├─ images_small/<sequence>/<camera>/<frame>.jpg
│     ├─ params_small/<sequence>/<camera>/<frame>.json
│     └─ dptm/<sequence>/<camera>/<frame>_{dpt,conf}.npy
└─ DDAD -> /home/B_UserData/dongzhipeng/Datasets/DDAD
   └─ processed/
      ├─ bins_train.json
      ├─ bins_test.json
      ├─ bin_infos/<bin_token>.pkl
      ├─ images_small/<scene>/<camera>/<frame>.jpg
      ├─ params_small/<scene>/<camera>/<frame>.json
      └─ dptm_small/<scene>/<camera>/<frame>_{dpt,conf}.npy
```

目标配置提供 `processed_root`，分别为 `data/PandaSet/processed` 和 `data/DDAD/processed`。仅在新加载器中按项目根目录解析相对路径。两个新软连接与原 nuScenes 入口统一放在 `data/`，不修改共享文件，不提交数据或机器相关软连接。Git 忽略规则使用 `/data/PandaSet`、`/data/DDAD`，不以尾部斜杠限定为真实目录，确保软连接本身也被忽略。

### 3.2 历史路径重定位

PKL 的 `data_path` 可能为 `data/PandaSet/processed/...`、`data/DDAD/processed/...`，也可能带有其他检出目录的历史路径。即使相对路径与当前入口一致，新加载器也统一按配置定位，负责：

1. 识别 `images_small` 下的 sequence/scene、camera、frame 相对路径。
2. 在当前 `processed_root` 下定位 RGB。
3. 由相同标识定位内参、尺度深度和置信度。
4. 对未知路径布局、缺失文件和元数据不一致，报告 dataset、bin token、camera 及具体路径。

不修改 PKL，不修改原 nuScenes 路径处理，不依赖切换到 SVF-GS 工作目录。SVF-GS 仅作为协议和验证参考，不作为新加载器的运行依赖。

### 3.3 图像、深度与内参

- 复用 224×400 预处理产物，不重新裁剪数据或在线运行深度模型。
- 112×200 的 RGB 缩放及深度/置信度插值与参考加载器一致；224×400 使用现有尺寸。
- 逐图读取内参，根据实际图像缩放比例调整 `fx, fy, cx, cy`，检查 RGB/深度/置信度尺寸对应。
- 深度保持 Metric3D-v2 尺度和相机轴向深度含义，不归一化、不替换成单位射线距离。
- 不读取动态分割文件，不创建无用途的全 1 动态掩码。
- `outputs.depth` 与 `outputs.depth_m` 使用同一份尺度深度，避免新增相对深度依赖。

```text
sx = target_width / source_width
sy = target_height / source_height
fx' = fx * sx, cx' = cx * sx
fy' = fy * sy, cy' = cy * sy
```

## 4. 仅 train/val/test 的数据接口

### 4.1 目标 Dataset 工厂

在 `cross_dataset/datasets/` 新增 `PandaSetDataset`、`DDADDataset` 和仅供二者使用的 `build_target_dataset(config, split)`。新训练、评估入口使用该工厂，不修改原数据文件或包初始化。

新增数据模块导入时不构造模型、不初始化 CUDA。必要的图像格式和几何计算小函数放在新模块中，逐项核对原公式，不抽走原函数或改变原导入路径。

### 4.2 split 对应规则

| 新 Dataset split | SVF-GS Dataset split | 索引来源 | 取样方式 | shuffle |
| --- | --- | --- | --- | --- |
| `train` | `train` | `bins_train.json` | 全部 token | 训练 DataLoader 开启 |
| `val` | `val` | `bins_test.json` | `np.linspace(0, N - 1, 10, dtype=int)` 对应的 token | 否 |
| `test` | `test` | `bins_test.json` | 全部 token，保留原顺序 | 否 |

**新入口只支持上述三个名称。** 不引入 `total`、`mini`、`mini-test`、`demo` 或 `center150` 等别名和额外划分，不根据参考项目的历史命令缩减测试集。原入口已有功能保持原样。

本次重新读取索引的结果：

| 数据集 | train | val | test | train/test 关系 |
| --- | ---: | ---: | ---: | --- |
| PandaSet | 3120 | 10 | 3120 | 两份索引集合相同 |
| DDAD | 1265 | 10 | 395 | 两份索引不相交 |

val 是参考协议从 test 抽取的十项，不另造独立验证集。PandaSet 的划分重合应在目标域训练记录中如实说明，不能将其解释为独立留出集。零样本实验的关键是源权重未在目标域训练。

校验源索引非空、token 唯一、取样位置合法。当前测试集足够生成十个不同 val 样本；其他数据副本若不满足条件，明确报错，不静默改变规则。

开发检查可以显式使用 `--max-samples N` 限制所选 split 的前 N 项，但它只是调试限制，不新增 split，不能把结果标记为完整 split。

### 4.3 六视角组织

固定保留 SVF-GS 的槽位顺序：

```text
CAM_FRONT, CAM_FRONT_RIGHT, CAM_FRONT_LEFT,
CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT
```

每个槽位读取 `sensor_info[cam][0]`，输出与输入使用同帧同一组六个相机。配置为 `only_input=True`、`use_center=True`、`use_first=False`、`use_last=False`。不拼接 nuScenes 的十二个邻帧目标；不受支持的邻帧配置明确报错。

### 4.4 Batch 字段

以下为 DataLoader 拼成 batch 后的形状，`B` 为每进程 batch size，`H,W` 为加载分辨率：

| 字段 | 形状/类型 | 含义 |
| --- | --- | --- |
| `bin_token` | 长度 B 的字符串列表 | 样本唯一标识 |
| `inputs.rgb` | `[B,6,3,H,W]` | `[0,1]` 范围 RGB |
| `inputs_pix.depth_m` | `[B,6,H,W]` | Metric3D-v2 尺度深度 |
| `inputs_pix.conf_m` | `[B,6,H,W]` | 深度置信度 |
| `inputs_pix.ck` | `[B,6,3,3]` | 缩放后的内参 |
| `inputs_pix.c2w` | `[B,6,4,4]` | 与本项目射线约定一致的相机位姿 |
| `inputs_pix.fx/fy/cx/cy` | `[B,6]` | 焦距及主点 |
| `inputs_pix.rays_o/rays_d` | `[B,6,H,W,3]` | 射线原点和非归一化方向 |
| `inputs_vol.w2i` | `[B,6,4,4]` | Volume-GS 使用的世界坐标到图像投影 |
| `outputs.rgb` | `[B,6,3,H,W]` | 与输入相同的目标 RGB |
| `outputs.depth/depth_m/conf_m` | `[B,6,H,W]` | 尺度深度参考及置信度 |
| `outputs.c2w` | `[B,6,4,4]` | 目标相机位姿 |
| `outputs.fovx/fovy` | `[B,6]` | 目标视场角 |
| `outputs.rays_o/rays_d` | `[B,6,H,W,3]` | 目标射线 |

动态物体掩码字段不作为新 Dataset 的必需输出，也不创建无实际用途的全 1 掩码张量。

## 5. 几何约定与 DDAD 核验项

### 5.1 保持两条几何链一致

Pixel-GS 和 Volume-GS 都依赖几何数据，不能只验证 RGB 和深度形状。

以预处理中的相机到参考坐标变换 `T_cv` 为起点：

```text
F = diag(1, -1, -1, 1)
c2w_gl = T_cv @ F
rays_d_camera = [(u + 0.5 - cx) / fx, -(v + 0.5 - cy) / fy, -1]
world_point = rays_o + rays_d * depth_m
w2i = K4 @ inverse(T_cv)
```

其中 `K4` 为嵌入 4×4 矩阵的内参。现有 `load_info()` 返回的 `w2c` 使用转置存储方式，调用链中的 `viewpad @ w2c.T` 与上式对应；不能对返回值重复转置或重复翻转 Y/Z 轴。

应验证：

- 深度反投影后的点再用 `w2i` 投影，能回到原像素中心。
- 相机前方点的投影深度为正。
- Pixel-GS 射线、Volume-GS 可见性以及渲染器使用同一空间坐标。
- 预处理主点居中约定与基于 FOV 的渲染接口一致。
- 保持源模型的空间范围、体素结构和尺度参数，不对目标域单独调参。

### 5.2 DDAD 已观察到的差异

此前只读抽查现有数据得到以下观察；本版实施时重新核对：

- nuScenes、PandaSet 样本的前向大致为参考坐标 `+Y`；DDAD 样本前向大致为 `+X`。
- DDAD 当前映射为 `CAM_BACK_RIGHT → CAMERA_09`，但三个抽查样本中，`CAMERA_09` 相对前向的水平夹角约为 177°，接近正后方。
- `CAM_BACK → CAMERA_07` 的对应夹角约为 117°～120°。

这说明槽位命名需要结合实际标定核对。Omni-Scene 包含学习到的相机嵌入和 Volume-GS 空间结构，相关差异可能影响迁移结果。

本次实施先忠实复现 SVF-GS 当前数据约定，并保存明确的相机顺序和坐标说明。若后续校正坐标或槽位，应两个项目共同采用相同规则、重新评估并单独标记协议，不能将校正后的结果与旧协议结果直接混用。

## 6. 完整实验配置：继承原设置，不改原设置

### 6.1 配置层次

采用现有 MMEngine 配置机制，仅新增文件：

```text
原 configs/OmniScene/omni_gs_nusc_novelview_r50_<resolution>.py（只读）
  → configs/PandaSet/ 或 configs/DDAD/ 的完整目标配置
    → configs/ZeroShot/ 的源域到目标域零样本配置
```

拟定八份配置：

```text
configs/PandaSet/omni_gs_pandaset_r50_112x200.py
configs/PandaSet/omni_gs_pandaset_r50_224x400.py
configs/DDAD/omni_gs_ddad_r50_112x200.py
configs/DDAD/omni_gs_ddad_r50_224x400.py
configs/ZeroShot/omni_gs_nusc_to_pandaset_r50_112x200.py
configs/ZeroShot/omni_gs_nusc_to_pandaset_r50_224x400.py
configs/ZeroShot/omni_gs_nusc_to_ddad_r50_112x200.py
configs/ZeroShot/omni_gs_nusc_to_ddad_r50_224x400.py
```

完整配置只覆盖 Dataset 身份、`processed_root`、六路同帧数据约定、实验名称、工作/输出目录及目标评估选项。零样本配置额外提供 `source_dataset`、`source_config`、显式检查点设置、`split='test'` 和结果目录。

源权重必须是本项目架构兼容的 OmniScene 模型，不能直接加载 SVF-GS 权重。首批提供 nuScenes 源域模板，其他源域可提供真实匹配的配置；每个源域和分辨率组合均需核实实际训练记录。

### 6.2 模型默认值

以下取自当前 nuScenes 配置，作为两目标数据集对应配置的默认继承基准。

| 模块 | 默认配置 |
| --- | --- |
| 图像 Backbone | `mmdet.ResNet`，depth 50，4 个 stage 输出，DINO ResNet-50 初始化，`norm_eval=True` |
| Neck | `mmdet.FPN`，输入通道 `[256,512,1024,2048]`，输出通道 128，4 个尺度 |
| Pixel 编解码模块 | `MVDownsample2D`、`MVMiddle2D`、`MVUpsample2D`，配置 `num_layers=1`，8 个 attention heads |
| Pixel 通道 | `[128,256,512,512]` |
| Pixel 压缩比例 | `patch_sizes=[8,8,4,2]` |
| 相机数量 | 6 |
| TPV 网格 | `192×192×16` |
| Volume encoder | `TPVFormerEncoder`，3 层，1 个特征层级，embed dim 128 |
| Encoder 层序列 | 两层 hybrid self-attention + image cross-attention，再接一层 hybrid self-attention |
| Encoder FFN | 256 通道，dropout 0.1 |
| Pillar / deformable sampling | `num_points_in_pillar=[8,16,16]`，`num_points=[16,32,32]` |
| Hybrid attention | 16 anchors，32 points，初始化模式 0 |
| 位置编码 | `TPVFormerPositionalEncoding`，`num_feats=[48,48,32]` |
| Volume decoder | `VolumeGaussianDecoder`，输入/隐藏/输出通道 `128/256/128`，高斯参数维数 14 |
| 每体素高斯数 | `gpv=3`，三个轴的网格放大系数均为 1 |
| Decoder 偏移/尺度上限 | 按原配置的空间范围、网格尺寸和放大系数公式计算 |
| 空间范围 | `[-50,-50,-3,50,50,12]` |
| 近远平面 | `0.1 / 1000.0` |
| 激活检查点 | `use_checkpoint=True`，沿用原模型行为 |

保留 `model` 中模块类型及完整参数配置能力，不引入 SVF-GS 的 CADGA/EAGR 模块。不同模块选择必须由当前注册实现实际支持，不能只新增一个未被执行读取的开关。

### 6.3 训练配置值与实际执行语义

**配置字段和原入口的实际用法都需要对齐，不得将字段名称自行解释成新的训练规则。**

| 项目 | 原配置/实际行为；目标训练沿用 |
| --- | --- |
| train/val/test batch size | 每进程 1 |
| train/val/test workers | 均为 8 |
| 主基础学习率 | 顶层 `lr=1e-4` |
| 优化器 | 调用原 `configure_optimizers(cfg.lr)`：AdamW，backbone 倍率 0.1 |
| AdamW 参数 | betas `(0.9,0.999)`，eps `1e-8`，weight decay `0.01` |
| 裁剪/累积/精度 | `grad_max_norm=1.0`、`gradient_accumulation_steps=1`、`mixed_precision='no'` |
| 种子/记录器 | `seed=0`；原进程种子规则；`report_to='tensorboard'` |
| 循环停止条件 | `max_epochs=30`，原代码按 epoch 结束 |
| `max_train_steps` | 5000；参与调度参数和训练前向 `iter_end`，原代码不据此停止循环 |
| `warmup_steps` | 1000；原调度构造还使用进程数倍率，详见第 7 节 |
| `lr_scheduler_type` | 保留原字符串 `constant_with_warmup`；原执行实际创建 LinearLR + CosineAnnealingLR |
| 日志/验证/保存频率 | `print_freq=100`、`val_freq=500`、`save_freq=2000`，沿用原 `global_iter` 判断位置 |
| 验证调用 | 沿用原 `validation_step(batch, directory)`，不增加参数或装饰器 |
| 保存与恢复 | 沿用原 Accelerate 状态格式与 `resume_from='latest'`，不新增强制元数据格式 |

原 `_base_/optimizer.py` 中的 `lr=5e-5` 和 `paramwise_cfg` 保留原样，它们没有被原训练入口用来创建优化器。目标入口也调用原模型方法，不用新的 optimizer 构建器替换这条链路。

默认总迭代次数取决于 epoch 数和准备后的 DataLoader 长度。从头训练、单进程、batch size 1、梯度累积 1 时，PandaSet 为 `30×3120=93600` 次 batch 迭代，DDAD 为 `30×1265=37950` 次；其他设置按实际 loader 和原计步规则记录。**默认不承诺在 5000 次更新停止，也不把 `max_epochs` 改为备用字段。**

### 6.4 损失与静态数据

| 损失项 | 原默认类型/权重 |
| --- | --- |
| 融合 RGB | L2 / 1.0 |
| 融合感知 | 原 LPIPS / 0.05 |
| 融合尺度深度 | 置信度加权绝对误差 / 0.01 |
| Volume RGB | `l2_mask` / 1.0 |
| Volume 感知 | 原空间掩码处理 / 0.05 |
| Volume 尺度深度 | `mask`、置信度加权 / 0.01 |

全部对象视为静态，不加入动态筛选。原模型的空间范围 `mask_dptm` 和 Metric3D 置信度属于几何及深度监督处理，保持原样，不与动态掩码混为一谈。

Volume 辅助监督条件原本是 `iter < iter_end - 1000`。训练传入 `iter_end=cfg.max_train_steps`，默认门限为 `global_iter < 4000`；由于真实训练按 epoch 结束，不能描述成“整个训练最后 1000 步才关闭”。验证使用原方法的默认前向参数，不自行传入新的训练进度。

### 6.5 新配置的静态检查

配置检查只作用于新增入口持有的配置对象：

- 对应分辨率下，目标 Backbone、Pixel-GS、Volume-GS、camera/loss 参数与原基准一致。
- 核对 `dataset_params` 与 `model.dataset_params` 的目标身份和模型消费字段；变化限于必要的数据字段。
- 核对加载、渲染、感知损失分辨率及 TPV、attention、位置编码、decoder 维度一致。
- 检查 batch size、worker、数据路径和三种 split 的合法性。
- 保存完整解析配置，明确哪些训练字段由原模型方法固定，哪些由入口消费。

MMEngine 子配置覆盖不会自动重新计算父配置里的 Python 派生字典。优先继承对应分辨率的父配置；需要目标字段覆盖时显式同步必要嵌套字段并校验，不修改父配置，不建立所有数据集共用的新配置系统。模型支持范围以原实现为准，不以新增开关扩展原模型。

## 7. 仅新增目标训练入口的兼容适配

### 7.1 最小适配方式

新增 `cross_dataset/train.py`，以冻结版根目录 `train.py` 为参照建立目标专用入口，保持可审查的对应关系；不重写成通用 trainer，不替代原入口。

必要差异限定为：

1. 读取新增目标配置，仅允许 PandaSet/DDAD。
2. 使用新 Dataset 工厂构造 `train`、`val`，输出六视角 batch。
3. 指定目标实验独立目录，避免接续或覆盖原 nuScenes workdir。
4. 对新入口自身的路径或单卡模型未包装 `.module` 等访问差异做局部兼容，不改原方法或训练算法。

其余训练主体以原执行行为为基准，包括模型构建、优化器调用、DataLoader shuffle、Accelerate 参数、zero_grad/backward/step 顺序、scheduler 调用、种子、验证、保存与恢复。若需保护配置快照，可在新调用端向原 builder 传入配置副本，不能修改 builder。

### 7.2 必须保留的调度与计步

设进程数为 `P`，原代码构造：

```text
LinearLR:
  start_factor = 1 / (warmup_steps * P)
  end_factor = 1
  total_iters = warmup_steps * P

CosineAnnealingLR:
  T_max = max_train_steps * P
  eta_min = lr * 0.1

SequentialLR:
  milestone = warmup_steps * P
```

目标入口保留这些公式及原 Accelerate scheduler 包装/推进方式，不去掉进程数倍率，不改 cosine 周期，不改变 `step_scheduler_with_optimizer` 默认行为。

原 `global_iter` 每个 batch 结束递增，保存/验证判断在递增前、优化器调用后执行。从头训练、单进程且累积为 1 时，第一个 batch 后即以 `global_iter=0` 触发相应动作；后续在 500、1000 等标签验证，在 2000、4000 等标签保存。保留这些标签的含义，不改为“已完成优化器更新数”，不增加原代码没有的最终验证/保存。

### 7.3 训练兼容验收边界

- 对新入口与原训练主体做静态及轻量调用顺序对照；差异必须能归入第 7.1 节。
- 配置完整覆盖原模型、损失、学习率、分辨率、batch size、epoch/迭代相关字段和验证保存频率。
- GPU 允许后，验证六视角数据可进入原模型并得到损失、梯度和训练产物。
- 使用短 epoch 配置或独立测试夹具验证兼容性，不给正式入口增加新的总步数停止规则。
- 沿用原恢复方式，不新增 `training_state.json` 作为必须条件，不宣称实现了精确的数据位置恢复。
- 原代码计算 `first_epoch/resume_step` 后未完整用于后续循环等已知限制，写入说明，不顺便重构恢复流程。
- 非默认梯度累积、多卡和恢复的可靠性只报告实际测试结论。超出接口适配的训练逻辑问题单列，不并入本次开发。

## 8. 独立零样本推理与评估

### 8.1 新入口流程

新增 `cross_dataset/evaluate.py`，与原 `evaluate.py` 完全分离：

```text
读取目标配置及源模型说明
  → 检查数据入口、split、显式源检查点
  → 构建目标 Dataset / DataLoader
  → 调用原 builder 构建原样 OmniScene 模型
  → 严格加载源域模型权重
  → eval + no_grad 调用原 forward_test()
  → 六视角原位重建指标
  → 按 bin 汇总、检查覆盖、保存独立结果
```

默认 `--split test`，可选择 `train` 或 `val` 做诊断并明确记录。新推理入口不构造训练优化器、不反向传播、不微调、不恢复源数据迭代进度。零样本配置不能交给目标训练入口执行。

### 8.2 源权重与模型兼容性

- 显式 `--load-from` 或新配置的明确路径为必需项；缺权重直接报错，不使用随机初始化继续评估。
- 支持目录中的 `model.safetensors` 或 `pytorch_model.bin`，只加载模型参数。
- 严格检查状态项和形状，正确处理 safetensors 共享参数别名，不靠 `strict=False` 忽略架构不匹配。
- 源域与目标域须不同；源模型架构、分辨率及空间范围等需与实验设置相符。
- 记录检查点绝对路径、权重摘要、源配置、源域声明和实际可用的来源证据。
- 旧权重没有来源元数据时如实记录；严格加载成功不能单独证明训练数据来源或分辨率。
- 不改写原检查点、模型构造代码或原评估加载逻辑。

### 8.3 指标定义

正式记录 PSNR、SSIM、LPIPS，原指标函数只读复用。目标新入口按如下顺序聚合：逐视角计算 → 六路平均为 bin 的 `all_6` → 对所选 split 的唯一 bin 等权平均。

新目标配置默认 `eval_args.compute_pcc=False`。需要 PCC 时调用已有函数，以 Metric3D-v2 深度为参考，按每个 bin 的六视角联合计算；不新增相对深度或动态掩码依赖。该开关和深度选择不影响原 nuScenes 入口。

### 8.4 batch 与多进程处理

只在新增评估代码中实现：

- 按实际 batch size 和六个目标视角处理预测。
- 用 `accelerator.unwrap_model()` 访问原推理方法，适配新入口单/多卡包装。
- 聚合逐 bin 记录，去除分布式尾批补齐重复；异常重复和覆盖缺失明确报错。
- 按唯一样本数汇总，不将大小不同的 batch 均值等权平均。
- 可视化与 PLY 按唯一 bin 保存，独立创建样本目录，避免重复覆盖。

这些要求仅用于目标数据集新入口，不应用到原评估脚本，不调整其 nuScenes 指标分组或产物。

### 8.5 结果记录

```text
outputs/zero_shot/nusc_to_pandaset_r50_112x200/test/checkpoint-100000/
├─ resolved_config.py
├─ evaluation_manifest.json
├─ per_bin_metrics.csv
├─ evaluation_summary.json
└─ visualizations/                 # 配置开启后保存
```

manifest 至少记录源/目标数据集、源配置、检查点摘要、冻结基线与当前代码版本、split、数据根目录、分辨率、相机顺序、输入/输出数量、协议、预期 token、实际覆盖、样本限制、指标定义及完成状态。

PandaSet `test` 对应 3120 个 bin，DDAD `test` 对应 395 个 bin。DDAD 历史部分样本结果不能冒充完整 test 结果；方法比较时应让参考项目使用相同的 `test` 索引和分辨率。

限量运行只标记完成受限选择，不标记完整 split 完成。失败、缺权重、覆盖不全或中断时，不留下全量完成声明。结果目录与源检查点、原 nuScenes 输出目录分离。

## 9. 允许新增及修改的文件清单

以下为本次新增实现；原有文件仅修订本文，`.gitignore` 保持此前已授权的两条新增规则，不再扩大改动：

```text
cross_dataset/
├─ __init__.py                      # 不导入或初始化模型
├─ datasets/
│  ├─ __init__.py                   # 只导出目标 Dataset 与工厂
│  ├─ pandaset.py
│  ├─ ddad.py
│  └─ common.py                     # 目标路径、图像、深度、几何与打包
├─ configuration.py                # 只检查新配置，拒绝 nuScenes 目标
├─ checkpoint.py                   # 目标评估的严格加载与来源记录
├─ evaluation.py                   # 目标指标汇总与覆盖检查
├─ ply.py                          # 原 PLY 导出的限定副本，不修改原函数
├─ evaluate.py                     # 新目标评估入口
├─ train.py                        # 原训练主体的最小目标数据适配
└─ tests/
   ├─ test_data_cpu.py
   ├─ test_config_cpu.py
   ├─ test_evaluation_cpu.py
   ├─ test_training_compat_cpu.py
   ├─ check_baseline_unchanged.py
   ├─ cpu_only.py                  # 仅测试使用的 CUDA 初始化阻断
   └─ distributed_evaluation_cpu.py # CPU/Gloo 双进程汇总夹具

configs/PandaSet/                   # 两档新增完整配置
configs/DDAD/                       # 两档新增完整配置
configs/ZeroShot/                   # 四份新增零样本配置
data/PandaSet                      # 数据软连接，不提交
data/DDAD                          # 数据软连接，不提交
.gitignore                         # 仅新增 /data/PandaSet 和 /data/DDAD
docs/PandaSet与DDAD零样本泛化及训练评估接入方案.md
```

测试和辅助代码集中在新增目录，不向原 `tools/`、`data/`、`model/` 增加供原路径自动发现的实现。每个新文件都必须服务于两目标数据集的本次需求，不加入通用 optimizer、scheduler、精确恢复框架或 nuScenes 评估重构。

如需拆分新 `common.py` 等文件，可在 `cross_dataset/` 内调整，但不得扩大到冻结范围。共享数据、权重、软连接不纳入源码提交。

## 10. 拟定运行方式

以下接口已实现，但这些模型 GPU 命令本次均未执行。模型运行仍等待用户明确允许；设备编号按实际获准资源选择。

从项目根目录使用模块入口，不修改原启动脚本：

```bash
conda activate omniscene

# PandaSet：确认源权重及其来源后，执行完整 test。
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config-file accelerate_config.yaml --module cross_dataset.evaluate \
  --py-config configs/ZeroShot/omni_gs_nusc_to_pandaset_r50_112x200.py \
  --load-from checkpoints/checkpoint-100000 \
  --split test \
  --output-dir outputs/zero_shot/nusc_to_pandaset_r50_112x200/test/checkpoint-100000

# DDAD：完整 test 为 395 个样本。
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config-file accelerate_config.yaml --module cross_dataset.evaluate \
  --py-config configs/ZeroShot/omni_gs_nusc_to_ddad_r50_112x200.py \
  --load-from checkpoints/checkpoint-100000 \
  --split test \
  --output-dir outputs/zero_shot/nusc_to_ddad_r50_112x200/test/checkpoint-100000

# 目标域训练兼容入口：使用完整配置，沿用原训练语义。
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --config-file accelerate_config.yaml --module cross_dataset.train \
  --py-config configs/PandaSet/omni_gs_pandaset_r50_112x200.py \
  --work-dir workdirs/pandaset_r50_112x200
```

DDAD 训练选择对应完整配置及独立 workdir；224×400 使用对应配置与真实匹配的源权重。`--split val` 选择协议中的十个验证样本；单样本检查用 `--split test --max-samples 1` 和独立目录。

新增评估入口要求输出目录为空，防止与旧评估或训练产物混合。训练兼容入口允许恢复已有目标 workdir，但要求其中已有匹配的目标配置快照；不引入额外训练状态格式。训练主体的原恢复限制仍按第 7 节保留。

零样本配置默认声明源域为 `nuScenes`，可用 `--source-dataset`、`--source-config` 指定其他匹配的源模型配置。声明与严格参数加载均不能单独证明权重来源。可通过 `--source-metadata` 提供 JSON，或在权重同目录放置 `source_metadata.json`，包含：

```json
{
  "checkpoint_sha256": "实际模型权重文件的 SHA-256",
  "source_dataset": "nuScenes",
  "resolution": [112, 200]
}
```

元数据字段必须与本次权重摘要和源域/分辨率声明一致，记录为 `hash_bound_metadata_provided`；这表示提供了与权重绑定的来源说明，不代表独立审计了原训练过程。未提供时记录 `unverified_legacy_checkpoint`。PCC 按配置中的 `eval_args.compute_pcc` 控制，默认关闭；模型、损失和训练超参数仍继承对应原配置。

`PYTHONNOUSERSITE=1` 仅作用于新命令进程，避免用户级包覆盖 Conda 包；不修改原命令、共享依赖或全局设置。新入口不接管原 nuScenes 命令，不承诺让原入口接受新数据集配置。

## 11. 分阶段开发与验收

### 阶段 0：冻结基线

- 记录基线提交、受跟踪文件清单和内容哈希，确认原有运行代码和配置无暂存或工作区差异。
- 允许新增路径限定为第 9 节，禁止改动其他基线文件；例外仅为本文和 `.gitignore` 中已授权的两条精确规则。
- 核对原配置和包导入关系，新增功能不能被原入口自动加载。
- CPU 阶段不运行完整模型构造或 CUDA 初始化。

### 阶段 1：目标数据加载

- 新增两个 Dataset、三个 split 和目标工厂。
- 逐 token 对齐参考 `train/val/test`，核对顺序、数量及重叠关系。
- 检查全部索引对应的必需文件是否存在。
- 两数据集、两分辨率抽首/中/末样本，对照 RGB、尺度深度、置信度、K、姿态、射线和 `w2i`。
- 验证 batch 契约、输入输出同帧、投影往返误差和缺文件报错。
- 动态掩码与相对深度文件不存在时，加载仍正常。

### 阶段 2：配置与训练兼容

- 新增八份配置；与原对应配置相比，仅有必要的数据、实验、评估字段差异。
- 原模型、损失权重、优化器调用、调度公式、实际停止与验证保存语义不重新定义。
- 新训练入口与原主体的差异限于接口适配，不引入新采样器、通用训练引擎或强制检查点格式。
- CPU 轻量夹具核对接口、公式与调用顺序，不据此宣称完整原样网络已经跑通。

### 阶段 3：目标评估 CPU 检查

- 验证三个 split、权重路径错误、来源声明及受限选择处理。
- 用合成预测验证 `all_6` 汇总、不同 batch size、尾批、去重和覆盖检查。
- 可读取权重文件的键、形状等元信息；不为严格模型加载测试改变原 CUDA 构造方式。
- CPU 导入若受模型依赖阻碍，在新模块中延迟导入，不重构原模型/数据包。

### 阶段 4：用户允许后进行 GPU 验证

- 原样模型严格加载源权重，核对架构、分辨率、预训练来源。
- 各目标单样本前向，检查六路输出、深度、高斯、有限指标及可视化。
- 用明确标记的受限 test 检查 batch size、多进程处理，再执行完整 test。
- 独立短实验验证目标训练接口、原损失及反向传播，不为测试改写正式训练规则。
- 未执行内容标注待验证，不提前承诺真实模型稳定性或泛化指标。

### 阶段 5：原流程不可变验收

最高优先级条件：**基线中的原有运行代码、配置保持一致，原命令与导入路径不变。** 文档修订与 `.gitignore` 中已授权的两条规则单独核验。

- 检查冻结文件无修改、删除、重命名或模式变化，覆盖暂存区与工作区；`.gitignore` 仅允许两条目标数据入口规则，不得扩大忽略范围。
- 新文件仅出现在允许目录，不存在覆盖原模块的同名导入或运行时替换。
- 对冻结文件做内容对比，不能仅以旧数据抽样数值相同替代零差异检查；确认三条数据软连接的目标正确，原 `data/nuScenes` 未变。
- 每阶段执行边界检查；若发现违规，先撤销该阶段开发造成的越界改动，再继续。
- 交付时分别说明 CPU 检查、GPU 模型验证和正式实验的实际状态。

## 12. 结果解释与排除事项

PandaSet/DDAD 输入与输出相同，只报告 `all_6` 输入视角重建。不能替代原 nuScenes 的新视角结果，不把新入口的指标聚合写回原评估代码。

方法对比固定目标 split token、相机顺序、分辨率、预处理产物、源权重身份及聚合方式。完整 DDAD test 的 395 个样本与历史部分样本结果分别标记。

明确排除：原 OmniScene 数据或训练/推理修复、公共配置治理、优化器更换、调度纠正、预算重定义、恢复框架重写、原模型 CPU 化、动态掩码生成、额外相对深度推理、SVF-GS 模型组件迁移，以及未获 GPU 授权的模型实验。

## 13. 只读代码依据

以下链接用于理解和核对已有实现，不表示允许修改这些文件。

### 本项目

- [nuScenes 数据加载](../data/dataloader.py)
- [相机信息与条件数据读取](../data/transforms/loading.py)
- [112×200 实验配置](../configs/OmniScene/omni_gs_nusc_novelview_r50_112x200.py)
- [224×400 实验配置](../configs/OmniScene/omni_gs_nusc_novelview_r50_224x400.py)
- [训练入口](../train.py)
- [评估入口](../evaluate.py)
- [模型、优化器、损失和验证前向](../model/omni_gs.py)
- [Volume encoder 投影](../model/volume/tpvformer_encoder.py)
- [高斯渲染器](../model/gaussian.py)
- [射线与相机工具](../model/utils/ops.py)
- [指标定义](../tools/metrics.py)

### SVF-GS 参考项目

以下相对链接假设两个仓库同位于 `~/Projects`：

- [PandaSet Dataset](../../SVF-GS/data/pandaset_dataset.py)
- [DDAD Dataset](../../SVF-GS/data/ddad_dataset.py)
- [PandaSet 条件数据读取](../../SVF-GS/data/transforms/pandaset_loading.py)
- [DDAD 条件数据读取](../../SVF-GS/data/transforms/ddad_loading.py)
- [DataModule 划分规则](../../SVF-GS/data_module.py)
- [配置构建](../../SVF-GS/configs/build_config.py)
- [PandaSet 预处理](../../SVF-GS/scripts/preprocess_pandaset.py)
- [DDAD 预处理及相机映射](../../SVF-GS/scripts/preprocess_ddad.py)
- [检查点来源与身份校验](../../SVF-GS/tools/experiment_state.py)
- [评估汇总](../../SVF-GS/trainer.py)

## 14. 本次实现与 CPU 验收记录（2026-09-18）

### 14.1 已实现范围

- 两个独立 Dataset 与目标工厂，只支持 `train/val/test`，沿用已有预处理产物和参考相机顺序；PKL 图像路径按当前目标 `processed_root` 重定位，不依赖 SVF-GS 工作目录。
- 相同六路输入/输出，完全不读动态掩码或相对深度文件；`outputs.depth` 与 `depth_m` 使用相同 Metric3D-v2 尺度深度。原损失中的空间范围掩码、置信度处理保持原样。
- 四份完整目标配置、四份零样本配置，分别继承对应分辨率的原配置。除目标数据、实验目录和评估所需字段外，所有原配置项逐项一致。
- 新训练入口保留原优化器调用、调度公式、Accelerate 参数、epoch 停止、计步、验证保存及恢复主体。原训练循环与新循环的 AST 对比只存在两处 `.module` 访问兼容差异；入口层另外适配配置、Dataset、独立 workdir、配置副本和空恢复路径初始化。
- 新评估入口严格加载模型参数，支持 safetensors 共享参数别名与 PyTorch state_dict；不加载优化器或恢复训练进度。`all_6` 按唯一 bin 等权汇总，支持 PCC 开关、逐 bin 可视化和 PLY。
- 评估专用 `EvaluationShard` 显式标记尾批补齐记录，各真实索引只归属一个进程；补齐记录不计分、不写可视化，额外的非补齐重复或缺失 token 报错。此分片器只用于新评估，训练 DataLoader 没有引入新 sampler。
- 结果写入 `resolved_config.py`、`evaluation_manifest.json`、`per_bin_metrics.csv`、`per_view_metrics.json`、`evaluation_summary.json`，并区分 `running/complete/limited/failed`。失败或中断不会留下完整 split 的完成声明。

### 14.2 已执行的 CPU 检查

17 项 CPU 单元/集成检查通过，另完成一次真实 CPU/Gloo 双进程夹具检查。全部使用 `CUDA_VISIBLE_DEVICES=''`、`PYTHONNOUSERSITE=1`；测试阻断 CUDA 初始化，没有构造完整 OmniGaussian，也没有运行 GPU 模型实验。

| 检查 | 本次结果 |
| --- | --- |
| 原流程冻结 | 基线 52 个 Git blob 的工作区内容、暂存内容及模式核对通过；`.gitignore` 仅两条已授权规则例外，三条数据软连接目标保持一致 |
| 配置 | 八份配置可解析；所有原模型、损失、优化器配置、调度/训练、分辨率、batch、验证保存字段逐项继承一致 |
| 索引 | 两数据集三个 split 的 token 顺序与当前 SVF-GS 定义逐项一致；PandaSet 为 3120/10/3120，DDAD 为 1265/10/395（train/val/test） |
| 全部必需文件 | PandaSet 3120 个唯一 bin、78000 个逐 bin 必需文件；DDAD 1660 个唯一 bin、41500 个逐 bin 必需文件，均存在；索引 JSON 另已读取验证 |
| 数据抽样对齐 | 两数据集 × 两分辨率 × 三 split × 首/中/末，共 36 次样本对照；RGB、尺度深度、置信度、K、c2w、射线、w2i 最大绝对差均为 0；仅 PandaSet fovy 最大差为 `1.1920928955078125e-7` |
| 几何与原接口 | 抽样投影往返误差通过 `2e-3` 像素容差；原 `get_data/plucker_embedder` 方法在独立 CPU 数据夹具中接收两数据集 batch=2 的新 batch，未修改或实例化原模型 |
| 缺文件与静态协议 | 无动态掩码/相对深度的临时夹具正常加载；缺少尺度深度时报出数据集、split、token 和文件名；错误 split/时序设置拒绝 |
| 权重辅助函数 | 小型 CPU 模型严格加载、共享别名、架构不匹配拒绝、来源元数据检查通过；无权重时在 Accelerator/模型构造前报错 |
| 指标聚合 | batch=1/2/4、模拟进程数=1/2/3/8 的覆盖、尾批、重复、非有限值、限量标识检查通过；聚合结果不因分批方式变化 |
| CPU 双进程 | 5 个真实样本+3 个补齐样本，唯一覆盖为 5；限量 1 个样本+3 个补齐样本，标记 `limited`；跨进程错误收集通过 |
| 训练语义 | 原/新训练循环 AST、优化器/调度器/Accelerate 构造、轻量调用轨迹一致；轨迹验证 max_train_steps 不取代 epoch 停止，以及原迭代 0 的验证保存行为 |

指标夹具使用原 PSNR/SSIM 函数，LPIPS/PCC 使用轻量替身验证调用和聚合；不将其视为真实 LPIPS 网络或真实深度 PCC 的模型验收。小模型严格权重测试也不替代完整 OmniScene 模型的严格加载。

只读检查的现有权重：

```text
路径：checkpoints/checkpoint-100000/model.safetensors
大小：414320936 bytes
参数键数量：757
safetensors 元信息：{"format": "pt"}
SHA-256：02627a5f9fa9103ed0a129456905a71e8d364f472f0d389023a1a7297c1ada67
来源状态：unverified_legacy_checkpoint
完整原样模型严格加载：未执行
```

冻结基线文件哈希清单的综合 SHA-256 为 `dedae92770785d59ddca47566b454256802b4c93e35595335f8f6115d65838ae`。可用下面的边界检查命令重新验证，不修改暂存区。

### 14.3 CPU 复核命令

在项目根目录、`omniscene` Conda 环境执行：

```bash
# 原有文件、暂存区、模式与软连接边界。
python -m cross_dataset.tests.check_baseline_unchanged

# 全部 CPU 测试；真实数据对照需保留同级 SVF-GS 项目。
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s cross_dataset/tests -p 'test_*_cpu.py' -v

# CPU/Gloo 双进程夹具，不构造完整模型。
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 \
  ACCELERATE_USE_CPU=true ACCELERATE_TORCH_DEVICE=cpu \
  python -m torch.distributed.run --standalone --nproc_per_node=2 \
  --module cross_dataset.tests.distributed_evaluation_cpu
```

最后一条命令的 CPU 设备环境变量只用于本测试进程，兼容当前 Accelerate 1.7 与 PyTorch 2.1；没有改动共享依赖或正式训练配置。

### 14.4 仍待用户允许 GPU 后验证

完整原样模型的严格权重加载、两数据集真实前向、LPIPS 网络、可视化/PLY 数值质量、真实模型 batch/multi-GPU 行为、训练损失和反向传播，以及正式全量零样本指标，均未执行。原模型中显式 CUDA 构造保持不动，原训练多卡、梯度累积和恢复的已有行为也没有被本次适配修正。
