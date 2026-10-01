# PandaSet 与 DDAD 零样本泛化及训练评估接入方案

> 修订日期：2026-10-01。在 2026-09-23 完成 **6 路中央输入 → 18 路目标渲染** 与 DDAD 可选自车掩码适配的基础上，补充修正 DDAD 公共坐标系与 nuScenes 训练坐标系的对齐。
>
> **最高优先级原则不变：原有 OmniScene 的数据加载、训练、推理代码及配置绝对不可改动。保留当前 `cross_dataset` 独立架构，只适配目标数据加载、评估统计及必要的目标配置与测试。**
>
> **实施状态：已在独立目标代码中完成 18 视角加载、三组统计和 DDAD 可选掩码适配；仅进行小规模 CPU 验证，未运行完整模型或使用 GPU。开发调试产物统一存放 `/tmp/omniscene-temporal18/`。** GPU 严格加载、真实前向及正式全量评估仍须等待用户明确授权。2026-09-18 的 17 项 CPU 检查属于旧六视角版本，本版验证另记于第 13 节。

## 1. 范围与不可突破的边界

### 1.1 两个版本基线

- 原 OmniScene 冻结基线：`a8318b4da99879c625a60154d111a510e5ac5a91`。原 `data/`、`model/`、`builder/`、`tools/`、根目录 `train.py/evaluate.py/demo.py`、`configs/OmniScene/`、启动脚本及依赖设置继续保持原样。
- 本次适配起点：`1f2c6f56bbf77a7216db5e862af5edfdf6adea47`，已包含独立的 PandaSet/DDAD 六视角加载与评估。升级在这些目标文件上进行，不再创建另一套训练器、数据模块或模型包装框架。
- 本次只读参照：SVF-GS `af39b31d984ba128020282764265fa38d31ce767`；其中 `be86052` 实现 18 视角流程，`af39b31` 增加 DDAD 掩码评估。

原 `.gitignore` 中此前已授权的 `/data/PandaSet`、`/data/DDAD` 两条规则继续保留；三条数据软连接不修改。SVF-GS 工作区和共享数据只读，本轮不提交或推送代码。

### 1.2 保留的执行架构

```text
原路径：原 train.py / evaluate.py / demo.py
          → 原 nuScenes Dataset → 原 OmniScene 模型与原行为

目标路径：cross_dataset.train / cross_dataset.evaluate
          → 当前 PandaSetDataset / DDADDataset / 目标 Dataset 工厂
          → 原 builder 与原样 OmniScene 模型
          → cross_dataset.evaluation 的目标指标和结果文件
```

升级集中于第二条路径的数据与评估内容：

1. 中央 6 路用于特征提取和高斯重建，前后帧 12 路与中央 6 路共同用于目标渲染。
2. 读取 SVF-GS **已经完成**的 `processed`、选帧清单、深度元数据和掩码清单。
3. 评估输出 `all_18`、`novel_12`、`input_6`；DDAD 可选仅在 novel 12 使用自车有效掩码。
4. 原模型已经按照目标相机张量的实际 `V` 渲染，复用其 `forward_test()`，不修改 encoder、decoder、渲染器、损失或优化器。

### 1.3 明确不做的事情

- 不迁入或调用 SVF-GS 的预处理脚本、原始数据 SDK、选帧生成器、Metric3D 网络、权重下载器或深度推理。
- 不重新裁剪或写回 RGB，不重建内外参、索引、深度或置信度，不下载/去噪/生成自车掩码。
- 不导入 SVF-GS 的 Python 运行包，不修改 `sys.path` 指向该项目。参考实现仅用于核对接口和数值，部署本项目时不依赖另一个代码仓库的可导入性。
- 不移植 CADGA/EAGR、Hydra、训练状态框架或其结果分析框架；只借鉴必要的资产读取和指标定义。
- 不修改原 nuScenes 指标、数据加载、训练设置，不通过 monkey patch 或注册项替换间接改变原路径。
- 不引入动态物体分割或动态掩码。新协议已有时刻变化，不再以“输入输出相同”解释此决定；这里明确约定不增加动态物体筛选。

## 2. 已核对的变化与当前资产

### 2.1 六视角版本与本次目标

| 项目 | 已提交六视角实现（历史） | 本次升级实现 |
| --- | --- | --- |
| 数据协议 | `svfgs_single_frame_v1` | `svfgs_temporal18_v1` |
| 默认输出 | 中央 6 路，`only_input=True` | 前后 12 路 + 中央 6 路，`only_input=False` |
| 输出顺序 | 中央相机顺序 | 0–11 按相机 before/after 交错；12–17 为中央输入 |
| RGB 运行时缩放 | PIL 默认插值 | 显式 `PIL.BILINEAR`，与新参考加载器一致 |
| 文件寻址 | 从图像路径推断其他文件 | 使用 sensor 中明确保存的五类相对资产路径 |
| 数据完成性 | 读取 bins 与 PKL | 核对 selection、manifest、bins、PKL 和资产身份 |
| 正式汇总 | `all_6` | `final/all_18`，同时保存两个诊断分组 |
| DDAD 自车区域 | 全图评估 | 默认 `ddad_ego_novel12_v1`；可手动关闭为全图 |
| train 资产 | 旧产物有训练索引 | 当前新产物没有训练索引，不允许回退到 test |

不能只将评估张量的 `6` 改成 `18`：新资产的索引范围、DDAD 相机映射、图像变换和资产身份均已更新。旧 `all_6` 结果不能改名后当作本版结果；旧 README 命令的 3120/395 样本说明也不能继续用于新数据。

### 2.2 2026-09-23 共享资产实测快照

| 项目 | PandaSet | DDAD |
| --- | ---: | ---: |
| 原始范围（SVF-GS 说明） | 本地 39 个序列，3120 帧 | 官方 validation 50 个场景，3950 帧 |
| 当前有效 test bin | 264 | 324 |
| 当前有效场景 | 38 | 49 |
| 去重图像资产 | 4680 | 5760 |
| manifest `complete` | `true` | `true` |
| 当前 train 的 selection / manifest / bins | 均不存在 | 均不存在 |
| val | test 均匀抽取 10 个 | test 均匀抽取 10 个 |

这些数量用于本次对照验收，运行时从清单读取，不在加载器里硬编码长度。项目 `test` 是参考项目发布的评估划分：PandaSet 不是完整公开数据集的官方 test，DDAD 对应官方 validation。

本轮核对的语义选帧摘要：

```text
PandaSet selection_sha256:
fe392ab407dc8b11c3fa4c1bac81d98a7783d91655bfe027c19cf017abda576a
DDAD selection_sha256:
773fdd0e82f59deb8d0696ae64a6d8e2c5aaf199426d561e7fa1c3d9270eb8c0
DDAD ego mask manifest 文件 SHA-256:
473adeaf83e9d1f086e107a234bd89eed248e8bfcbdc3784762f49635384f793
```

这些是当前资产快照，不是所有后续资产版本的硬编码白名单。未来资产变化应重新核对并记录身份；不能混合不同清单、裁剪或掩码版本的结果。

## 3. 只读消费预处理产物

### 3.1 数据入口与文件契约

继续使用现有软连接：

```text
data/PandaSet -> /home/B_UserData/dongzhipeng/Datasets/PandaSet
data/DDAD     -> /home/B_UserData/dongzhipeng/Datasets/DDAD

<dataset>/processed/
├─ selection_test.json
├─ manifest_test.json
├─ bins_test.json
├─ bin_infos/<bin_token>.pkl
├─ images_small/<scene>/<camera>/<frame>.jpg
├─ params_small/<scene>/<camera>/<frame>.json
├─ dptm/ 或 dptm_small/
│  └─ <scene>/<camera>/<frame>_{dpt.npy,conf.npy,meta.json}
└─ ego_masks/vidar_v1/              # 仅 DDAD，可选读取
   ├─ manifest.json
   ├─ source_1216x1936/CAMERA_*.png
   ├─ cleaned_1216x1936/CAMERA_*.png
   └─ 224x400/<pixel_sha256>.png
```

PandaSet 深度目录为 `dptm`，DDAD 为 `dptm_small`。有正式 train 资产时才读取对应 `selection_train.json`、`manifest_train.json`、`bins_train.json`，不由本项目生成。

每个 sensor 已保存 `data_path`、`intrinsic_path`、`depth_path`、`confidence_path`、`depth_meta_path`，均相对于自己的 `processed_root`。加载器直接读取这些字段，不用中央图像的 stem 拼其他相机/时刻的路径。路径解析后必须仍在 configured root 内；不能混入另一套旧资产或越界路径。

`source_path`、原始数据位置和 Metric3D 配置/权重路径是来源记录，读取派生资产不要求这些原始路径或网络权重在本项目中存在。旧 `processed_only_input` 不参与本版默认加载，也不作缺文件回退。

### 3.2 清单与资产校验

只在 `cross_dataset` 的数据读取层实现必要检查，保持与 SVF-GS 相同的资产身份语义：

1. `manifest_<storage_split>.json` 存在且 `complete=true`，schema 为 `svfgs_temporal18_v1`。
2. 读取 `selection`，排除其自身 `selection_sha256` 字段后，以参考实现的规范 JSON 序列化规则重新计算摘要；核对 dataset、storage split 和 protocol。
3. `manifest`、`selection`、`bins` 的选帧摘要一致；bins 顺序与 selection 中的 bin 顺序完全一致，无重复且非空。
4. 按 manifest 核对所读 PKL 的文件摘要；PKL 的 token、schema、selection 身份一致，每相机恰有 `[center,before,after]` 三项。
5. 根据 `asset_id`、`camera`、内参、图像摘要和深度元信息校验所读 RGB、深度、置信度的对应关系；拒绝 `synthetic=true`、错误深度来源、变化的资产文件或不完整的 bin。
6. 图像为 224×400；原尺度深度/置信度为 float32、相同尺寸且有限，深度满足参考协议范围（非负、存在正值、最大不超过 300m）。不额外归一化或裁剪有效深度。

`selection.protocol` 的语义字段与 SVF-GS 当前 `build_zero_shot_config()` 对齐，包括 schema、stride/offset、距离阈值、图像尺寸、相机映射、裁剪方法、深度来源等。`raw_root/processed_root` 属于运行位置，不参与协议相等判断；掩码配置也不能塞进该协议字典。`mini_size=100/demo_size=10` 可作为上游清单身份保留，但不因此扩展本项目 split API。

哈希、几何核对都只读文件，不写回清单；缺失数据时明确报错，说明需要由 SVF-GS 准备完成，不在本项目触发修复或预处理。

### 3.3 选帧语义只解释、不重新执行

参考产物已经按原始帧序号 `0,10,20,...` 选择候选中央帧，并在同场景全部原始帧中按累计 XY 轨迹长度选择前后最接近 1.6m 的帧；两侧实际位移至少 0.1m，距离并列时取时间更近者。

加载时完全采用冻结结果，读取 `frame_ids`、`frame_indices`、`frame_gaps`、`distance_m` 和 `scene_id` 作为身份与诊断信息。不从旧 stride 索引重选端点、不跨场景补帧，也不把“两侧目标 1.6m”描述为每个 bin 严格跨度 3.2m。

## 4. 数据划分、相机顺序与 batch

### 4.1 仅保留 train / val / test

| 本项目 split | 读取与选择规则 | 当前可用性 |
| --- | --- | --- |
| `test` | 新 `bins_test.json` 全部有序 token | PandaSet 264，DDAD 324；对应 SVF-GS Dataset 的 `test`（其命令行 `test_split=total`） |
| `val` | 在同一 test 清单上取 `np.linspace(0,N-1,min(10,N),dtype=int)` | 各 10 个，与当前参考 Dataset 的 `val` 一致 |
| `train` | 只接受独立、完成发布的 train 清单与资产 | 当前不可运行，缺文件直接报错 |

不新增 `total`、`mini-test`、`demo` 等别名。`--max-samples N` 仍仅用于显式限量诊断，不能冒充新的正式划分。未来 DDAD 训练需由上游使用官方 train，PandaSet 需另定独立序列划分；不复制、重命名或拆分现有 test 来冒充 train。

### 4.2 相机与时间索引

| 输入槽位 / 语义 | PandaSet | DDAD | 对应输出 before / after / center |
| --- | --- | --- | --- |
| 0 / CAM_FRONT | front_camera | CAMERA_01 | 0 / 1 / 12 |
| 1 / CAM_FRONT_RIGHT | front_right_camera | CAMERA_06 | 2 / 3 / 13 |
| 2 / CAM_FRONT_LEFT | front_left_camera | CAMERA_05 | 4 / 5 / 14 |
| 3 / CAM_BACK | back_camera | CAMERA_09 | 6 / 7 / 15 |
| 4 / CAM_BACK_LEFT | left_camera | CAMERA_07 | 8 / 9 / 16 |
| 5 / CAM_BACK_RIGHT | right_camera | CAMERA_08 | 10 / 11 / 17 |

DDAD 新物理顺序为 **01、06、05、09、07、08**。旧六视角版的 01、05、06、07、08、09 映射不适用于新语义，不能为了保持旧输出数值而继续使用。

输入取每路 `sensor_info[cam][0]`；novel 依相机顺序取 `[1]`、`[2]`；最终输出为 `novel + center`。RGB、K、pose、深度、置信度、射线、路径及 eval_mask 全部使用同一顺序。末尾六路与输入逐元素对应，不能把输入也扩为 18 路。

### 4.3 保留现有模型 batch 字段

以下为单样本形状，DataLoader 仅在最前面增加 B：

| 字段 | 形状 / 语义 |
| --- | --- |
| `bin_token`、新增 `scene_id` | 字符串；DDAD 不从 token 猜测相机或文件名 |
| `inputs.rgb` | `[6,3,H,W]` |
| `inputs_pix.depth_m/conf_m` | `[6,H,W]` |
| `inputs_pix.ck/c2w` | `[6,3,3]` / `[6,4,4]` |
| `inputs_pix.fx/fy/cx/cy` | `[6]` |
| `inputs_pix.rays_o/rays_d` | `[6,H,W,3]`，方向不归一化 |
| `inputs_vol.w2i` | `[6,4,4]`，只投影输入图像特征 |
| `outputs.rgb` | 默认 `[18,3,H,W]` |
| `outputs.depth/depth_m/conf_m` | 默认 `[18,H,W]`；depth 与 depth_m 均为 Metric3D 尺度深度 |
| `outputs.c2w/fovx/fovy` | `[18,4,4]` / `[18]` / `[18]` |
| `outputs.rays_o/rays_d` | `[18,H,W,3]` |
| `outputs.eval_mask` | 仅启用 DDAD 自车掩码时出现，bool `[18,H,W]` |
| `eval_mask_manifest_sha256` | 仅掩码开启时出现的身份字符串，batch 内必须一致 |

图像路径及 view/asset 身份可作为数据层元数据保存，用于审计和可视化，不送入 encoder。无需模仿 SVF-GS 的 `inputs.depths/confs` 或新增模型内部结构。

保留已有 `only_input=True` 作为显式六视角诊断模式；它读取**新选帧清单的中央六路**，输出 `final/all_6`，不生成虚假的 novel 分组，也不等于旧 3120/395 清单的历史实验。默认及正式配置均设 `only_input=False`。`use_center=True/use_first=False/use_last=False` 继续只约束输入帧选择；目标前后帧由新协议决定，不要求将原 `use_first/use_last` 改为 True。

## 5. 几何与图像读取

### 5.1 使用已完成的跨时刻位姿

上游已为每张图像计算：

```text
T_cv = inverse(T_world_from_lidar(center)) @ T_world_from_camera(image)
```

每个 bin 的 18 路预处理位姿统一到**同一个中央 LiDAR 坐标系**。本项目读取每个 sensor 的 `sensor2lidar_transform`，不以静态标定外参代替前后帧变换，不分别将三个时刻归一化，不基于 DDAD 时间戳自行推断 stem。

**DDAD 加载边界必须额外对齐公共坐标轴。** DDAD 预处理参考系为 `+X 前 / +Y 左 / +Z 上`，而 nuScenes 训练权重使用 `+X 右 / +Y 前 / +Z 上`。在 `DDADDataset.read_info()` 中先调用原有清单/位姿校验，再对内存中的中央 6 路及前后 12 路全部执行：

```text
A_ddad = [[0,-1,0,0], [1,0,0,0], [0,0,1,0], [0,0,0,1]]
T_model = A_ddad @ T_cv
R_model = A_ddad[:3,:3] @ R_cv
t_model = A_ddad[:3,:3] @ t_cv
```

更新 `sensor2lidar_transform/rotation/translation` 后，复用不变的公共 `camera_tensors()` 重新派生几何量。该变换只在 DDAD 子类执行，所有划分、全图/自车掩码和 `only_input` 模式一致；不写回预处理文件，不旋转模型范围或修改已学习参数。PandaSet 保持 `T_model=T_cv`，原 nuScenes 路径保持原样。

公共参考系变换与相机局部 OpenCV→OpenGL 翻转不同。保留原相机轴约定：

```text
F = diag(1,-1,-1,1)
c2w_gl = T_model @ F
ray_cam = [(u+0.5-cx)/fx, -(v+0.5-cy)/fy, -1]
world_point = ray_o + ray_d * metric_depth
w2i = K4 @ inverse(T_model)
```

射线不归一化；Metric3D 米制深度是光轴深度。`w2i` 只按输入 6 路供 Volume encoder 使用，目标 18 路使用自己的 c2w/FOV；原模型的 Plücker 特征与反投影位置由新射线自然更新。统一左乘保留相对位姿、投影和米制尺度，但模型对公共坐标方向的响应不应假定不变。

**本段修正 2026-09-23 方案中“DDAD 不引入额外轴旋转”的遗漏。** 之前与原始 SVF-GS 加载张量完全一致，只能证明复现了其数据坐标，不能证明与本项目 nuScenes 训练坐标一致。修复不涉及开源模型、渲染器或原模型 bug；启动指令、实验名称、输出目录和评估协议不变。DDAD 的 manifest 来源信息另记 `camera_frame`、`reference_to_model`，不改选帧或掩码协议。

### 5.2 原图变换已完成，运行时只缩放

- 保存图像已按原主点完成 `principal_center_float_v1` 浮点裁剪/重采样；读取 `params_small` 中的 `camera_intrinsic`、`pixel_transform=A`、`source_hw`、`image_hw`。
- CPU 元数据核对可计算 `K_processed=A @ K_raw` 和参考裁剪公式的 A，验证它们一致。这只是检查矩阵，不读取原图做第二次裁剪，不写出任何预处理资产。
- 224×400 → 112×200 时 RGB、深度、置信度显式使用 PIL 双线性；K 的第 0 行乘宽比例，第 1 行乘高比例，深度数值单位不变。224×400 不做多余缩放。
- 主点必须与处理后图像中心一致，FOV 使用每张图像自己的 fx/fy/cx/cy；原渲染器保持原样。
- DDAD 自车掩码缩放单独使用最近邻，不套用 RGB/深度的双线性插值。

## 6. DDAD 自车遮挡掩码

### 6.1 复用共享模板，不生成新掩码

已准备的资产为 `data/DDAD/processed/ego_masks/vidar_v1/`，schema 为 `svfgs_ddad_ego_mask_v1`，当前修订 `native_area_opening_128_v1`。上游记录来源为 VIDAR 提交 `0d84851ce4d86a9f132f8027898ff981e751db79`。

资产已在原始 1216×1936 分辨率去除面积 ≤128 的 8 邻域独立黑块，再按 RGB 的浮点像素变换完成裁剪。当前共 6 张原始模板、6 张清理模板、22 种标定变换、19 张共享处理后 PNG，覆盖 49 个场景的 294 个场景/相机组合。本项目只读这些文件，不重复上述操作。

像素 `0` 为排除的自车区域，`255` 为有效区域。模板是共享相机模板，不是逐场景分割；SVF-GS 已记录部分后相机车身边缘存在漏盖。保留该质量说明，不通过膨胀或观察模型误差自行扩大遮挡区域。

### 6.2 加载与对齐

新增轻量 DDAD 掩码读取 helper，放在 `cross_dataset/datasets/` 内：

1. 从 `(scene_id, sensor.camera)` 查 `scene_camera_to_variant → variants → mask_id → masks`。不能仅按 CAM 槽位查一张固定图；不同原主点对应不同裁剪变换。
2. 核对 manifest 的 dataset/schema/source commit、selection 语义摘要和 selection 文件摘要、相机顺序、0/255 语义、处理尺寸与变换类型。
3. 核对清单引用的掩码文件摘要；用目标图像自己的原始 K、参数 JSON 与变换条目核对 `A`、`K_processed`、source/image 尺寸。必要矩阵检查容差与参考值 `1e-6` 一致。
4. 直接读取处理后 224×400 PNG，运行时 `PIL.NEAREST` 缩放为 H×W，转换为 bool；按 `(mask_id,H,W)` 在 worker 内缓存。
5. 过去/未来按各自目标 sensor 的相机与场景解析；前 12 路使用有效掩码，后 6 路强制全 1。前相机模板本身全白，仍按全图指标计算。
6. 关闭时不读取或校验掩码目录；开启后缺文件、缺场景、相机/几何/哈希不符直接报错，不用全 1 静默兜底。

### 6.3 开关与作用边界

目标 `eval_args.eval_use_ego_mask` 在 DDAD 两种分辨率中默认 `True`，PandaSet 保持 `False`；ZeroShot 配置继承对应设置。独立评估可用 `--no-eval-use-ego-mask` 手动关闭，也保留 `--eval-use-ego-mask` 显式开启。具体规则：

| 场景 | 掩码行为 |
| --- | --- |
| DDAD 独立评估，`split=test/val` | 默认开启；`val` 仍是独立评估的十样本诊断 |
| DDAD `only_input=True` 的独立评估 | 默认开启，但六路 mask 全 1，分数应与同清单的关闭模式一致 |
| PandaSet 或 nuScenes | 拒绝开启；原 nuScenes 入口本身不接入此开关 |
| 训练入口、训练中验证 loss | 加载配置时固定关闭，不将独立评估默认值带入训练 |
| 独立评估使用 `split=train` | 必须显式关闭掩码，否则拒绝 |

SVF-GS 的 `mode=test` 是**运行模式**，不是仅指数据 `split=test`；本项目对应 `cross_dataset.evaluate`。保留既有 `train/val/test` API，同时将此新掩码功能限定在目标域独立评估。

`eval_mask` 不是动态掩码，不替换原 `mask_dptm`、Metric3D 置信度、输入 RGB、GT、深度或射线。原 `forward_test()` 不需要返回掩码：评估入口直接从同一 batch 获取并传给统计 helper，**不仿照参考项目修改 `model/omni_gs.py:get_data/test_step`**。源权重兼容检查也不加入掩码身份。

## 7. 统一的视角分组与像素指标

### 7.1 三个视角组

| 分组 | 输出切片 | 用途 |
| --- | --- | --- |
| `all_18` | `[0:18]` | 前馈 OmniScene 的正式结果 |
| `novel_12` | `[0:12]` | 新视角诊断；与基于优化的高斯方法比较时另行报告 |
| `input_6` | `[12:18]` | 输入重建诊断 |

保存与 SVF-GS 可直接对应的 `final/all_18`、`final/novel_12`、`final/input_6` 汇总键。本项目只报告自身最终融合渲染的 `stage=final`，不虚构 SVF-GS 模块阶段或高斯计数。

RGB 指标先按视角计算，在每个 bin 的对应组内以 float64 等权平均，再对该组全部唯一 bin 等权平均；不对不同大小的 batch 均值再等权平均，不先合并全数据集 MSE 再转 PSNR。

对 PSNR、SSIM、LPIPS 均应满足：

```text
score(all_18) = (12 * score(novel_12) + 6 * score(input_6)) / 18
```

这在掩码开启时也成立：12 个新视角各自在自己的有效像素上计算，6 个输入视角保持全图，然后仍按视角等权平均。不能改成只报 novel 12，也不能按视角有效像素数重新加权。PCC 是相关系数，不适用该加权公式。

### 7.2 全图指标

`pixel_protocol=full_image`。继续只读调用本项目 `tools.metrics` 的原 PSNR、SSIM、LPIPS/PCC 函数，不改公共工具；开关关闭或单个视角 mask 全 1 时，走原全图路径。

数据升级包含索引、相机和 RGB 插值变化，因此“掩码关闭回归”指同一套新 18 视角数据上的全图基准，不要求复现旧六视角产物的数值。

### 7.3 DDAD 默认 masked 指标

**强制一致性要求：启用 DDAD 自车掩码时，指标计算与汇总必须与 SVF-GS 一致，尤其是 `final/novel_12`；不能仅复用同一份掩码而采用另一套 masked 指标定义。** 以第 1.1 节固定的 SVF-GS 版本中 `configs/build_config.py:build_eval_mask_config`、`tools/metrics.py:compute_image_metrics/compute_eval_pcc`、`tools/ablation_metrics.py:render_records` 和 `tools/ablation_results.py:summarize_records` 为数值对照依据，覆盖视角选择、有效像素、指标参数、全有效视角分支和汇总顺序。后续参考实现变更须重新核对并记录版本，不静默跟随。

`pixel_protocol=ddad_ego_novel12_v1`；只对 mask 非全 1 的目标视角补算，计算限定在新增的目标评估 helper。设 M 为 bool 有效掩码：

| 指标 | 必须对齐的计算方式 |
| --- | --- |
| PSNR | 沿用原 `[0,1]` 裁剪；`MSE=sum(M*(pred-gt)^2)/(3*sum(M))`，再 `-10*log10(MSE)` |
| SSIM | 原 11×11 高斯窗口，sigma=1.5、`use_sample_covariance=True`、`channel_axis=0`、`data_range=1`；读取 SSIM map，仅平均 M 经 11×11 方形腐蚀后的有效窗口中心及通道，图像边界同样不进入支持区 |
| LPIPS | 独立缓存 `LPIPS(net='vgg', spatial=True).eval()`；`pred_eval=where(M,pred,gt)`，沿用 `normalize=True`；距离图只在 M 有效位置求归一化均值 |
| PCC（可选） | 每个 bin 的每个视角组内，将有效预测/参考深度展平后计算，再对 bin 等权平均；参考仍为 Metric3D-v2 |

具体而言，`novel_12` 使用输出 `[0:12]`，每个 bin 的 PSNR、SSIM、LPIPS 均为这 12 个逐视角分数的 float64 算术平均，再对全部唯一 bin 等权平均。模板全白的新视角仍计入这 12 个视角，并调用原全图函数；不得只平均有遮挡的相机、先汇集 12 路像素再算 PSNR，或按有效像素数量加权。SSIM、LPIPS 不额外套用 PSNR 的 `[0,1]` 裁剪，也不额外缩放、裁剪图像或修改参考实现的归一化参数。`input_6` 保持原全图结果，`all_18` 按第 7.1 节规则汇总；若启用 PCC，其组内有效深度展平方式也必须一致，但默认不报告 PCC 的设置保持不变。

禁止将 pred 和 GT 同时涂黑后直接计算原全图指标，因为这会把无效区域的零误差计入分母。SSIM 必须排除跨入无效区域的窗口；LPIPS 的 GT 填充只发生在指标内部，不修改模型输入、渲染结果或保存图像。该 LPIPS 定义具有 VGG 卷积上下文，不宣称等价于标准全图 LPIPS 或逐像素独立误差。

无有效像素、无完整 SSIM 窗口、PCC 有效点不足/常量深度或最终非有限指标应明确报错，不自动跳过失败样本、赋满分或导出完整成功结果。计算 PCC 前保证深度切片连续，保持参考的组内展平语义；默认 `compute_pcc=False` 不变，不加载 DA-v2 或新增相对深度。

### 7.4 不混合不同实验身份

每条记录都携带 `pixel_protocol`、`mask_manifest_sha256`；关闭时哈希为空，开启时哈希必填。即使 input 6 的有效像素仍是全图，也记录整个运行的掩码协议，以便与同次 all 18 成套核对。

跨进程、汇总或配对比较时，拒绝混合不同选帧摘要、视角组、像素协议或掩码版本。旧六视角 CSV 缺少像素协议时最多解释为旧全图结果，不能自动转换为新 18 视角记录。

## 8. 配置与训练兼容性

### 8.1 保留八份配置及继承层次

```text
原 configs/OmniScene/omni_gs_nusc_novelview_r50_<resolution>.py（只读）
  → configs/PandaSet/ 或 configs/DDAD/ 的完整目标配置
    → configs/ZeroShot/ 的源域到目标域零样本配置
```

沿用现有 112×200 / 224×400 八个配置文件和名称，不新增平行配置体系。实际调整项仅限（四份 ZeroShot 配置通过继承生效）：

- 目标数据协议改为 `svfgs_temporal18_v1`，默认 `only_input=False`；processed_root 仍指向现有新产物目录。
- 必要的 `temporal_cfg` 读取协议；顶层 `dataset_params` 与 `model.dataset_params` 同步，后者仅保持现有配置一致性，不由模型消费额外测试开关。
- `eval_args.eval_use_ego_mask` 在 DDAD 独立评估中默认 `True`，PandaSet 和训练保持 `False`；使用独立 mask 配置和三视角组统计，`compute_pcc=False` 保持不变。
- 结果目录追加 `novel18_s10_d1p6_min0p1`；掩码开启追加 `_ego_novel12_v1`，显式六视角诊断再区分 `_input6`。保持源权重目录独立。
- 新入口的配置检查允许合法的 18 输出；`num_cams=6`、Pixel 的相机数及编码器输入视角数始终不变。

MMEngine 覆盖子配置不会重新计算父配置的派生字典，继续使用对应分辨率父配置并同步必要嵌套字段。清单中的 224×400 是预处理尺寸，运行分辨率由原 camera/loss 配置一致决定；不将源分辨率校验放宽为“参数形状相同即可”。

### 8.2 模型完整设置保持原样


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

### 8.3 原训练配置与语义保持原样


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
| `warmup_steps` | 1000；原调度构造还使用进程数倍率，详见本节末的原调度说明 |
| `lr_scheduler_type` | 保留原字符串 `constant_with_warmup`；原执行实际创建 LinearLR + CosineAnnealingLR |
| 日志/验证/保存频率 | `print_freq=100`、`val_freq=500`、`save_freq=2000`，沿用原 `global_iter` 判断位置 |
| 验证调用 | 沿用原 `validation_step(batch, directory)`，不增加参数或装饰器 |
| 保存与恢复 | 沿用原 Accelerate 状态格式与 `resume_from='latest'`，不新增强制元数据格式 |

原 `_base_/optimizer.py` 中的 `lr=5e-5` 和 `paramwise_cfg` 保留原样，它们没有被原训练入口用来创建优化器。目标入口也调用原模型方法，不用新的 optimizer 构建器替换这条链路。

默认实际迭代量仍由 `max_epochs × len(prepared_train_dataloader)` 与原 batch 计步规则决定。当前没有新协议的训练资产，不再给出基于旧 3120/1265 训练清单计算的本版训练步数，也不以 264/324 个测试 bin 估算训练规模。

原 LinearLR / CosineAnnealingLR 的 warmup、T_max 及 milestone 仍乘进程数；优化器、scheduler 调用顺序、epoch 停止条件与验证保存标签不变。原 `global_iter` 在保存/验证判断之后递增，因此原迭代 0 的保存/验证行为仍保留，不增加最后一次验证/保存。

损失继续保持融合 L2=1.0、融合 LPIPS=0.05、融合置信度深度损失=0.01，Volume 对应权重为 1.0/0.05/0.01，原空间掩码及置信度处理不变。Volume 辅助项仍使用 `iter < iter_end - 1000`，默认训练门限是 4000，不重新解释为实际训练最后 1000 步。

`cross_dataset/train.py` 主体无需改变。由现有 `load_config(..., training=True)` 所在的目标配置层进行训练资产存在性及掩码禁用预检查，使缺少 train 产物时在模型/GPU 构造前报错。纯配置解析与继承对照不代表当前数据已经支持训练。未来上游正式提供独立 train 资产后，由同一 Dataset 接口满足六输入、十八目标的原训练调用，原恢复方式及其限制也保持不动。

## 9. 评估入口、权重与输出

### 9.1 只适配数据和统计连接

继续使用 `cross_dataset.evaluate`：

```text
读取目标配置/掩码开关/显式源权重
  → 只读校验完成的 temporal18 清单（掩码开启时校验掩码清单）
  → 当前 Dataset 工厂与 EvaluationShard
  → 原 builder + 原样 OmniScene 模型 + 严格模型权重加载
  → eval/no_grad 调用原 forward_test(batch)
  → 从 batch 获取 eval_mask、scene_id、资产身份
  → 每视角指标 → 每 bin 三个分组 → 分布式收集与完整覆盖检查
  → 独立结果目录
```

源权重继续使用本项目 `workdirs/[reference]omni_gs_nusc_novelview_r50_112x200/checkpoint-100000` 与同目录保存配置。它是 OmniScene 权重，不是 SVF-GS 权重；只借用后者准备的数据和评估协议。

`cross_dataset/checkpoint.py` 的显式路径、严格键/形状加载、共享参数别名、来源说明逻辑无需改变。目标数据/掩码元信息属于评估身份，不能要求旧 OmniScene 权重携带目标域掩码或选帧信息。无来源 sidecar 时仍如实记录 `unverified_legacy_checkpoint`。

### 9.2 多进程覆盖和文件粒度

沿用 `EvaluationShard` 的明确补齐标记和当前 batch size 设置，不修改训练 sampler。以下三组要求适用于默认 18 目标模式；显式 `only_input=True` 时只核验 `final/all_6`：

- 根据生效配置校验 pred/GT 为 `[B,18,3,H,W]` 或诊断模式的 `[B,6,3,H,W]`，深度和掩码的 B/V/H/W 必须一致；不接受任意视角数后猜测分组。
- 真实样本身份仍是唯一 bin；每个真实 bin 必须恰有三个 `stage=final` 分组记录。
- 记录唯一键改为 `(bin_token,stage,view_group)`；正常三组记录不应被误判为三个重复 bin。
- 补齐样本的所有组记录一起排除；非补齐重复、缺少任一组、错误 token 或协议不一致均失败。
- 每组独立核对与同一 `expected_tokens` 完全覆盖，`evaluated_count` 是唯一 bin 数而非 CSV 行数。
- 可视化和 PLY 每个真实 bin 只保存一次；在 18 路图中标明 before/after/center 顺序或按三组六相机布局，不能仍固定只保存六个目标。
- 开启掩码后保存的 RGB/深度仍是原渲染结果；可另存 mask 诊断图，但不将指标内部 GT 填充图当作模型预测。

### 9.3 结果文件与身份

保留既有文件位置和完成状态框架，仅扩展内容：

| 文件 | 升级内容 |
| --- | --- |
| `resolved_config.py` | 实际生效的数据协议、源模型配置、only_input、掩码开关及目录 |
| `evaluation_manifest.json` | 现有源权重/代码/状态信息，增加 selection/manifest 摘要、深度来源、相机物理映射、视角顺序和掩码身份 |
| `per_bin_metrics.csv` | 18 目标模式每 bin 三行，含 scene_id、stage、view_group、view_count、RGB 指标、可选 PCC、像素协议和掩码摘要；六视角诊断为一行 |
| `per_view_metrics.json` | 每 bin 的 18 路值及 camera/time/asset_id、有效像素数，供索引和加权核验 |
| `evaluation_summary.json` | 现有完成状态字段 + `final/all_18`、`final/novel_12`、`final/input_6` 三组值与各组 num_bins/协议身份 |
| `visualizations/<bin>/` | 开启保存时使用，一次写入完整目标视角及可选 PLY |

不为了模仿 SVF-GS 新增训练器或强制改用其 `data_provenance.json` 文件；等价数据来源内容写入本项目现有 manifest 即可。掩码信息包括 manifest 路径/摘要、asset_revision、source commit、插值及指标配置、模板质量说明。

`running/complete/limited/failed` 机制保留。正式完整评估须三组均覆盖同一完整 test 清单；当前预期 PandaSet=264、DDAD=324。使用 `--max-samples` 一律标记 `limited`，中断/失败不生成完整成功声明。mask on/off 使用不同协议目录，即使汇总键相同也不能混用或覆盖。

## 10. 修改范围与实施顺序

### 10.1 限定文件范围

**实施严格限定于下列独立目标文件及本文。** 原架构与冻结边界不变：

| 文件范围 | 必要变化 |
| --- | --- |
| `cross_dataset/datasets/common.py` | 完成清单读取、18 目标展开、BILINEAR 缩放、逐图参数读取与几何打包 |
| 同目录 `pandaset.py/ddad.py/__init__.py` | 保留 Dataset 类和工厂接口，接入目标协议配置 |
| 同目录新增 `assets.py/ego_mask.py` | 只读资产/元数据检查、DDAD 掩码读取与缓存；不放生成流程 |
| `cross_dataset/configuration.py` | 18 输出配置校验、输入/输出数量区分、掩码合法性、缺 train 资产预检查 |
| `configs/PandaSet/`、`configs/DDAD/`、`configs/ZeroShot/` | 八份现有配置的必要数据/评估字段与独立输出路径 |
| `cross_dataset/evaluation.py` | 三分组统计、分组覆盖、像素协议验证、CSV/JSON 与 18 路可视化 |
| 新增 `cross_dataset/metrics.py` | 目标专用 masked PSNR/SSIM/LPIPS/PCC；全图复用原工具 |
| `cross_dataset/evaluate.py` | CLI 开关覆盖、从 batch 传掩码给指标、扩展 manifest 和结果身份 |
| `cross_dataset/tests/` | 替换旧数据数量/形状假设，补 18 视角/掩码/分组测试，保留原冻结与训练兼容检查 |
| `cross_dataset/README.md` | 已更新实际命令、样本数、三组结果、掩码开关及 `/tmp` CPU 对照流程 |

`cross_dataset/train.py`、`checkpoint.py`、`ply.py` 原则上保持现状；若实施发现额外接口冲突，先明确原因，不扩大到原代码修正。已有训练循环不承担掩码统计或数据协议变更。测试冻结清单仍以原基线核验，不能因升级而放宽原文件边界。

### 10.2 实施顺序

1. 固定上游参照版本、资产协议及相机/视角顺序；在现有目标配置中表达新契约。
2. 完成只读资产检查和 6→18 数据加载，先通过无掩码数据/几何 CPU 对照。
3. 完成三组全图统计和分布式覆盖，保持原单视角指标函数不变。
4. 接入 DDAD 已处理掩码和目标专用 masked 指标，再核对独立目录及结果身份。
5. 更新 README、旧测试与文档状态；核验原 OmniScene 文件零差异。
6. 用户明确允许后再做原样模型 GPU 验证及完整 test；不在开发时默许 GPU。

## 11. CPU / GPU 验收要求

### 11.1 CPU 数据与协议验收

- 当前清单的 token 顺序、完整性、val 均匀选取与 SVF-GS 对齐；缺 train 资产明确失败；mini/demo/total 不成为本项目 split。
- 两数据集、两分辨率、test/val 首中末样本与参考当前加载器对照全部输入和 18 输出字段；记录绝对误差，不能复用旧六视角的 36 次对照结论。
- 检查输出 12–17 与输入的 RGB、尺度深度、K 对应关系、pose 和射线一致；前 12 的 before/after 顺序、scene_id、asset_id 正确。
- 抽移动较慢、时间间隔较大的 bin 核对统一中央坐标与投影往返；核对 DDAD 物理相机朝向，禁止回用旧映射。
- 检查 RGB 双线性、depth/conf 双线性、mask 最近邻及每图 K 的分辨率缩放。
- 对全部引用文件做只读覆盖/身份核对；合成临时资产仅用于负例夹具，不调用参考的预处理生成器，不写共享数据。
- 原 `get_data()` 的 CPU 方法夹具检查新的 batch 消费，依然不实例化带 CUDA 分配的完整模型。

### 11.2 CPU 掩码与指标验收

- **与 SVF-GS 的数值对照是必过验收项。** 将完全相同的预测 RGB/深度、GT、bool 掩码、18 路顺序、bin 清单及指标配置交给两侧实现，逐视角、逐 bin 和最终三组汇总逐级核对，重点检查 `final/novel_12`；比较的是同一组张量的计算结果，不要求两个不同模型的渲染分数相同。
- 参考结果由固定版本 SVF-GS 在独立环境生成并保存为测试夹具，本项目测试只读取夹具，不引入对 SVF-GS 运行包的依赖。对照覆盖部分掩码、全 1 新视角、不同有效面积、多个 bin、两种运行分辨率和退化输入；记录参考提交、依赖/LPIPS 权重版本、dtype、逐指标误差及预先确定的容差。真实 LPIPS 对照未完成或任一层结果不一致时，不能宣称已与 SVF-GS 对齐，不以放宽容差或调整汇总口径绕过差异。
- 缺掩码、错误相机/场景、变更哈希、错误 K/A/尺寸/像素值时失败；关闭时无需掩码目录。
- 同数据开/关掩码时，输入、目标 RGB/深度、pose、射线完全不变；掩码后 6 路及前相机 novel 视角全 1。
- 全 1 掩码逐视角结果与原全图函数一致，input 6 与开关关闭时一致；only_input=True 的同清单诊断同样成立。
- 只改被排除区域的预测，masked RGB 指标不变；改变有效区域时指标变化。验证 PSNR 分母、SSIM 腐蚀支持区域、LPIPS GT 填充和真实 VGG 距离图。
- 验证 PCC 组内有效深度展平及退化输入错误，不错误应用 RGB 的 12:6 加权关系。
- 合成 18 个不同逐视角值验证三个分组及 `all_18` 加权恒等式；覆盖不同 batch size、多进程尾批、小于进程数的样本选择。
- 检查三分组 CSV 共 3N 行但 evaluated_count=N，每组 token 完整；混合 pixel_protocol/mask hash、缺组或非法重复不能形成成功 summary。
- CPU LPIPS 实网测试以本地已有权重为前提；无法离线完成时如实记录未覆盖，不能用轻量替身宣称真实 LPIPS 验证通过。

### 11.3 GPU 授权后的验证

1. 严格加载指定 reference 权重，两数据集各一个真实 bin，确认模型保持六输入、输出完整 18 路。
2. DDAD 同一 bin 同权重开/关掩码，确认高斯、RGB、深度不变，input 6 全图指标不变，仅允许 novel 12 及其派生 all 18 分数变化。
3. 保留原模型所需字段及形状，只替换 `outputs` 中的 RGB/深度数值，保持 `inputs/inputs_pix/inputs_vol` 不变，检查高斯重建不变；目标 pose 仅控制渲染，不引入测试时微调。不为该检查修改原模型以支持缺少必需字段。
4. 诊断性 batch/multi-GPU 检查通过后再做完整 264/324 test；限量结果与正式结果分目录。
5. 原样网络训练兼容仍需独立合法 train 资产；不以 test 运行训练来完成验收，不修改原损失/停止/恢复逻辑。

## 12. 升级后运行方式

**以下接口已实现，正式 GPU 命令尚未执行，仍需用户明确允许使用 GPU。** `cross_dataset/README.md` 已同步。开发中诊断实验只取少量样本，输出必须放到 `/tmp`；正式实验目录仅用于后续授权的完整评估。

从项目根目录、`omniscene` 环境执行。source_config 显式使用 reference 保存的 112×200 配置，不能换成 SVF-GS 模型配置。

```bash
# PandaSet：新协议完整 test，264 个 bin，18 路全图。
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python -m cross_dataset.evaluate \
  --py-config configs/ZeroShot/omni_gs_nusc_to_pandaset_r50_112x200.py \
  --load-from 'workdirs/[reference]omni_gs_nusc_novelview_r50_112x200/checkpoint-100000' \
  --source-dataset nuScenes \
  --source-config 'workdirs/[reference]omni_gs_nusc_novelview_r50_112x200/omni_gs_nusc_novelview_r50_112x200.py' \
  --split test \
  --output-dir outputs/zero_shot/reference_nusc_to_pandaset_r50_112x200/test/checkpoint-100000/novel18_s10_d1p6_min0p1

# DDAD：新协议完整 test，324 个 bin，18 路全图。
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python -m cross_dataset.evaluate \
  --py-config configs/ZeroShot/omni_gs_nusc_to_ddad_r50_112x200.py \
  --load-from 'workdirs/[reference]omni_gs_nusc_novelview_r50_112x200/checkpoint-100000' \
  --source-dataset nuScenes \
  --source-config 'workdirs/[reference]omni_gs_nusc_novelview_r50_112x200/omni_gs_nusc_novelview_r50_112x200.py' \
  --split test \
  --no-eval-use-ego-mask \
  --output-dir outputs/zero_shot/reference_nusc_to_ddad_r50_112x200/test/checkpoint-100000/novel18_s10_d1p6_min0p1

# DDAD：同一完整 test，novel 12 排除自车，input 6 保持全图。
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 python -m cross_dataset.evaluate \
  --py-config configs/ZeroShot/omni_gs_nusc_to_ddad_r50_112x200.py \
  --load-from 'workdirs/[reference]omni_gs_nusc_novelview_r50_112x200/checkpoint-100000' \
  --source-dataset nuScenes \
  --source-config 'workdirs/[reference]omni_gs_nusc_novelview_r50_112x200/omni_gs_nusc_novelview_r50_112x200.py' \
  --split test \
  --eval-use-ego-mask \
  --output-dir outputs/zero_shot/reference_nusc_to_ddad_r50_112x200/test/checkpoint-100000/novel18_s10_d1p6_min0p1_ego_novel12_v1
```

保持输出目录不存在或为空的规则；默认目录在 CLI 覆盖生效后按实际 only_input/掩码状态生成。显式 `--output-dir` 优先，仍须为空并在 manifest 中准确记录协议，不能复用已有另一协议的结果目录；再次运行选择独立目录。`--split val` 用于同协议十样本诊断，`--max-samples 1 --output-dir /tmp/omniscene-ddad-smoke-run1` 使用独立临时诊断目录并标记 limited；都不能冒充完整 test。

## 13. 实施与核对记录

### 13.1 本版开发验证（CPU，2026-09-23）

- 23 项 CPU 测试全部通过（约 50 秒），另有 CPU/Gloo 两进程汇总检查通过；未启动 GPU 或完整模型。
- 已实现：6→18 目标加载、三组 float64 汇总、DDAD 独立评估掩码开关、错误资产拒绝、协议结果隔离和 18 路可视化。八份目标配置通过原有继承关系生效，训练循环、优化器、权重读取模块保持原样；当前缺少独立 train 产物时提前报错。
- 原文件冻结：继续以 `a8318b4` 的 52 个受保护 Git blob 校验，训练调用轨迹另作 CPU 夹具对照；未放宽冻结规则。
- 数据对照：固定 SVF-GS `af39b31` 在其独立 `svfgs` 环境导出夹具，本项目在 `omniscene` 环境只读消费。两数据集、两分辨率，各取 test 首/中/末 3 个 bin，共 12 次逐字段对照，最大绝对误差 0；test 和 val token 顺序与参考一致。另核对中央末 6 路与输入一致、投影往返及原 `get_data()` 的 CPU 接口。
- 资产覆盖：只读核对 PandaSet 264 个 PKL 与 23664 个去重必需文件、DDAD 324 个 PKL 与 29124 个去重必需文件均存在。PKL 均校验摘要与跨时刻身份；完整 RGB/深度文件的数值与哈希按样本加载时校验，未另做全数据解码实验。
- 掩码指标对照：每种分辨率 2 个合成 bin，共 4×18 个视角，使用真实 VGG LPIPS；分别比较逐视角 RGB、逐 bin 的三组 RGB/PCC、最终三组汇总，最大绝对误差均为 0。容差预先设为 `atol=1e-6, rtol=1e-5`，未因结果调整；记录依赖版本、VGG 权重摘要和 `cuda_initialized=false`。PCC 仅作兼容性对照，正式默认仍关闭。
- 补充检查覆盖全白视角回归、无效区预测变化不影响 masked 指标、SSIM 支持窗口、退化深度/空掩码、错误来源身份、分组三组完整覆盖、多种 batch/rank 补齐与一次性导出。CPU/Gloo 两进程以 5 个合成样本和 1 个限量样本核对汇总与失败消息传播。
- 产物均在 `/tmp/omniscene-temporal18/`：`reference/` 是参考夹具，`loader-comparison.json`、`metric-comparison.json` 为数值证据，`cpu-tests.log`、`distributed.log` 为检查日志。临时目录可能被系统清理；README 记录了重建参考夹具和重跑检查的命令，运行不依赖这些临时文件。

GPU 严格模型加载、真实模型前向、完整 test 指标仍未执行；CPU 对照不构成这些结论。独立训练数据当前缺失，因此未运行真实目标域训练。后续 GPU 调试仍须授权、少样本，并将结果放入 `/tmp`。

### 13.2 规划阶段的只读核对

2026-09-23 已阅读 SVF-GS 三份零样本文档及 temporal Dataset、条件加载、掩码读取、指标/分组汇总、协议配置和相关测试；已对本项目现有独立加载、评估与原模型渲染接口进行对照。

只读核对了两份 selection 的规范 JSON 摘要、manifest 完成状态、bins 顺序和三者的 selection 身份；各抽一个 PKL 校验文件摘要、三时刻字段和相机映射，并读取对应参数/深度元信息。DDAD 掩码 selection 绑定、manifest 文件摘要及全部 6 原始 + 6 清理 + 19 处理掩码的文件摘要核对通过。

上述为规划阶段的只读记录；随后开发阶段的 CPU 验证见第 13.1 节。始终未运行预处理或完整模型，也未将抽样数值对照当作全部 RGB/深度资产的数值验收。

### 13.3 旧六视角记录如何保留

`1f2c6f5` 保存 2026-09-18 的六视角实现与原文档。当时 17 项 CPU 检查及 CPU/Gloo 双进程夹具通过，数据为 PandaSet 3120/10/3120、DDAD 1265/10/395（train/val/test）；原文件冻结核对包含 52 个 Git blob。

这些都是旧资产与旧代码契约的历史事实。当前共享 `processed` 已发布新 18 视角产物，旧测试中的数量、train 文件和相机假设已在本轮开发中更新；不能原样重跑后将失败解释为新产物错误，也不能将旧通过记录写为本版通过。旧详细记录可查该提交中的本文件，不在当前规范中继续保留过时命令。

### 13.4 DDAD 公共坐标系修复复查（CPU，2026-10-01）

- 根因已从实际资产确认：nuScenes 前相机光轴约为 `(-0.0034,0.9998,0.0188)`，DDAD 原始前相机光轴约为 `(0.9977,0.0674,-0.0094)`。仅有相对位姿和投影自洽的检查未覆盖两者公共坐标朝向的差异。
- 生产代码只修改 `cross_dataset/datasets/ddad.py`。校验原始位姿后，在内存中统一左乘第 5.1 节矩阵，并同步 rotation/translation；共享加载器从新位姿重算 c2w、rays、w2i。增加 DDAD 来源元数据说明，不改变协议标识或目录名称。
- 17 项相关 CPU 测试通过，覆盖真实 DDAD 首/中/末 bin、两分辨率、6×18 相对位姿、米制距离、投影、反投影、Plücker 几何、掩码开关、六视角诊断、PandaSet 原参考对照、原配置和训练调用冻结。新增检查将真实 nuScenes 相机朝向作为锚点，避免仅因投影正确就误判公共坐标正确。
- DDAD 位姿、射线、反投影点与按轴变换的参考值逐元素一致；w2i 在 float32 中重新求逆，与先求逆再变基的旧夹具存在舍入差异，最大系数差分别为 `7.6294e-5`（112×200）和 `1.5259e-4`（224×400）。测试以矩阵尺度的四个 float32 ulp 约束该差异，并独立保留 `2e-3` 像素投影检查；不修改生产计算精度或指标容差。RGB、深度、内参、掩码及 PandaSet 数据对照保持不变。
- 验证日志及修复前朝向快照在 `/tmp/omniscene-ddad-axes-20261001/`；参考夹具仍只读复用 `/tmp/omniscene-temporal18/reference/`。未运行 GPU、未修改共享数据或删除旧实验结果，未处理原模型 bug；未对修复后的模型指标提升作结论。

原启动指令和实验名称可继续使用，由用户清理旧零样本结果后重评；原开源模型、渲染器、nuScenes/PandaSet 运行路径、训练设置与三组评估协议全部保持不变。

## 14. 只读依据

### 本项目

- [现有独立数据加载](../cross_dataset/datasets/common.py)
- [现有目标配置检查](../cross_dataset/configuration.py)
- [现有独立评估入口](../cross_dataset/evaluate.py)
- [现有指标、汇总及可视化](../cross_dataset/evaluation.py)
- [目标训练兼容入口（保持主体）](../cross_dataset/train.py)
- [原模型的数据整理与 forward_test（只读）](../model/omni_gs.py)
- [原渲染器的目标相机循环（只读）](../model/gaussian.py)
- [原指标函数（只读）](../tools/metrics.py)
- [112×200 原配置（只读）](../configs/OmniScene/omni_gs_nusc_novelview_r50_112x200.py)
- [224×400 原配置（只读）](../configs/OmniScene/omni_gs_nusc_novelview_r50_224x400.py)

### SVF-GS（仅参考，不迁入其架构或预处理）

- [18 视角协议与正式报告口径](../../SVF-GS/docs/零样本泛化实验/18视角零样本泛化实验规划.md)
- [PandaSet 资产说明](../../SVF-GS/docs/零样本泛化实验/PandaSet%20数据集适配方案.md)
- [DDAD 资产及自车掩码定义](../../SVF-GS/docs/零样本泛化实验/DDAD%20数据集适配方案.md)
- [协议和掩码配置](../../SVF-GS/configs/build_config.py)
- [18 视角 Dataset](../../SVF-GS/data/temporal_dataset.py)
- [图像、深度与相机张量读取](../../SVF-GS/data/transforms/temporal_loading.py)
- [已处理 DDAD 掩码读取](../../SVF-GS/data/transforms/ego_mask.py)
- [清单/路径/几何契约的只读参考](../../SVF-GS/tools/temporal_data.py)
- [全图与 masked 指标](../../SVF-GS/tools/metrics.py)
- [逐 bin 三视角组记录](../../SVF-GS/tools/ablation_metrics.py)
- [分组汇总及像素协议检查](../../SVF-GS/tools/ablation_results.py)
- [掩码 CPU 测试用例](../../SVF-GS/tests/test_ego_mask_evaluation.py)
