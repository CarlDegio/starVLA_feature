# EDL 真实机器人数据转换与分任务训练

三个真实任务转换成 LeRobot v2.1 目录布局，放在 `playground/Datasets/edl_real/` 下。每个任务单独训练一个 QwenEDL 模型，不混合其他两个任务。

下表为原始录制规模；当前清洗后数据的数量见下文。

| 子集 | 轨迹数 | 帧数 | 视频数 |
| --- | ---: | ---: | ---: |
| `classification_the_blocks` | 100 | 68,957 | 300 |
| `insert_the_two_tubes_into_the_rack_one_by_one` | 103 | 70,679 | 309 |
| `place_the_slippers_on_the_shoe_rack` | 101 | 90,818 | 303 |

## 数据约定

- 三路视频按 **top、left、right** 顺序使用，保存为 **320×240、30 FPS、H.264**。现有加载器继续将帧缩放成 **224×224** 再交给 Qwen 图像处理器；没有修改模型或图像预处理实现。
- 动作和状态都是 14 维，顺序为 **左臂 6 个关节、右臂 6 个关节、左夹爪、右夹爪**。保留源 `action-*.npy` 中记录的控制目标，不做差分；默认 `action_mode: abs`。夹爪保留连续值。
- 预测窗口为 30 步（当前步及后续 29 步动作，即 `action_t` 到 `action_{t+29}`，30 Hz 下约 1 秒），只输入当前时刻三路图像与 state，不输入历史图像或未来 state。窗口在加载时构造，无需重转视频或 Parquet；轨迹末尾不足 30 步时重复最后一个动作补齐。动作按各任务自身的 `q01/q99` 归一化，统计维度顺序与训练动作完全一致。后续部署 real 子集时，应按对应模型保存的统计量与训练变换逆转归一化，保持全部 14 维连续；本次只准备数据和训练方案，未修改部署行为。
- QwenEDL 真机训练输入使用三路图像、任务指令及当前帧 14 维机器人状态，`include_state: true`。状态顺序与动作相同：左臂 6 关节、右臂 6 关节、左夹爪、右夹爪。
- MP4 与数组按原始帧索引对应，训练时间戳为 `frame_index / 30`。各相机原始毫秒时间戳另存于 `observation.camera_timestamp_ms.*`，不据此推断控制时刻或重新同步轨迹。
- 原始 `metadata.json` 的任务名如 `task2` 不够描述动作，默认使用下表指令。可通过 `--instruction-json` 传入 `{ "任务目录名": "具体任务指令" }`。已完成转换可用 `--update-instructions-only` 更新指令；其他转换设置的更改仍需新的输出目录。

| 子集 | instruction |
| --- | --- |
| `classification_the_blocks` | Collect the blocks on the table by color: place the gray blocks in the left basket and the pink blocks in the right basket. |
| `insert_the_two_tubes_into_the_rack_one_by_one` | Insert the test tube into the test tube rack. |
| `place_the_slippers_on_the_shoe_rack` | Place the slippers on the shoe rack. |

只更新已转换数据集的任务文本及续转标记，不重新编码视频、不改原始数据：

```bash
bash examples/LIBERO/train_real/convert_edl_real.sh --update-instructions-only
```

目录示例：

```text
playground/Datasets/edl_real/
├── classification_the_blocks/
│   ├── data/chunk-000/episode_000000.parquet
│   ├── videos/chunk-000/observation.images.top/episode_000000.mp4
│   ├── videos/chunk-000/observation.images.left/episode_000000.mp4
│   ├── videos/chunk-000/observation.images.right/episode_000000.mp4
│   └── meta/
│       ├── info.json
│       ├── modality.json
│       ├── episodes.jsonl
│       ├── episodes_stats.jsonl
│       ├── tasks.jsonl
│       ├── stats.json
│       └── conversion_complete.json
├── insert_the_two_tubes_into_the_rack_one_by_one/
└── place_the_slippers_on_the_shoe_rack/
```

`meta/episodes.jsonl` 记录转换序号与原始轨迹目录的对应关系。首次转换保留全部 304 条轨迹；新版按下述停顿规则过滤，未自动划分验证集或按成功率过滤。

### 开头静止帧裁剪与归一化审计

转换脚本默认检查所有 14 维 **action**：与首帧 action 比较，任何一维持续超出容差即判为移动。默认关节容差 `0.005`、夹爪容差 `0.01`，单位与原始记录一致；连续 3 帧确认，裁掉这 3 帧之前的开头静止前缀，保留移动的第一帧。使用相对首帧的累计位移，避免将缓慢运动误判为静止；不是要求全部关节同时移动。容差是显式的处理参数，不是从归一化统计推导的阈值，也不用于改变 action 值。

参数为 `--idle-joint-threshold`、`--idle-gripper-threshold`、`--motion-confirm-frames`；`--no-trim-leading-idle` 可禁用前缀裁剪。三路视频、action、state 和相机时间戳同步裁剪，训练时间戳和 frame/index 从零重新编号。额外保存 `source_frame_index` 列，轨迹元数据记录 `source_start_frame`、`source_num_frames` 和 `trimmed_leading_frames`，便于回溯。全局统计从保留帧重新计算，mean/std 使用 float64 累加。

**中间停顿至少 1 秒则排除整条轨迹**，由 `--max-idle-seconds 1` 控制（0 禁用）。在滑动窗口内，每一维的 `max-min` 均不超过上述容差才算停顿，因此正常单臂操作不会仅因另一臂恒定而被排除。30 Hz 下使用 31 帧覆盖 1 秒；只计首次运动之后的停顿。**停顿起点落在原始轨迹最后 3 秒内时，不因此排除整条轨迹**，由 `--end-idle-grace-seconds 3` 控制；不会因为停顿与最后 3 秒有部分重叠就豁免更早的停顿。

开头长停顿只裁前缀；连续静止到结束的尾段也裁掉，保留尾段的第一帧作为最终动作目标。尾段需至少连续 3 帧且全段所有维度的范围均在容差内。末尾停顿后如又有动作，会保留该动作，不把最后 3 秒一律删除。三路视频、action、state、时间戳按同一 `[source_start_frame, source_end_frame)` 裁剪，并分别记录 `trimmed_leading_frames` 和 `trimmed_trailing_frames`。整条无持续运动、记录不完整或帧数不一致也会排除，原因和源帧区间记录到 `meta/excluded_episodes.jsonl`；`conversion_plan.jsonl` 保存所有源轨迹的处理决定，保留轨迹重新连续编号。不会从原始目录删除文件。

**小 action 范围采用均值对称扩展，不据此删除轨迹或维度**：若原始 `q99-q01 < 0.01`，设置归一化下界 `q01 = mean-0.005`、上界 `q99 = mean+0.005`；其余维度保持原范围，std 保留真实值。`--min-action-quantile-span` 可调整下限（0 禁用）。全局和逐轨迹 action 统计均应用此规则；state 统计不扩展。原始经验分位数保存在 `stats.raw.json` 和 `episodes_stats.raw.jsonl`。扩展后的 q01/q99 是归一化边界，不再是这些窄范围维度的原始经验分位数。

训练实际读取 `meta/stats_gr00t.json`，转换器同步生成绝对动作模式的统计缓存，确保训练与导出统计使用扩展后的边界。真实数据训练前置检查拒绝缺失或不一致的缓存，可用相同转换命令续转重建。当前训练仍按各任务的全局统计归一化，不改为逐轨迹归一化。

先生成无视频编码的全量审计报告，不修改已有数据：

```bash
playground/.venvs/edl-real/bin/python examples/LIBERO/train_real/audit_edl_real.py \
  --report-dir playground/Datasets/edl_real_audit_final_20261002
```

报告包含 `normalization_report.json`、完整的 `global_dimensions.csv`、逐轨迹异常维度 `episode_flags.csv` 和裁剪起点 `trimming_preview.csv`。默认将原始单位下 `q99-q01 <= 0.01` 或 `std <= 0.001` 标记为待检查，可用 `--small-quantile-span` 和 `--small-std` 调整。报告区分原始数据、只裁前缀和排除中间停顿后的统计，同时给出拟采用的归一化边界；审计命令不修改数据集。

2026-10-02 对全部 304 条轨迹检查：

| 子集 | 排除轨迹数 | 排除整轨迹帧数 | 开头裁剪帧数 | 结尾裁剪帧数 | 最终轨迹数 | 最终帧数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| classification_the_blocks | 1 | 747 | 1,122 | 1,848 | 99 | 65,240 |
| insert_the_two_tubes_into_the_rack_one_by_one | 17 | 11,368 | 1,833 | 957 | 86 | 56,521 |
| place_the_slippers_on_the_shoe_rack | 1 | 730 | 2,422 | 1,021 | 100 | 86,645 |

合计排除 19 条轨迹，保留 **285 条、208,406 帧、855 段视频**；整轨迹排除和首尾裁剪共减少 22,048 帧（约 9.57%）。该次检查无缺失或帧数不一致记录。任务级 action 全局统计在处理前后均没有小于 0.01 的范围；逐轨迹统计在最终保留数据中，积木有 3 条轨迹共 22 个维度需要扩展，插管有 86 条共 236 个维度，拖鞋为 0。这些扩展本身不删除任何帧。

积木轨迹 `20260930_190608_6de40956` 的开头 533 帧会作为前缀裁剪，但它在源帧 562–600 又有约 1.27 秒中间停顿，因此仍因中间停顿被排除，并非因为开头停止时间长。

当前完整数据已通过全量数据检查及训练加载验证，并替换正式 `playground/Datasets/edl_real`。转换使用 tmux `edl_real_rebuild`，日志和清理记录在 `playground/Datasets/.download_tasks/edl_real_rebuild_20261002/`，验证报告在 `edl_real/validation_report.json`。旧版完整转换及 5 个转换短测目录已按用户要求删除，原始录制和训练 checkpoint 保留。清洗后的新训练作业见下文。转换设置已纳入续转签名，新设置不能复用旧视频/Parquet。后续若更改规则，应先用新的输出目录，例如：

```bash
PYTHON="$PWD/playground/.venvs/edl-real/bin/python" \
bash examples/LIBERO/train_real/convert_edl_real.sh \
  --output-root playground/Datasets/edl_real_trimmed \
  --ffmpeg /mnt/workspace/liuzihao/miniconda3/envs/RoboTwin/bin/ffmpeg
```

## 转换与检查

从仓库根目录执行。需要 Python、FFmpeg（含 libx264）以及轻量转换依赖，不必安装 LeRobot：

```bash
python -m pip install -r examples/LIBERO/train_real/requirements-convert.txt
bash examples/LIBERO/train_real/convert_edl_real.sh --output-root playground/Datasets/edl_real_trimmed --workers 4
```

可用 `PYTHON=/path/to/python` 指定解释器，或传 `--ffmpeg /path/to/ffmpeg`。默认输出就是 `playground/Datasets/edl_real`，原始三个目录保持不变。已完成轨迹会按源文件签名、转换设置和输出大小复用；转换中断后执行相同命令续转。被排除轨迹明确记录原因；编码失败等执行错误仍中断，不会留下伪装成成功的数据集。

检查每条轨迹的动作、状态、时间戳与源文件保留区间逐值一致，以及全部视频的尺寸、FPS、帧数：

```bash
python examples/LIBERO/train_real/validate_edl_real.py
```

在安装了仓库数据加载依赖的环境中，额外检查现有训练加载器、224×224 图像、30×14 动作窗口、轨迹末尾填充、FAST 编码/解码与部署统计顺序：

```bash
python examples/LIBERO/train_real/validate_edl_real.py --sample-training
```

结果写入 `playground/Datasets/edl_real/validation_report.json`。

## 三个独立训练方案

`configs/` 下每个任务有自己的 YAML，分别指定 `data_mix: edl_real_<任务名>` 和独立 `run_id`。注册定义在本目录 `data_config.py`，由 LIBERO 原有 `train_files/data_registry/data_config.py` 导入。

### 当前帧 joint/gripper state 输入

三个真机 YAML 同时启用 `datasets.vla_data.include_state: true` 和 `framework.state_input.enabled: true`，`framework.action_model.state_dim: 14`。加载器只取当前帧（`delta_indices=[0]`），输出 `(1, 14)`，不使用未来状态。14 维顺序是 `[left_joint_0..5, right_joint_0..5, left_gripper, right_gripper]`。

state 使用本任务的 `observation.state` 统计量，而非 action 统计量，逐维计算 `2 * (state - q01) / (q99 - q01) - 1`，截断到 `[-1, 1]`；`q01 == q99` 的常量维设为 0。复用现有 Qwen 文本输入方式：归一化 state 离散为 256 档，作为 `[STATE] <14 个档位> [ACTION]` 加入指令。训练 forward 和训练期间的动作评估/预测共用此逻辑，state 不作为动作预测标签，不新增 state encoder 或模型权重。

QwenEDL 的 state 开关默认关闭，LIBERO 的 YAML、脚本和输入保持原样。新的真机默认 run_id 带 `_state` 后缀，避免混用原有不含 state 的实验。做无 state 对照时，需同时覆盖 `--datasets.vla_data.include_state false --framework.state_input.enabled false`，并使用独立 `RUN_ID`。

已经转换的数据包含 state 及其统计量，无需重新转换。`preflight_real.py` 检查 state 配置、维度顺序与统计缓存，`validate_edl_real.py --sample-training` 额外检查归一化数值及导出 state 统计。部署这类新模型时也必须提供相同顺序的当前实测 state，并按 checkpoint 的 state 统计与训练变换归一化后传给 QwenEDL；目前共享 policy server 仅自动处理动作反归一化，不会替客户端归一化 state。

验证：

```bash
python -m unittest discover -s examples/LIBERO/train_real -p test_state_input.py
```

此前在 15 步窗口下，本机已用 `playground/Datasets/edl_real` 的三个任务（285 条轨迹、208,406 帧）完成预检查，并逐任务抽取起始、中间、末尾帧，共 9 个样本，核对当前 state、独立归一化统计、三路 224×224 图像及 15×14 动作窗口。每个任务的中间样本均通过 Qwen3-VL-4B + FlashAttention 2 前向/反向，loss 和 embedding 梯度有限，单样本峰值显存 18.53 GiB。记录在 `playground/.env_setup/starvla_20261009/real-state-verification.json`；这验证训练链路，不代表训练后的真机成功率。

此前在 15 步窗口下，在当前 4 张 B200 上，使用积木真实数据、每卡 10 个 worker、bf16、FlashAttention 2、ZeRO-2、关闭激活检查点、梯度累积 1，分别对每卡 BS 16 和 BS 32 做了 5 步前向/反向及 AdamW 更新短测，均正常退出且未保存 checkpoint。BS 16 每卡张量显存峰值为 68.7–69.3 GiB，`nvidia-smi`（含缓存和通信开销）峰值为 88.3–93.8 GiB；BS 32 分别为 115.2–116.0 GiB 和 135.3–157.7 GiB。当前默认 BS 32、4 卡全局 BS 128；学习率不自动扩大。记录在 `playground/.env_setup/starvla_20261009/bs16-memory.json` 和 `bs32-memory.json`，实际显存会随 batch 的动作 token 长度变化，短测不代表完整训练全程的最大值。

当前 30 步窗口已逐任务检查起始、中间、末尾帧，共 9 个真实样本，确认动作从当前帧开始、末尾重复最后动作补齐、FAST 解码形状为 `(1, 30, 14)`，state 仍只取当前帧。记录在 `playground/.env_setup/starvla_20261009/h30-data-verification.json`。使用相同 4 卡、每卡 BS 32 设置又完成 5 步训练更新并正常退出，每卡张量显存峰值为 119.4–120.1 GiB，`nvidia-smi` 峰值为 161.2–166.5 GiB。排除首步初始化后，计算更新平均约 1.15 秒/步（不含等待读取 batch、评估和保存）；仅为短测。记录在 `playground/.env_setup/starvla_20261009/bs32-h30-memory.json`。数据文件和归一化统计不依赖窗口长度，无需重新转换。

参考 `playground/Checkpoints/qwen3fast_libero_all_edl_1e-2/config.full.yaml`：

- 同一 `Qwen3-VL-4B-Instruct-Action` 基座与 FAST tokenizer，使用 QwenEDL 的显式 state 输入开关；不增加模型参数。
- 30,000 steps、5,000 warmup、每设备 batch 32、**不累积梯度**（`gradient_accumulation_steps: 1`）、每 15,000 steps 保存。每次读取一个 batch 就更新一次；默认 4 卡，全局 batch 为 128。
- 真机默认 `trainer.save_final_model: false`，只保存 `checkpoints/steps_15000_pytorch_model.pt` 和 `steps_30000_pytorch_model.pt`；结束时不重复保存 `final_model`。专用入口仍完成 WandB flush 和各进程同步，原 LIBERO 入口保持原样。
- 每个训练进程使用 10 个 DataLoader worker 子进程（`datasets.vla_data.num_workers: 10`），4 卡共 40 个 worker；每个 worker 预取一个 batch。共享加载器在未指定该参数时仍使用 4 个 worker，LIBERO 原配置不受影响。
- VLM 学习率 `1e-5`、基础学习率 `2.5e-5`、FAST 模块配置学习率 `1e-4`，不因 batch 或卡数变化自动缩放。当前 FAST tokenizer 无可训练参数，实际更新的是 VLM 参数。AdamW 配合 `cosine_with_min_lr`：前 5,000 步线性 warmup，随后余弦衰减，到第 30,000 步降至 `1e-6`；不冻结 VLM。
- 动作维度为真实双臂所需的 14，预测窗口为 30 步（`action_horizon: 30`、`future_action_window_size: 29`、`past_action_window_size: 0`）。生成预算保持 256 tokens。
- `framework.edl` 显式记录 `digamma`、top-k 25、KL 权重 **0.01**、annealing 15,000、`softplus`。

参考实验的 YAML 未记录 EDL loss 参数，当前 `QwenEDL.py` 内硬编码 KL 权重为 0。专用 `train_edl_real.py` 入口在构建后设置已有 loss 属性，让新 YAML 的 `0.01` 真正生效；EDL loss 适配不改动模型参数键或原 LIBERO 训练入口。运行这套 YAML 时应使用专用入口，直接交给原 `train_starvla.py` 不会应用新 EDL loss 参数。

专用入口在导入共享 trainer 前强制将 Accelerator 梯度累积设为 1，与共享 DeepSpeed 配置的 1 一致。前置检查拒绝其他累积值；自定义 DeepSpeed 配置若不为 1 或 `auto`，也会在启动时拒绝。

最终方案使用 `gradient_checkpointing: false`，专用入口明确关闭 Qwen 的激活检查点。模型构建后进入训练模式；动作评估期间切换为 eval 模式使用生成缓存，评估后恢复训练模式。这些设置仅作用于 real 训练入口。

当前 `PolicyServerWrapper` 使用 `PolicyNormProcessor` 按注册的数据配置逆转训练变换，并不调用旧的公共 `FrameworkTools.unnormalize_actions`。LIBERO 客户端在环境适配时将夹爪按 0.5 阈值映射成 -1/+1；real 部署客户端不能直接沿用这个单臂适配。

旧的公共 `FrameworkTools.unnormalize_actions` 保持原样，默认仍将索引 6 当夹爪二值化。如果后续 real 子集的部署代码选用这个旧助手，可显式传 `gripper_channel_idx=-1` 跳过二值化，并按选定的 `q01/q99` 和 `mask` 恢复连续动作；它原有的 `[-1, 1]` 截断规则仍适用。本次未改任何现有部署脚本。

默认三个模型分别从同一基础 VLM 初始化，未自动加载 LIBERO 实验的已训练权重。

三个任务各用自己的 `data_mix`，每个 mixture 只包含对应任务的数据；每次启动创建独立模型、独立 checkpoint 目录和独立 wandb run。`all` 是依次启动三次独立训练，不会把任务混在一个模型中训练，也不会自动继承前一个任务的模型权重。

wandb 项目为 `starVLA_EDL_Real`，entity 为 `carldegio`，run 名为各任务的 `run_id`。专用训练入口优先使用已有的 `WANDB_API_KEY` 环境变量，否则读取仓库根目录的 `.wandb_api_key`（已被 `.gitignore` 忽略，文件权限为 600）；可用 `WANDB_API_KEY_FILE` 指定其他 key 文件。凭证只放入进程环境，不写入训练 YAML、脚本副本或模型配置。

启动脚本和专用 Python 入口均清除大小写的 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY` 和 `NO_PROXY`，以及 `WANDB_HTTP_PROXY`、`WANDB_HTTPS_PROXY`，训练及 wandb 使用服务器直连，不依赖客户端的代理隧道。tmux 可在客户端断开或关机后继续运行；训练服务器本身关机则不能继续执行。

在已有 StarVLA 训练环境中分别执行：

```bash
bash examples/LIBERO/train_real/run_real_train.sh classification_the_blocks
bash examples/LIBERO/train_real/run_real_train.sh insert_the_two_tubes_into_the_rack_one_by_one
bash examples/LIBERO/train_real/run_real_train.sh place_the_slippers_on_the_shoe_rack
```

按顺序训练三个模型：

```bash
bash examples/LIBERO/train_real/run_real_train.sh all
```

当前机器有 4 张 B200，默认每次使用 4 个训练进程，每卡 batch 32；`NUM_PROCESSES` 可覆盖卡数。指定可见设备、改训练步数或先打印命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_PROCESSES=2 bash examples/LIBERO/train_real/run_real_train.sh classification_the_blocks
bash examples/LIBERO/train_real/run_real_train.sh all --dry-run
bash examples/LIBERO/train_real/run_real_train.sh classification_the_blocks --trainer.max_train_steps 10000 --trainer.num_warmup_steps 1000
```

模型分别保存到 `playground/Checkpoints/qwen3fast_edl_real_<任务名>_edl_1e-2_state/`。已有权重的目录默认拒绝覆盖；设置不同 `RUN_ID` 建立新实验，或用 `RESUME=1` 沿用仓库原有模型权重和步数恢复逻辑（不表示完整恢复 optimizer/scheduler 状态）。`RUN_ROOT_DIR`、`MAIN_PROCESS_PORT`、`ACCELERATE_CONFIG`、`PYTHON` 也可通过环境变量设置。

## 已取消的三任务队列（2026-10-10）

原 tmux `edl_real_state_h30_train_4gpu` 的积木、插管、拖鞋三任务队列已根据用户要求取消。积木停止在约第 1,417 步，尚未保存 checkpoint；插管、拖鞋还未启动。原日志、配置快照和 WandB 记录保留，旧队列不会自动继续，四个训练进程及其数据 worker 已退出。

队列脚本、源码和配置快照、日志位于 `playground/Checkpoints/.train_tasks/real_state_h30_bs32_4gpu_20261010/`。`status` 记录当前任务，`completed_tasks` 在任务正常退出并确认两个 checkpoint 存在后追加；任一任务失败即停止队列。

```bash
cat playground/Checkpoints/.train_tasks/real_state_h30_bs32_4gpu_20261010/status
tail -f playground/Checkpoints/.train_tasks/real_state_h30_bs32_4gpu_20261010/classification_the_blocks.log
```

## 当前拖鞋任务 KL 对比队列（2026-10-10）

tmux `edl_real_slippers_kl_compare_4gpu` 按顺序运行拖鞋任务的两个独立实验：`framework.edl.kl_weight=1e-2`，然后 `1e-3`。对应 run_id 为 `qwen3fast_edl_real_place_the_slippers_on_the_shoe_rack_edl_1e-2_state` 和 `qwen3fast_edl_real_place_the_slippers_on_the_shoe_rack_edl_1e-3_state`。两个实验从相同基座重新初始化、seed 42，除 KL 权重和输出名称外配置完全一致；保持同一拖鞋数据、30 步动作窗口、14 维当前 state、4 卡、每卡 BS 32、10 个 worker、VLM 峰值学习率 `1e-5`、15,000 步 KL annealing 和不累积梯度。

每个实验训练 30,000 步，仅保存 15,000 与 30,000 步 checkpoint。WandB 在 `carldegio/starVLA_EDL_Real` 在线记录，启动时清除代理变量。任一实验失败即停止，不继续下一个。

队列脚本、两个解析后的实验 YAML、源码快照和日志在 `playground/Checkpoints/.train_tasks/slippers_kl_compare_h30_bs32_4gpu_20261010/`。`status` 记录当前实验，`completed_experiments` 在训练结束并确认两个 checkpoint 和实际 KL 配置后追加。正式训练以各模型目录的 `config.full.yaml` 为准；第二个实验通过启动参数覆盖默认拖鞋 YAML 的 KL 权重。

```bash
tmux attach -t edl_real_slippers_kl_compare_4gpu
cat playground/Checkpoints/.train_tasks/slippers_kl_compare_h30_bs32_4gpu_20261010/status
tail -f playground/Checkpoints/.train_tasks/slippers_kl_compare_h30_bs32_4gpu_20261010/slippers_kl_1e-2.log
```

## 历史服务器的单机 8 卡作业（2026-10-02）

以下为原服务器的环境、BS 8 配置与运行记录，当前机器的默认配置以上面的 4 卡、每卡 BS 32、10 个 worker 为准。

已准备独立环境 `playground/.venvs/edl-real`，复用 RoboTwin 环境的 PyTorch 2.7.1/CUDA 12.8；训练依赖安装在独立环境中，不修改 RoboTwin。专用 CUDA 编译工具链在 `playground/.tools/cuda`。基础训练依赖为 Transformers 4.57.0、Accelerate 1.5.2、DeepSpeed 0.16.9、FlashAttention 2.8.3。FlashAttention 使用与 PyTorch 2.7、Python 3.10、CXX11 ABI 匹配的官方 wheel，已在 H20 上通过 bf16 前向/反向测试。真实训练前置检查要求已安装请求的 FlashAttention，入口打印实际 attention backend。

8 卡运行时每卡 BS 8、不累积梯度，总 BS 为 64，激活检查点关闭，学习率仍按 YAML 配置，不自动随卡数扩大：

```bash
PYTHON="$PWD/playground/.venvs/edl-real/bin/python" \
CUDA_HOME="$PWD/playground/.tools/cuda" \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_PROCESSES=8 \
bash examples/LIBERO/train_real/run_real_train.sh all
```

本次 tmux 会话为 `edl_real_train_8gpu`。最终 BS8/FlashAttention/无激活检查点的作业脚本与日志在 `playground/Checkpoints/.train_tasks/edl_real_8gpu_flash_bs8_20261002/`：先做 4 步短训练（包含评估、保存和 wandb 上传），成功后依次训练三个完整的单任务模型；任何阶段失败时停止，不继续下一任务。`status` 记录当前阶段，`smoke.log` 和各任务的 `.log` 记录完整输出，`completed_tasks` 记录已完成任务。此前各次短测和中断作业的日志、权重保留在原目录。

2026-10-02 已按用户要求停止上述正式训练及整个三任务队列，等待数据处理与归一化审查；不会自动启动下一任务。现有日志和模型产物保留。

2026-10-02 完成全量清洗转换后，用户已授权重新正式训练。新 tmux 会话为 `edl_real_train_cleaned_8gpu`，队列脚本、每任务日志和数据处理元数据快照在 `playground/Checkpoints/.train_tasks/edl_real_cleaned_8gpu_20261002/`。三个任务按积木、试管、拖鞋顺序各训练 30,000 步，每个模型从同一基座重新初始化，run_id 为 `qwen3fast_edl_real_<任务名>_edl_1e-2_cleaned_20261002`，不续训旧数据实验。8 卡、单卡 BS 8、无梯度累加、无激活检查点、FlashAttention、15 步动作窗口和原学习率保持不变，代理环境变量清除，WandB 在线记录。任一任务失败时队列停止。

```bash
tmux attach -t edl_real_train_cleaned_8gpu
cat playground/Checkpoints/.train_tasks/edl_real_cleaned_8gpu_20261002/status
```

```bash
tmux attach -t edl_real_train_8gpu
cat playground/Checkpoints/.train_tasks/edl_real_8gpu_flash_bs8_20261002/status
```
