# edl_real 虚拟 client：逐 chunk 比较预测与 GT

只读取 LeRobot v2.1 转换数据，通过现有 StarVLA real WebSocket server 推理。
不打开物理相机、不导入机器人驱动、不连接 GUI、不执行真实动作。

## 运行

模型服务使用现有入口。拖鞋 state 模型仍预测 30 步，server 默认只返回前 15 步、30 Hz：

```bash
cd /home/tiancai/liuzihao/starVLA_feature
./policy_server_real.zsh serve --task slippers
```

另一个终端运行虚拟 client。已有同任务 server 时直接使用，不要重复启动：

```bash
cd /home/tiancai/liuzihao/starVLA_feature
bash examples/LIBERO/virtual_client/run.sh --task slippers --episode 0
```

默认数据根目录为 `playground/Datasets/edl_real`。`--data-root` 也可指定其他 edl_real
根目录或单个任务子目录。`--episode` 是转换后的 episode_index，从 0 开始；
原始轨迹名、裁剪前的 source_frame_index 会写入报告。支持 `blocks`、`tubes`、`slippers`，
server 必须与 `--task` 对应，horizon 和 fps 会核对；输入 state 为原始值，归一化由服务端负责。

按回车推进、只看开头几个 chunk，或指定起始帧：

```bash
bash examples/LIBERO/virtual_client/run.sh --task slippers --episode 0 --step
bash examples/LIBERO/virtual_client/run.sh --task slippers --episode 1 --max-chunks 5
bash examples/LIBERO/virtual_client/run.sh --task slippers --episode 0 --start-frame 300
```

`--step` 在每个 chunk 预测并保存后等待 Enter；输入 q 会结束并生成已完成部分的报告。
也支持 `--host`、`--port`、`--timeout` 和 `--output`。输出目录不能已存在，以免覆盖比较结果。

## 推进与比较的含义

默认 `model_horizon=30, horizon=15`：

1. 读取第 0 帧 top/left/right RGB 图像、原始 GT state 和数据集指令。
2. 模型完整预测 30 步，server 返回前 15 个原始绝对 action，与 GT 的第 0～14 帧比较，保存结果。
3. 再读取第 15 帧观测，比较第 15～29 帧；之后读取 30、45……帧。
4. 最后不足 15 帧时仍保存服务返回的 15 步预测，只对 episode 范围内的有效 GT 评分，不填充尾部计分。

client 自动跟随 server 返回的 `horizon`，无需另设跳步参数。旧进程需重启 server 才会生效；
启动时会打印模型 horizon 和实际帧步长，报告也保存 `frame_stride`。
用 server 的 `--execution-horizon 30` 可以恢复原先每 30 帧推理。
real server 默认 `--fast-decode-mode truncate`，长度异常时将实际 DCT 系数补零或截断到 420 个，
固定按 30×14 做逆 DCT，再返回前 15 步。补零在还原 `min_token` 偏移后进行，长度不足不再拒绝。
`--fast-decode-mode strict` 可以拒绝全部长度错误。含 U+FFFD 或重构结果非有限值仍明确报错。
虚拟 client 对 U+FFFD 非法字符拒绝默认额外重试 3 次（总共最多 4 次），自动重连并核对模型 metadata；
重试始终使用同一帧图像和 state，确定性生成下结果通常不会改变。`--invalid-token-retries 0` 关闭重试。
四次均失败或遇到其他错误时停止，
已写入的 chunk 保留，summary 标记 error；可用 `--start-frame` 从其他帧开始独立测试。
EDL token 诊断仍对应完整模型预测；`diagnostic__full_model_actions` 保留恢复后的完整 30 步序列，
`diagnostic__fast_decode_*` 保存原系数长度、截断量、补零量和固定的解码步数。

图像按转换数据的帧索引顺序解码，RGB 格式；state/GT 从训练数据的
`左臂6、右臂6、左夹爪、右夹爪` 重排为协议使用的
`左臂6、左夹爪、右臂6、右夹爪`。服务端输出已经反归一化，比较时不再进行归一化或限速。
训练使用当前动作加未来动作，因此 GT 对齐从当前帧开始，没有额外 +1 偏移。

这里的“执行完 action”是虚拟时间前进一个 horizon；默认不按墙钟等待 0.5 秒，也不模拟运动学。
下一次读取的始终是数据集真实观测，预测动作不会改变下一次图像或 state。
它可以验证输入链路、轨迹拟合和固定输出问题，不能测闭环成功率。
若该 episode 用于训练，误差仅代表训练数据回放表现，不能据此判断泛化或排除过拟合。

## 结果

默认输出到本目录 `outputs/<task>_ep<index>_<timestamp>/`：

- `report.html`：浏览器直接打开。完整 14 维轨迹图，下方可展开每个 chunk 的输入图像、
  误差和完整预测/GT 数组（不只显示末步，也不截断数值）。
- `comparison.png`：每个 joint/夹爪的预测与 GT 对比曲线；竖线标记 chunk 分界。
- `comparison.csv`、`comparison.npz`：拼接后的有效帧预测、GT；NPZ 还包含 GT state 和列名。
- `summary.json`：模型 checkpoint、数据来源、全程 joint MAE/RMSE/max error、夹爪 MAE、
  每维 MAE、是否读完整条 episode、恒定/重复 chunk 数。
- `chunks.jsonl`：逐 chunk 误差、三路图像 SHA256/是否变化、原始 state 是否变化、
  `all_actions_identical`（chunk 内所有行完全相同）、
  `same_as_previous_prediction`（本次整个 chunk 与上次完全相同）。判断使用原始数组，而非显示精度。
- `chunk_XXXXXX.npz`：本次完整预测、有效 GT、输入 state、三路原始图像及 server 返回的数值诊断。
- `chunk_XXXXXX_images.png`：本次 top / left / right 三路图像拼图。

关节误差和速度以 rad、rad/s 计，夹爪单独以归一化值计，不混算。
Ctrl+C/异常时保留已写入的逐 chunk 数据，summary 标记 interrupted/error；不将其报告为完整轨迹。

## 严格 FAST 解码审计（2026-10-11）

使用 `--execution-horizon 30 --fast-decode-mode strict` 同时关闭执行前缀截取和异常系数长度恢复。当前普通虚拟 client 在首个拒绝 chunk 处停止；为统计整条轨迹，可使用 `audit_fast_decode`，它在进程内复用 real server 的 `load_policy` / `YAMStarVLAPolicy` 与虚拟 client 的 `Episode`，逐 chunk 捕获原始生成 token、EOS、DCT 系数长度，记录拒绝后继续。无效 chunk 不生成动作文件、不补零、不截断。

```bash
OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 python -m examples.LIBERO.virtual_client.audit_fast_decode \
  --ckpt playground/Checkpoints/qwen3fast_edl_real_place_the_slippers_on_the_shoe_rack_edl_1e-3_state/checkpoints/steps_30000_pytorch_model.pt \
  --episode 0 --stride 15 --teacher-forced-chunks 6 --output playground/fast_decode_audit
```

本机 `steps_30000` 复现：第 0 条轨迹 827 帧，每 15 帧采样，共 56 个完整 30 步预测，39 个严格解码失败（69.64%）；其中每 30 帧采样的 28 个 chunk 有 19 个失败（67.86%）。全部 GT 编码得到 420 个合法系数；全部输入图像、语言和归一化 state 与训练读取器一致，前 6 个训练 prompt 前缀与生成 prompt 逐 token 相同。生成无一次达到 256 token 上限，全部以 EOS 结束。前 6 个样本在真实前文下的逐 action-token 预测正确率约 72.92%，其拼接结果均不满足严格解码长度。

训练 forward 监督的是 GT FAST token；评估实际生成预测时，原始 FAST `decode()` 捕获长度错误并返回零动作，训练继续，MSE 也未单独记录解码有效率。拖鞋 `1e-3` 训练日志有 3,098 次此类错误；300 次评估 × 4 卡 × 每卡 4 个 chunk，共 4,800 个，约 64.54%。因此高失败率在训练评估中已存在。零归一化动作经 q01/q99 逆变换得到分位区间中心，不是恢复成功的轨迹或数据均值。

原始审计结果在 `playground/.env_setup/fast_decode_audit_20261011/`：`slippers_ep0_strict_run2/summary.json`、逐 chunk JSONL/原始 token NPZ、训练日志错误统计，以及严格 WebSocket client 的失败报告。420 指逆 BPE 后的 DCT 系数数量，不是模型必须生成 420 个 FAST token。模型未受到整条序列解码长度约束，token 预测错误可能改变展开系数个数；本次证据支持模型输出结构错误，而非该轨迹输入预处理不一致。

## 依赖与测试

启动器使用 `starvla` 环境。本机已补装 pyarrow、matplotlib，原环境已提供 PyAV。
其他机器可在推理环境安装：

```bash
python -m pip install -r examples/LIBERO/virtual_client/requirements.txt
python -m unittest examples.LIBERO.virtual_client.test_client -v
```

测试使用合成 Parquet、RGB 视频及伪 policy：验证 0/3/6 帧推进、GT 同帧对齐、关节重排、
尾部仅一帧的统计、输入变化但预测恒定的识别，以及拒绝错任务和覆盖输出。
