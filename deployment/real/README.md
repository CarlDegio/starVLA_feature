# StarVLA real → YAM 推理

当前真机采集使用 **单次 horizon + 人工 episode 标签**：
模型服务 `./policy_server_real.zsh serve --task blocks`，
GUI `./policy_gui_real.sh`，可选键盘控制 `./policy_client_real_manual.sh`。
复制目录 `run_gui.sh` 已切换到这个专用界面。
详细的 n / Stop episode / y、n、drop、返回本轮起点和 EDL 数据格式见
[真机采集说明](../../examples/LIBERO/edl_pred_real/README.md)。

此入口加载当前三个单任务 QwenEDL checkpoint，复用 `PolicyServerWrapper`，
通过 OpenPI WebSocket 协议直接接入 YAM 客户端。服务、offline、client 模式
都不导入机械臂驱动，也不会发送机器人指令。

## 启动

2026-10-10 state 接入验证：新 slippers 30000-step checkpoint 在录制样本上输出有限的
`30×14` 动作（单次约 1.42 秒），没有发送机器人指令。
state 重排、独立统计量归一化、常量维度、裁剪和训练文本一致性测试，
30 步 mock GUI/保存/导出以及旧 15 步兼容测试，共 20 项通过。

```bash
cd /home/tiancai/liuzihao/starVLA_feature
./policy_server_real.zsh serve --task blocks
# 或 tubes / slippers；切换前先停止当前服务
```

默认模型完整解码后仅返回前 **15 步**（30 Hz，标称执行窗口 0.5 秒）。
`--execution-horizon 30` 可恢复 30 步模型的完整输出；不能超过 checkpoint 的 horizon。
握手中 `model_horizon` 是模型预测长度，`horizon` / `action_chunk_size` / `open_loop_horizon`
是实际返回和执行长度。real server 默认 `--fast-decode-mode truncate`：
正常长度保持原解码；长度错误时保留最多模型 horizon 行的完整 14 维 DCT 系数行，
不补零，以实际保留行数做逆 DCT，再取前 `execution_horizon` 步。
例如 418→406 系数→29 步→前 15 步；421→420 系数→30 步→前 15 步。
这会改变短序列的时间基底和幅值缩放，并不等于可靠地删除错误时间步。
含 Unicode 替换字符 U+FFFD、完整行不足执行长度时明确报错，不回退到零动作或统计中点。
`--fast-decode-mode strict` 则拒绝所有长度错误。普通 LIBERO 服务和训练逻辑不变。
EDL token 诊断仍对应完整生成序列；详细诊断中的 `full_model_actions` 保留本次实际解码的
完整反归一化动作（YAM 顺序，可能是 29×14 等）。`fast_decode_recovered`、
`fast_decode_coefficient_count`、`fast_decode_used_coefficients`、
`fast_decode_dropped_coefficients`、`fast_decode_horizon` 记录恢复情况。
不可恢复的请求在服务日志及错误响应中包含原因、系数数目和 FAST token IDs；客户端不应执行。

默认 `127.0.0.1:8002`、GPU 0、BF16、`starvla` conda 环境，本地离线加载。
默认用对应任务的一份录制样本预热，监听成功才打印 `READY`。
不依赖录制样本启动可加 `--skip-warmup`。跨机器连接可加 `--host 0.0.0.0`。

```bash
curl --noproxy '*' http://127.0.0.1:8002/healthz
```

可用 `GPU_ID`、`PORT`、`HOST`、`TASK`、`CKPT`、`STARVLA_ENV` 环境变量，
或 `--port`、`--host`、`--task`、`--ckpt` 参数覆盖；显式参数优先。
自定义 checkpoint 必须符合对应任务和 14 维 absolute joint targets 配置。
支持旧图像/文本模型，以及 `framework.state_input.enabled=true` 的 QwenEDL state 模型。
`--task slippers` 默认加载新上传的
`qwen3fast_edl_real_place_the_slippers_on_the_shoe_rack_edl_1e-2_state/checkpoints/steps_30000_pytorch_model.pt`。
`blocks`、`tubes` 默认仍为原 15 步模型；`--ckpt` 可以显式指定旧拖鞋模型或其他 step。
原 `policy_server.zsh` 继续提供 StarVLA `examples` / `data` 协议。

复制目录也提供同一入口：

```bash
cd /home/tiancai/liuzihao/starvla-yam-inference
./run.sh serve --task blocks
./run.sh serve --task tubes
./run.sh serve --task slippers
```

以上三个命令为备选，不能同时占用同一端口。
复制目录的 `run.sh` / `infer.py` 委托本仓库实现，不加载 π0.5 或复制一套适配逻辑。
原 π0.5 入口保留为 `run_pi05.sh` / `infer_pi05.py`，说明保留为 `README.pi05.zh.md`；
`weights/`、`policy.py`、原 `manifest.json`、`SHA256.json` 及历史 `validation/`
仍属于 π0.5，不代表 StarVLA 的权重或验证结果。

## YAM 客户端参数

新的 `run_gui.sh` 复用 `/home/tiancai/yam-abc-reproduce` 的会话、相机和急停功能，
打开 StarVLA 专用手动采集界面（8042），不提供连续执行按钮。
若使用保留的 `run_gui_legacy.sh`，原 GUI 的 **Client** 区域仍兼容下列参数；
原 **Start Server** 按钮启动的是其他模型，不负责本服务。

| 参数 | 值 |
|---|---|
| policy host / port | `127.0.0.1` / `8002` |
| control_hz | `30`，当前 YAM station 已是此值 |
| open_loop_horizon | 跟随服务返回长度，默认 `15`（模型本身仍可预测 `30`） |
| action_stride / execution_speed | `1` / `1.0` |
| RTC | 关闭 |
| 客户端图像 resize | 不预处理，服务端统一 resize |

专用手动 GUI 校验 horizon/频率并固定每次执行一个完整 chunk。
原 GUI 不会自动应用服务 metadata，连接时需核对表中参数。
旧模型训练配置是 `action_horizon=15`、`future_action_window_size=14`、`past_action_window_size=0`。
因此每个样本包含当前动作加后续 14 个动作；按 30 Hz 采样周期计作 15/30=0.5 秒的窗口，
相邻首末采样点实际间隔是 14/30≈0.467 秒。**30 Hz 是动作采样/客户端控制频率，不是模型推理频率**；
当前同步客户端在 chunk 耗尽时等待新预测，实际运行还会受推理延迟影响。
新 slippers state 模型为 30 步（当前动作 + 后续 29 步），标称窗口 1 秒，首末采样点间隔 29/30 秒。
不提供 RTC、动作插值、倍速重采样或异步补帧。

任务和默认文本来自 real 数据转换脚本，GUI 需填写对应文本：

| task | prompt |
|---|---|
| `blocks` | Collect the blocks on the table by color: place the gray blocks in the left basket and the pink blocks in the right basket. |
| `tubes` | Insert the test tube into the test tube rack. |
| `slippers` | Place the slippers on the shoe rack. |

若训练时通过 instruction JSON 覆盖过文本，应使用实际训练文本。
服务尊重每个请求的非空 `prompt`，不会强制替换为默认文本；
`--prompt` 可更改 metadata 和离线测试的默认文本。

## 接口转换

请求为二进制 msgpack，字典内容如下：

```python
{"images": {"top": rgb_top, "left": rgb_left, "right": rgb_right},
 "state": np.zeros(14, dtype=np.float32), "prompt": "任务文本"}
```

三路图像为 HWC uint8 RGB，固定按 top / left / right 排列，
使用与训练 loader 相同的 PIL `resize((224, 224))`。state 必须为有限的 14 维向量，
旧图像模型不以它作为模型输入，握手给出 `uses_state=false`。
state 模型握手为 `uses_state=true`：服务端把 YAM 顺序重排成左关节 6、右关节 6、左夹爪、右夹爪，
用该 checkpoint 的 `dataset_statistics.json` 中 **state 自身的 q01/q99** 及训练用
`RealStateTransform` 归一化，常量维度归零、裁剪到 [-1, 1]。
再按训练 `_pack_sample` 的 float16 精度转换，交给 QwenEDL 同一个 `_build_instructions`，
生成 `[STATE] ... [ACTION]`（256 档）。客户端始终发原始 rad/夹爪值，不能提前归一化。
action 仍单独使用 action 统计量反归一化，再重排回 YAM 顺序。
不要手动添加 `Your task is ...` 模板，模型内部已处理。

`PolicyServerWrapper` 用当前 checkpoint 的统计量完成反归一化。
适配器只去除 batch 轴、重排动作列，不套用 π0.5 的统计量、不再归一化。

| 内容 | 顺序 |
|---|---|
| StarVLA 内部动作 | 左关节 6、右关节 6、左夹爪、右夹爪 |
| YAM state / 输出动作 | 左关节 6、左夹爪、右关节 6、右夹爪 |

映射索引为 `[0,1,2,3,4,5,12,6,7,8,9,10,11,13]`。
输出为顶层 `actions: float32[horizon,14]`，绝对关节目标、关节单位 rad、夹爪归一化编码。
服务不裁剪输出；YAM 原有执行限制仍由客户端负责。
额外返回 `policy_timing`、`server_timing` 和 `diagnostics`。
EDL diagnostics 保留 StarVLA 原有 batch/token 维度，不进行动作列重排。
错误以 OpenPI 兼容的文本帧返回并关闭连接；HTTP `/healthz` 返回 `OK`。

## 录制样本验证

```bash
cd /home/tiancai/liuzihao/starVLA_feature
./policy_server_real.zsh offline --task blocks --repeat 1 --output /tmp/starvla_blocks
# 服务已启动后，独立终端验证 WebSocket：
./policy_server_real.zsh client --port 8002 --repeat 1 --output /tmp/starvla_client
```

输出目录必须尚不存在。默认样本位于相邻 `starvla-yam-inference/samples`，
可通过 `--samples-root` 或 `--input` 覆盖。client 默认跟随服务的 task、horizon 和 prompt。
默认使用本任务的 real 转换文本；`--use-sample-prompt` 可保留 NPZ 内的 π0.5 文本。
避免把其他任务或 `put_pen` 样本用于单任务模型的效果评估。

保存动作 `.npy`、EDL diagnostics `.npz`、metadata/耗时/MAE `report.json`。
录制样本的示教可能有 32 步，只比较与模型输出重叠的前 horizon 步。
这些测试不运行机器人，单个录制样本的 MAE 不能代替真机成功率。

适配器回归测试：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate starvla
python -m unittest deployment.real.test_yam_policy -v
```

## 本机验证记录（2026-10-07）

5 项适配/协议回归测试通过。三个真实 checkpoint 均对对应录制样本产生有限的
`15×14` 动作；现有 `yam-abc-reproduce` 客户端成功读取 16 个动作，
恰好请求两个 chunk，逐行比对确认第 15 步后正确切换。健康检查和新 client 入口也通过。
测试服务已停止；未启动 GUI、初始化机械臂或发送控制指令。

| 样本 | 单次耗时 | 前 15 步关节 MAE (rad) | 夹爪 MAE |
|---|---:|---:|---:|
| blocks（服务预热后，含 WebSocket） | 0.625 s | 0.0204 | 0.0056 |
| tubes（首次模型调用） | 1.190 s | 0.4095 | 0.4762 |
| slippers（首次模型调用） | 0.978 s | 0.0101 | 0.0038 |

每个任务仅测试一份样本，不是稳定性能基准。tubes 样本误差较大，接口测试通过
不代表该任务动作质量已经验证；需要进一步核对对应示教和模型效果。
即使 blocks 此次预热后耗时仍超过 0.5 秒，不应据此声称可连续无等待地按 30 Hz 执行。
详细结果保存在复制目录 `validation/starvla_20261007/`，与原 π0.5 验证记录分开。
