# 真机单次执行与 EDL 采集

单次执行由 **YAM 客户端**控制。模型 server 常驻，收到一次请求就返回一个 15 步 chunk；
GUI/终端每次人工触发最多执行这 15 步，然后等待，不会自动重推理。
记录也保存在客户端，因为客户端知道输入、预测、实际发送情况及人工 episode 标签。
server 负责计算 EDL 诊断，可选返回 top-k Dirichlet 参数，不判断真实任务成败。

## 启动与操作

终端 A，启动更新后的模型服务（已有旧版服务需由操作者停止后重启）：

```bash
cd /home/tiancai/liuzihao/starVLA_feature
./policy_server_real.zsh serve --task blocks
# tubes / slippers 为另两个任务。
```

终端 B，启动手动 GUI；不要与其他控制同一机械臂的 GUI/遥操会话并行：

```bash
cd /home/tiancai/liuzihao/starVLA_feature
./policy_gui_real.sh
```

浏览器打开 `http://127.0.0.1:8042`。复制目录 `starvla-yam-inference/run_gui.sh`
也指向此入口，原 GUI 启动器另存为 `run_gui_legacy.sh`。
已有 GUI 进程不会热更新，需操作者结束已有任务后安排切换。本实现不改原 YAM 源码。

1. 检查模型连接，可开启相机预览。检查连接和启动网页本身不初始化机械臂。
2. 设定 collection ID、seed namespace、任务文本、关节限速（默认 0.3 rad/s）。
3. 点击 **执行一个 horizon**。第一次会初始化从臂/相机，可能包含驱动的夹爪标定；
   记录首次推理前的真实关节姿态，然后请求一次推理、执行 15 步、等待。
   不调用旧启动流程的额外首动作推理/自动 ramp。30 Hz 是动作步频，不保证换段无等待。
4. 再次点击只执行下一个 horizon，多次触发属于同一条 episode。起始姿态和任务设置保持不变。
5. 点击 **Stop episode · 返回本轮起点**：立即取消尚未发送的预测动作，等当前请求退出，
   再以不超过 0.3 rad/s 的关节目标速度返回本轮第一次推理前的姿态（包含夹爪）。
   回位不恢复桌面物体布局，也不属于训练用策略动作。
6. 回位结束后弹框输入 `y`（成功）、`n`（失败）或 `drop`（删除本轮暂存记录）。
   取消弹框会保留为待标注，可点击“标记结果”重试，不会自动标为失败。
7. 标记或丢弃后，下一次触发开始新 episode，并重新记录其起始姿态。

页面“本轮返回姿态”是固定的 episode 起点，不随 chunk 改变。
“最新 chunk 的最后一个 action”显示最新预测第 15 步的 14 维原始目标，
同时显示 chunk 编号、状态和已下发步数（例如 15/15）。页面还显示该 chunk 的模型原始动作序列中，
相邻 action 差分乘 30 Hz 得到的全 14 维最大速度及其维度；同时单独显示“实测起点到第一个 action”也纳入的本 chunk 最大目标变化速度。
这些值都在限速、夹爪限幅之前计算；前者反映模型 action 序列内部变化，后者能暴露绝对 joint 目标与当前姿态之间的首步跳变。
新 chunk 等待预测时会清空旧目标。
该目标尚未经过客户端限速/夹爪限幅，完成下发也不代表机械臂已经到达该姿态。
结束 episode 后保留该 chunk 信息供标注参考，标记或 drop 后清空。

回位会检查反馈：关节误差不超过 0.05 rad、夹爪误差不超过 0.1；发送完目标后最多等待 3 秒。
若未到达、通信异常或发生急停，会显示“回位未完成”，保留数据供选择标签/drop，
不会将“发送了 home 目标”当作实际到达。急停后不自动重新使能或回位。
E-STOP 与暂停不同：它调用 YAM 原有停止/保持逻辑，恢复需要操作者 Reset Session。

可选终端 C，用字母触发 **同一个 GUI episode**：

```bash
cd /home/tiancai/liuzihao/starVLA_feature
./policy_client_real_manual.sh --collection-id real_train_v1 --seed-namespace real_train_v1
```

- `n` + 回车：一次推理、一个 horizon。执行过程中提前输入的字符会清空，不排队执行。
- `e` + 回车：结束 episode，返回本轮起点；也可直接在 GUI 点击 Stop episode。
- 状态为 `await_label` 时，输入 `y` / `n` / `drop`；此时 `n` 表示失败，不是继续执行。
- `q`：仅退出键盘控制，GUI/模型服务与当前 episode 继续保留。
- Ctrl+C：向 GUI 发送 E-STOP，不自动回位。

GUI 和终端共用同一执行锁；忙碌时第二次触发会被拒绝，不会排队执行。

## 保存内容与标签

默认客户端目录：`examples/LIBERO/edl_pred_real/records/<时间_UUID>/`。
可用 GUI 启动参数 `--output /path/to/records` 修改。

```text
episode.json                    # 模型/任务/采集身份、返回姿态、整条轨迹 success（未标注为 null）
chunk_000000/
  request.json                  # 指令、chunk 序号、UTC/单调时钟
  observation.npz               # 三路原始 RGB 图像、14 维 state
  prediction.npz                # YAM 顺序的原始模型动作、EDL 诊断、224×224 模型输入图像
  events.jsonl                  # 每个机械臂的命令调用与测量时序
  result.json                   # completed/cancelled/error、完成发送的行数
chunk_000001/...
return_home.json                 # 回位目标、反馈和结果，独立于策略轨迹
end_episode.json                # 结束/回位结果
```

每次先把请求和预测落盘，成功后才允许发送动作。磁盘写入失败会停止继续执行。
`command_attempt` 表示即将调用驱动；`command_returned` 表示调用已正常返回，
其中 `effective_target` 是驱动报告的目标，不是机械臂实际到达的位置。
`measured_before/after` 保存该步读到的真实状态；两只臂分别记录，能区分中途一侧失败。
原始预测动作、限速/夹爪处理后的发送目标和反馈不会混为同一个数组。

临时数据逐 chunk 保存以防进程中断；只有显式 y/n 才成为带标签的完整 episode。
`drop` 删除当前 episode 的整个目录，包含图像、预测、时序和标签，不影响其他 episode。
进程意外退出留下的 open/null 记录不会自动作为失败样本训练。

## Evidence、AU/EU 的含义

`action_token_evidence` 是被选中 action token 的 evidence。
为了能复核 AU/EU，手动客户端设置 `return_edl_details=true`，额外保存：

- `action_token_topk_alpha[1,N,K]`：每个 action token 的动作词表 top-k Dirichlet alpha；evidence 为 alpha−1。
- `action_token_topk_ids[1,N,K]`、`action_token_ids[1,N]`、mask、token 数量。
- 原模型计算的 action-token AU/EU、confidence、rank 等。

这不会更改模型参数、生成策略或训练行为。未请求详情的原 LIBERO/普通推理调用保持原输出行为。
AU 是 top-k Dirichlet 期望分类熵（K>1 时除以 log(K)）；EU 是 K/sum(alpha)。
离线导出会从 alpha 重新计算并逐 token 比对模型 AU/EU。
这里只保留动作词表 top-k 分布，不是完整词表的全部 logits。

## 导出与后续训练

`starvla` 环境已安装现有 edl_pred 所需的 h5py；YAM 环境采集阶段无需 h5py。

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate starvla
cd /home/tiancai/liuzihao/starVLA_feature
python -m examples.LIBERO.edl_pred_real.export \
  --records examples/LIBERO/edl_pred_real/records \
  --task blocks --collection-id real_train_v1 --seed-namespace real_train_v1 \
  --output examples/LIBERO/edl_pred_real/datasets/real_blocks.hdf5
```

输出不覆盖已有文件。默认排除 mock、未标注、未关闭及完全未执行的 chunk；
error chunk 会拒绝导出，需人工检查原始记录；部分执行后主动结束的 chunk 保留实际完成行数。
每个文件仅接受一个 task、采集身份和 checkpoint。HDF5 保持 `schema_version=1.0`，
提供原分类器需要的 `success`、`num_action_tokens`、`token_offsets`、AU/EU 等；
三路图像、动作、top-k 参数、时序记录另存于 `real_chunks`，不作为分类器输入。

复用原分类器实现即可：

```bash
python -m examples.LIBERO.edl_pred.train --config examples/LIBERO/edl_pred_real/configs/blocks.yaml
```

配置按 episode 划分数据，不能把同一 episode 的 chunk 拆到训练和验证中。
现有划分器要求每个任务至少 2 条成功、2 条失败；实际校准应使用更充分、独立的数据。
手动暂停后如果人工调整物体或机械臂，应结束当前 episode，避免把人为干预混进连续轨迹。

当前只完成采集、导出和分类器读取兼容验证，没有训练真实分类器或计算有效阈值。
原 `edl_pred/calibrate_rejection.py` 面向完整 sweep，并固定使用每条 episode 的第 1～10 个 chunk；
短真机 episode 不应通过复制/补齐 chunk 来满足这个假设。后续应按真实数据长度选择校准方案，
将训练/校准与独立测试采集分开。分类器自身的 AU/EU 与 VLA action-token AU/EU 也需区分。

## 验证

```bash
python -m unittest examples.LIBERO.edl_pred_real.test_recording examples.LIBERO.edl_pred_real.test_export deployment.real.test_edl_details -v
PYTHONPATH="$PWD:/home/tiancai/yam-abc-reproduce" /home/tiancai/yam-abc-reproduce/.venv/bin/python -m unittest deployment.real.test_gui -v
```

所有生命周期测试使用 mock 机械臂/相机。真实模型验证只使用录制样本；不进行机器人执行。

2026-10-07 验证：单次控制/写入失败/取消/drop/HDF5 读取/alpha 复算等 7 项测试，
GUI 回位、标注及急停 4 项 mock 测试均通过；原 real 适配器 5 项测试通过。
GUI JavaScript 也验证了取消弹框保持未标注、drop 明确提交以及回位期间禁止新触发。
真实 blocks checkpoint 通过录制图像跑通请求、详细 alpha、15 步模拟执行和 HDF5 导出；
与不开启详细 alpha 的动作逐元素完全相同。测试模型服务已停止，没有启动真实 GUI/机械臂。
实测文件位于复制目录 `validation/starvla_manual_20261007/`，均标记为 mock execution，
默认不会作为真机训练样本导出。
