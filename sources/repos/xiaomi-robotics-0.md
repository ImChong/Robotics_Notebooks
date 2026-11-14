# Xiaomi-Robotics-0

> 来源归档（ingest）

- **标题：** Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution
- **类型：** repo + paper（arXiv）+ 官方项目页
- **组织：** Xiaomi Robotics（论文与项目页一致）
- **代码：** https://github.com/XiaomiRobotics/Xiaomi-Robotics-0
- **项目页（论文摘要外链）：** https://xiaomi-robotics-0.github.io/
- **品牌站说明页：** https://robotics.xiaomi.com/xiaomi-robotics-0.html
- **论文：** <https://arxiv.org/abs/2602.12684>（归档见 [xiaomi_robotics_0_arxiv_2602_12684.md](../papers/xiaomi_robotics_0_arxiv_2602_12684.md)）
- **入库日期：** 2026-05-15
- **通读：** 已完成
- **核对范围：** `README.md`、`_tree.txt`（133 文件）、`deploy/{server,client}.py`、`scripts/{deploy,launch_libero}.sh`、`eval_libero/` / `eval_calvin/` / `eval_simplerenv/`（含 `main.py` 与 `eval_logs/` 实测结果）、`xr0/{README.md,configs/model/XR0.yaml,scripts/{train,deploy}.sh,tools/{train,weight_convert}.py,mibot/models/VLA/XR0.py,mibot/models/runner/base_runner.py,mibot/server/{deploy,runtime/server,runtime/client}.py}`
- **一句话说明：** 约 **4.7B** 参数的 **VLA**：**Qwen3-VL-4B-Instruct** 作多模态骨干，**16 层 DiT** 以 **flow matching** 生成动作 chunk；大规模跨本体机器人数据（~200M 步）与 **>80M** VL 样本共训以防灾难性遗忘；后训练针对 **异步 chunk 执行**（推理与执行重叠）引入 **动作前缀 + 局部因果窗口（local_window=4）/ Λ 形掩码、前缀随机遮蔽（keep_last_k=2, prefix_mask_prob=0.5）、RoPE +10 偏移、在线 L1 误差重加权与可选 FFT 频域损失**，开源含 Pretrain/LIBERO/CALVIN/SimplerEnv HuggingFace 权重、评估服务端/客户端及完整 `xr0/` 后训练代码。
- **沉淀到 wiki：** [Xiaomi-Robotics-0](../../wiki/entities/xiaomi-robotics-0.md)

---

## 仓库核查与实测日志对照（2026-11-14）

- **开源完整度：已完整开源**（模型权重托管于 HuggingFace `XiaomiRobotics/Xiaomi-Robotics-0-{Pretrain,LIBERO,CALVIN,SimplerEnv}`；推理与仿真评测入口在 `deploy/` 与 `eval_*/`；2026-04-27 起 `xr0/` 目录补齐后训练框架、数据转换与真机部署服务器）。
- **仿真评测日志一致：** 仓库 `eval_libero/eval_logs/*/merged_results.json`（Spatial 98.8% / Object 100.0% / Goal 98.8% / Long 97.2%）、`eval_calvin/eval_logs/{calvin_abc,calvin_abcd}/results.json`（1–5 步连续成功率 `1.000..0.881` 与 `0.997..0.918`）与 `eval_simplerenv/eval_logs/{bridge,fractal}/all_results.csv` 均与论文 Table 1/2/4/5 一致。
- **`xr0/` 源码核查要点：**
  - 训练入口：`xr0/scripts/train.sh` 启动 `torchrun tools/train.py`（Hydra `configs/config.yaml` + DeepSpeed），构建 `BaseRunner(LightningModule)`（先调用 `tools/weight_convert.py` 将 HF 权重转为带 `model.` 前缀的 `.pt`）。
  - 模型核心（`xr0/mibot/models/VLA/XR0.py`）：冻结 `Qwen3VLForConditionalGeneration` 的 input embeddings 并开启视觉梯度检查点；16 层 `DiT(hidden_size=1024, kv_heads=8, layer_num=16)` 挂载 VLM 后 16 层 KV Cache；默认 `state_shape=(1, 32)`、`action_shape=(30, 32)`、`num_steps=5`、`local_window=4`、`flow_sampling="beta"`（`Beta(1.5, 1.0)`）；`XR0.yaml` 默认 `enable_freq: true`、`prefix_mask_prob: 0.5`，但注意该样例配置 `async_train: false`（需显式覆盖开启异步前缀训练）。
  - 真机服务：`xr0/scripts/deploy.sh` 拉起 `python3 mibot/server/deploy.py --model <dir> --port <port>`，加载 `config.py` 与 DeepSpeed `mp_rank_00_model_states.pt`，经 length-prefixed `pickle` TCP socket 调用 `model.generate(batch)` 并反归一化 `(30, 32)` 动作块返回 `runtime/client.py`。

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [VLA](../../wiki/methods/vla.md) | 典型「VLM 条件 + 扩散式动作头」工业开源实例；强调 **实时异步 rollout** 而不仅是仿真分数 |
| [Action Chunking](../../wiki/methods/action-chunking.md) | 部署侧 **chunk 交叠执行**、前缀条件与 **时序捷径（抄前缀）** 问题及对策 |
| [Diffusion Policy](../../wiki/methods/diffusion-policy.md) | 动作生成侧采用 **DiT + flow matching**（连续控制语境） |
| [Teleoperation](../../wiki/tasks/teleoperation.md) | 自述 **338h / 400h** 级房内遥操作数据用于 Lego 拆解与毛巾折叠等任务 |
| [Manipulation](../../wiki/tasks/manipulation.md) | 仿真基准与 **双手灵巧操作** 真机评测叙事 |

---

## 设计要点（官网 / 论文归纳）

1. **数据：** 约 **200M** 机器人时间步（开源数据集如 DROID、MolmoAct 等 + 房内遥操作）与 **>80M** VL 样本（通用 VL + 机器人衍生 VL；含 grounding / VQA / caption / embodied reasoning 等子任务策展）。
2. **预训练阶段 1：** 在 **Choice Policies** 范式下让 VLM 同时预测 **N 个候选 action chunk 及分数**，winner-take-all 用 **L1** 选优更新；VL 与机器人轨迹按 **1:6** 混合以保 VLM 能力。
3. **预训练阶段 2：** **冻结 VLM**，仅训 **16 层 DiT**；观测与语言走 VLM 得到 **KV cache**（论文取 VLM **最后 16 层** KV 条件 DiT），本体状态与 noisy actions 经 MLP 编码；**flow matching** 损失，**Beta** 采样噪声时间步；DiT 内 **因果注意力** 建模动作时序。
4. **后训练（异步）：** 在 DiT 输入侧 **前缀已提交的干净动作**（训练 RTC 类思路）；为减轻「后期 token 抄前缀、弱化视觉响应」，采用 **RoPE 索引偏移**、**Λ 形注意力掩码**（紧随前缀的 noisy token 可看前缀以平滑衔接，更远 token 不能看前缀）、**前缀随机遮蔽**（保留末尾若干可见以维持一致性）、以及对 **flow matching 损失按在线 L1 误差重加权** 等。
5. **部署：** **异步**时先执行每 chunk 前 **Te** 步再触发下次推理，执行余下动作的同时算下一 chunk；下一 chunk 条件化当前 chunk **[Te, Te+Δtc)** 的动作前缀，并保证 **Δtc ≥ Δtinf**；多传感器对齐到 **30Hz** 时间线；推理 **5** 步 flow 积分；报告 **4090** 上 **Δtinf≈80ms**。

---

## 对 wiki 的映射

- 新建 **`wiki/entities/xiaomi-robotics-0.md`**：模型—数据—训练—异步部署一条线的实体页（双 Mermaid：预训练 / 异步 rollout）。
- 更新 `wiki/methods/vla.md`、`wiki/methods/action-chunking.md`：补充交叉引用，避免孤岛页。

---

## 外部参考（便于复核）

- Cai et al., *Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution*, arXiv:2602.12684
- [XiaomiRobotics/Xiaomi-Robotics-0（GitHub）](https://github.com/XiaomiRobotics/Xiaomi-Robotics-0)
- [Robotics @ Xiaomi 说明页](https://robotics.xiaomi.com/xiaomi-robotics-0.html)
