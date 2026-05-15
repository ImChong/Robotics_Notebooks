# Xiaomi-Robotics-0

> 来源归档（ingest）

- **标题：** Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution
- **类型：** repo + paper（arXiv）+ 官方项目页
- **组织：** Xiaomi Robotics（论文与项目页一致）
- **代码：** https://github.com/XiaomiRobotics/Xiaomi-Robotics-0
- **项目页（论文摘要外链）：** https://xiaomi-robotics-0.github.io/
- **品牌站说明页：** https://robotics.xiaomi.com/xiaomi-robotics-0.html
- **论文：** https://arxiv.org/abs/2602.12684
- **入库日期：** 2026-05-15
- **一句话说明：** 约 **4.7B** 参数的 **VLA**：**Qwen3-VL-4B-Instruct** 作多模态骨干，**DiT** 以 **flow matching** 生成动作 chunk；大规模跨本体机器人数据与 **~80M** VL 样本共训以防灾难性遗忘；后训练针对 **异步 chunk 执行**（推理与执行重叠）引入 **动作前缀 + Λ 形注意力掩码、前缀随机遮蔽、在线误差自适应损失重加权** 等，报告在 LIBERO / SimplerEnv / CALVIN 仿真与双臂真机（Lego 拆解、毛巾折叠等）上的结果与 **RTX 4090 上 ~80ms** 级延迟叙事，并开源权重与推理代码。
- **沉淀到 wiki：** [Xiaomi-Robotics-0](../../wiki/entities/xiaomi-robotics-0.md)

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
