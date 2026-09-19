# 模仿学习中的滚动预测 / Receding-Horizon 执行（一手资料归档）

> 来源归档（ingest）

- **标题：** Receding-Horizon Policy Execution in Imitation Learning — Primary References
- **类型：** paper（多篇一手论文索引）
- **入库日期：** 2026-09-19
- **一句话说明：** 梳理 IL 策略「预测一段、执行前缀、再重规划」的滚动执行范式；以 Diffusion Policy 与 ACT 为 canonical 对照，区分经典 receding horizon 与 ACT 式 temporal ensemble。

## 核心论文摘录

### 1) Diffusion Policy: Visuomotor Policy Learning via Action Diffusion（Chi et al., RSS 2023 / IJRR 2024）

- **链接：** <https://arxiv.org/abs/2303.04137>
- **项目页：** <https://diffusion-policy.cs.columbia.edu/>
- **代码：** <https://github.com/real-stanford/diffusion_policy>（**已开源**）
- **核心贡献：** 策略输出未来 **action sequence**（chunk），部署时 **只执行其中一部分**，下一控制周期用 **新观测重新去噪预测** 下一段——论文将这一闭环明确表述为 **receding-horizon control** 设计之一。
- **关键参数：** 典型预测视界 \(T_p\) 16–32 步；执行视界 \(T_e\) 小于 \(T_p\)（常见只执行前 8 步或更少，依任务与推理延迟而定）。
- **对 wiki 的映射：**
  - [receding-horizon-policy-execution](../../wiki/concepts/receding-horizon-policy-execution.md)（新建概念页）
  - [paper-diffusion-policy](../../wiki/entities/paper-diffusion-policy.md)
  - [diffusion-policy](../../wiki/methods/diffusion-policy.md)

### 2) ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware（Zhao et al., RSS 2023）

- **链接：** <https://arxiv.org/abs/2304.13705>
- **项目页：** <https://tonyzhaozh.github.io/aloha/>
- **代码：** <https://github.com/tonyzhaozh/act>（**已开源**）
- **核心贡献：** **Action Chunking with Transformers (ACT)**——CVAE + Transformer 一次生成 **K 步动作块**，而非逐步独立预测。
- **执行机制（易混淆点）：** 原论文默认 **Temporal Ensembling (TE)**：每个控制时刻都发起 **重叠 chunk 预测**，对同一未来时刻的多条预测做 **指数加权平均** 再执行。这与 Diffusion Policy 那种「播完执行前缀 → 丢弃剩余 → 整段重规划」的 **经典 receding horizon** **不等价**。
- **对 wiki 的映射：**
  - [receding-horizon-policy-execution](../../wiki/concepts/receding-horizon-policy-execution.md)
  - [paper-act](../../wiki/entities/paper-act.md)
  - [action-chunking](../../wiki/methods/action-chunking.md)

## 概念对照（面向 wiki 编译）

| 维度 | Diffusion Policy 式 receding horizon | ACT 式 temporal ensemble |
|------|--------------------------------------|---------------------------|
| 预测频率 | 每 \(T_e\) 步重规划一次 | **每步** 都预测新 chunk |
| 重叠区处理 | 旧 chunk 未执行部分 **丢弃** | 重叠区 **加权融合** |
| 闭环反馈 | 重规划时观测已更新 | 每步观测更新 + 多预测融合 |
| 典型动机 | 降推理频率 + 短开环前缀 | 降推理频率 + **平滑抖动** |

> **与 MPC receding horizon 的关系：** 控制论 MPC 每步解 OCP 只执行 **首控制量**；IL 滚动执行共享「有限视界 + 前缀执行 + 重规划」骨架，但 **无显式动力学约束优化**，而是学习式策略生成动作序列。见 [robot-control-paradigm-receding-horizon-ilc](../../wiki/overview/robot-control-paradigm-receding-horizon-ilc.md) 与 [model-predictive-control](../../wiki/methods/model-predictive-control.md)。

## 当前提炼状态

- [x] Diffusion Policy / ACT 项目页与仓库已交叉核查（2026-09-19）
- [x] wiki 概念页 `wiki/concepts/receding-horizon-policy-execution.md` 新建
- [x] 与 [why_action_chunking_improves_bc_corl2026.md](./why_action_chunking_improves_bc_corl2026.md) 机制叙事对齐（训练目标 vs 执行协议可分离）
