# Introducing RLE-Bench — Evaluating Coding Agents as Robot Learning Engineers

> 来源归档（blog / RLE-Bench 官方）

- **标题：** Introducing RLE-Bench — A Qualifying Exam for Coding Agents as Robot Learning Engineers
- **类型：** blog
- **作者：** RLE-Bench team
- **原始链接：** <https://rle-bench.github.io/blog/>
- **发表/更新：** 2026-09-09（页脚 Updated 9 September 2026）
- **入库日期：** 2026-09-16
- **抓取方式：** WebFetch 全文 + 项目页交叉核对
- **一句话说明：** 官方叙事：为何用 **机器人学习工程师（RLE）** 工作流评测 coding agent；四能力互补视角、九任务规格、Build–Act–Observe–Revise 环、初榜三分化结论与 T01/T04/T08 案例研究。

## 项目页与开源核查（步骤 2.5）

| 入口 | 结果 |
|------|------|
| 本博客 | 任务定义、开发/评测预算、演示视频引用、初榜叙事 |
| 项目页 | Leaderboard + 成本/上下文图表 |
| GitHub | **已开源** MIT 仓 + Harbor CLI |
| 论文 | **未挂 arXiv** |

## 核心摘录（归纳，非全文）

### 动机

- 多数 coding agent 评测停在 **终端/代码库**；物理世界带来 **部分可观测、不可撤销、动力学复杂** 的反馈。
- 核心问题：通用 coding agent 能否在 **物理接地环境** 中闭合 **观察 → 推理 → 决策 → 反思**？
- **RLE（Robot Learning Engineer）** 工作覆盖：直接控机、训策略、建感知、改硬件——评测应覆盖 **开发过程**，而非只看最终 policy。

### 四互补工作流（各 2–3 任务）

1. **Interactive Control** — 先会 **操作** 机器人（T01 厨房 agentic control；T02 为后续 agent **造 harness**；T03 须 **行动后推理**）。
2. **Policy Development** — 把单次解变成 **可复用能力**（T04 人形 tracking ONNX；T05 VLA **recipe** 六轨）。
3. **Perception & Estimation** — 下游决策取决于 **状态质量**（T06 位姿估计；T07 视觉+力 bin clearing）。
4. **Mechanical Design** — 硬件上限软件无法补（T08 移动底座；T09 GELLO 重力补偿共设计）。

### 开发 vs 评测

- 开发：公开仿真 + 预算内 `Build→Act→Observe→Revise`。
- 评测：Harbor sandbox；**hidden** 场景/seed/embodiment/扰动；T01 保留 agent 上下文，T02 移交 harness 包，T05 replay recipe 训练后评 VLA。
- 网络默认关闭；API allowlist。

### 初榜三分化（博客叙事，具体数以 leaderboard 为准）

| 维度 | 观察 |
|------|------|
| 视觉 grounding | GPT-6 Astra vs Claude Opus 5 感知工作流差距最大 |
| 策略学习 | 头部模型更接近（motion tracking + VLA recipe） |
| 物理设计 | 仍最难；可见目标达成 ≠ 通过稳定性/连通性检查 |

### 案例研究要点

- **T01 harness 消融**：除 Astra 外，多数模型 **L2/L3  richer interface** 成功率更高、成本更低；Astra 在 L1 最强但 **加 harness 反而 hurt**。
- **T04 hill climbing**：Astra ~4h PPO 迭代；345-seed 候选在 200-seed audit 失败两次后改 checkpoint 平均；最终 545 hidden seed 无 fall（开发分与 hidden test 分开报告）。
- **T08 设计**：Astra shelf/payload 满分但 stability 零分；Luna 设计 **断连 frame** 违连通约束。

## 对 wiki 的映射

- [RLE-Bench](../../wiki/entities/rle-bench.md) — 升格主实体
- [sources/sites/rle-bench-github-io.md](../sites/rle-bench-github-io.md) — 项目页归档
- [sources/repos/rle-bench.md](../repos/rle-bench.md) — 复现入口
