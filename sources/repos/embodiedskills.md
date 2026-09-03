# EmbodiedSkills（DCDmllm / ZJU）

> 来源归档

- **标题：** EmbodiedSkills — A Unified Framework for Orchestrating, Training, and Deploying VLA Agents
- **类型：** repo
- **来源：** 浙江大学等（Wei Wang、Wenqiao Zhang 等）
- **链接：** <https://github.com/DCDmllm/EmbodiedSkills>
- **论文：** [arXiv:2609.01281](https://arxiv.org/abs/2609.01281) — 归档见 [`sources/papers/embodiedskills_arxiv_2609_01281.md`](../papers/embodiedskills_arxiv_2609_01281.md)
- **入库日期：** 2026-09-03
- **一句话说明：** 开源 **AgentLoop** 运行时：Qwen3-VL 高层技能决策 + **guarded runtime**（preflight / verify / recover）+ 可替换 **OpenPI/π₀.₅** 低层 VLA；含 RoboTwin 2.0 / LIBERO / RMBench 适配器与轨迹 SFT 管线（Python 包名 `clawvla`，发行名 `embodiedskills`）。

---

## 开源状态（GitHub 核查，2026-09-03）

| 组件 | 状态 |
|------|------|
| AgentLoop 运行时 | **已发布**（`src/clawvla/`） |
| RoboTwin / LIBERO / RMBench 适配器 | **已发布**（`configs/runtime/`） |
| OpenPI π₀.₅ 集成 | **已发布**（`integrations/openpi/`） |
| Qwen3-VL Agent LoRA 训练脚本 | **已发布**（`scripts/train_qwen_agent_lora.sh`） |
| 预训练 checkpoint / 数据集 | **未随仓发布**（README 写明 checkpoints 在树外；需自备或按文档训练） |
| LICENSE | 仓库 **未声明** SPDX（`license: null`）；商用前需自行确认 |

**结论：** **已开源**（训练与部署代码完整）；权重与大规模生成数据需按 README 自行准备。

---

## 核心定位

把 **低层 VLA 策略** 包进 **可检查、可训练的 agent 层**：高层 VLM 提出 **typed skill call**，runtime 在物理执行前做 **phase / prerequisite / freshness** 校验，执行后用 **verification** 决定继续、重观测、重规划或恢复；全链路以统一 **trajectory schema** 记录，供组件级 SFT 与可选在线优化。

---

## 仓库结构（README 摘要）

| 路径 | 作用 |
|------|------|
| `src/clawvla/` | AgentLoop、runtime、skills、components、adapters |
| `configs/runtime/` | `robotwin.json` / `libero.json` / `rmbench.json` |
| `integrations/openpi/` | π₀.₅ 数据集 hook 与训练 recipe |
| `scripts/` | 专家子任务采集、合并、Qwen LoRA 训练封装 |
| CLI | `embodiedskills-run`、`embodiedskills-run-vllm` |

---

## 主结果（README 表）

| Benchmark | 设定 | Reference | EmbodiedSkills |
|-----------|------|-----------|----------------|
| RoboTwin 2.0 | 50 tasks × 100 ep | π₀.₅ **82.74** | **86.20** |
| LIBERO | Spatial/Object/Goal/Long | OpenPI **96.85** | **97.40** |
| RMBench M(n) | 4 memory tasks | X-VLA **7.3** | **12.5** |

**RoboTwin 消融（同 50 任务 / 5000 ep）：** 去中间 verification → **48.2%**；全任务指令替代语义子任务 → **34.4%**；每子任务限 1 action chunk → **19.5%**。

---

## 快速入口

```bash
embodiedskills-run \
  --config configs/runtime/robotwin.json \
  --instruction "Place the container on the plate." \
  --artifact-prefix robotwin_example \
  --max-steps 80 \
  --run
```

---

## 对 wiki 的映射

- 实体页：[EmbodiedSkills（论文）](../../wiki/entities/paper-embodiedskills.md)
- 方法页：[VLA](../../wiki/methods/vla.md) — agent 层与冻结/可替换低层策略
- 对照：[Harness VLA](../../wiki/entities/paper-harness-vla.md) — 另一路冻结 VLA + agentic 编排
