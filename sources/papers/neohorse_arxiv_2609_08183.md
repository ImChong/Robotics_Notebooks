# NeoHorse-1: Towards Recursive Self-Improvement via Agentic Post-Training with Routing Harness（arXiv:2609.08183）

> 来源归档（ingest）

- **标题：** NeoHorse-1: Towards Recursive Self-Improvement via Agentic Post-Training with Routing Harness
- **类型：** paper / llm-agent / post-training / routing / recursive-self-improvement / tool-use / coding
- **框架短名：** NeoHorse-1
- **arXiv abs：** <https://arxiv.org/abs/2609.08183>
- **arXiv HTML：** <https://arxiv.org/html/2609.08183v1>
- **PDF：** <https://arxiv.org/pdf/2609.08183>
- **机构 / 团队：** NeoHorse Team（[TokenRhythm Technologies](https://tokenrhythm.ai/)）
- **项目页：** <https://tokenrhythm.ai/>
- **代码：** <https://github.com/TokenRhythm/NeoHorse>（Apache-2.0；权重与推理示例；见 [`sources/repos/neohorse.md`](../repos/neohorse.md)）
- **模型：** [NeoHorse-1-4B / 9B HF 合集](https://huggingface.co/collections/TokenRhythm/neohorse-1)；[ModelScope 合集](https://www.modelscope.cn/collections/TokenRhythm/NeoHorse-1)
- **基座：** Qwen3.5-4B / Qwen3.5-9B
- **发表日期：** 2026-09-09（arXiv v1）
- **入库日期：** 2026-10-02
- **一句话说明：** 用 **routing harness** 把部署轨迹（能力需求预测、服务 tier、交互结果）转成 **user-turn 训练样本**，经结构/语义质检与 **routing 引导三阶段 SFT 课程** + **routing 引导 on-policy 蒸馏** 后训练 **NeoHorse-1**；**capability-guided allocation** 把评测短板写回下一轮数据配比，闭合 **evaluation–selection–update** 原型环；4B/9B 在十项 agent/代码/指令基准上 macro-average 分别从 **58.94→64.87**、**65.60→69.04**。

## 摘要级要点

- **RSI 机制主张：** 已部署的 **routing harness** 除任务输出外，还留下 **执行轨迹** 与 **能力边界可观测证据**；可把「系统能做什么/不能做什么」转成下一轮学习信号（相对静态 instruction–response 对）。
- **数据：** 主语料为 **10⁵–10⁶** 级 harness 轨迹；辅以公开 instruction/reasoning/tool/code/preference 数据扩覆盖。粒度：**trajectory → user-turn（训练单元）→ subscene（语义标注单元）**。
- **质检：** 精确/近重复去重、评测去污染、**结构验证**（工具调用因果闭包等）、**六维语义评测**（goal / instruction / tool / evidence / recovery / termination）、subscene 级 **Scene / Goal / Outcome** 标注。
- **Routing 信号：** 每 user-turn 记录 **预测能力需求**、策略调整后决策、实际服务 tier（C0–C3）；用于 **课程排序**（Section 4.2）与 **OPD 阶段进度**（Section 4.3），**不把「实际服务的 tier」当难度标签**（用户覆盖/可用性会污染）。
- **训练栈：** routing-guided **三阶段 SFT 课程** + 同进度下的 **routing-guided on-policy distillation**（teacher 在学生 rollout 前缀上供 token 级监督）。
- **闭环：** 分层评测 → **model-deficiency profile** → 下一轮 mixture 向短板区域倾斜；更新 checkpoint 回到 harness 继续产轨迹（原型 **harness-mediated RSI**）。
- **评测：** 10 项基准（QwenClawBench、WorkBuddy、PinchBench、VitaBench、BFCL v4、τ²-Bench、HumanEval、LiveCodeBench v6、IFEval、IFBench）；agent 类多用 **OpenSquilla** 等 harness；NeoHorse-1 与 Qwen3.5 基座同配 **thinking mode**（SGLang v0.5.17）。
- **主结果：** NeoHorse-1-4B **64.87** avg（基座 Qwen3.5-4B **58.94**）；NeoHorse-1-9B **69.04**（基座 **65.60**）；4B 后训练后在多项 agent/代码指标上 **追平或超过 Qwen3.5-9B 基座**。
- **数据消融（4B）：** 同课程配置下，**routing-harness 轨迹** 相对公开 **Toucan** 工具 agent 数据在五项可控基准上 **Avg +2.65 pp**（Table 3 口径）。
- **局限（文内定位）：** 报告为 **initial prototype**；完整多轮 RSI 需跨迭代扩展任务面与 sustained feedback loop；仓库 **未发布** 完整训练/数据飞轮代码。

## 核心摘录（面向 wiki 编译）

### 评测 macro-average（文 Table 1–2，本库索引）

| 模型 | Avg（十项 macro） | 相对同规模基座 Δ |
|------|-------------------|------------------|
| Qwen3.5-4B | 58.94 | — |
| NeoHorse-1-4B | 64.87 | +5.93 |
| Qwen3.5-9B | 65.60 | — |
| NeoHorse-1-9B | 69.04 | +3.44 |

### 开源边界（2026-10-02 核查 GitHub + HF）

| 项 | 状态 |
|----|------|
| NeoHorse-1-4B/9B 权重（BF16/GGUF/MLX） | **已发布** |
| NeoHorse-Jev-4B 决策模型 | **已发布**（`jev/`） |
| 推理示例 `examples/chat.py`、`tool_call.py` | **已发布** |
| 技术报告 PDF（仓内） | **已发布** |
| Routing harness 生产环境、训练数据管线、SFT/OPD 训练脚本 | **未在仓库提供** |

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-neohorse-1.md`](../../wiki/entities/paper-neohorse-1.md)
- 概念交叉：[`wiki/concepts/recursive-self-improvement.md`](../../wiki/concepts/recursive-self-improvement.md)（harness 数据飞轮 vs 四层 RSI 标准对照）
- 仓库归档：[`sources/repos/neohorse.md`](../repos/neohorse.md)
- 站点归档：[`sources/sites/tokenrhythm.md`](../sites/tokenrhythm.md)
