# MiMo-V2.6 技术报告（Scaling RL Toward Self-Improvement）

> 来源归档

- **标题：** MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement
- **类型：** paper（官方技术报告 PDF，非 arXiv）
- **机构：** LLM-Core Xiaomi（Xiaomi MiMo Team）
- **PDF：** <https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL/blob/main/MiMo_V2_6_technical_report.pdf>
- **模型卡 / 评测表：** <https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL>
- **HF 集合：** <https://huggingface.co/collections/XiaomiMiMo/mimo-v26>
- **入库日期：** 2026-09-23
- **一句话说明：** MiMo-V2.6 原生全模态 MoE 系列通过 **三维 RL 算力扩展**（更大 batch / 更多任务环境 / 更大 Grader 算力）推进 **RSI（递归自我改进）**；Pro **1.02T 总参 / 42B 激活**、Flash **310B 级**；混合 **Code / General / Visual / Cyber** 任务与 **Multi-Harness** 训练；开源 **7k+ RL 环境**、**verl + uni-agent** 框架与 **MiMo-V2.6-Distill-Qwen-9B** 复现基线。

## 开源核查（步骤 2.5，2026-09-23）

| 资源 | 状态 | 链接 |
|------|------|------|
| **Pro / Flash RL 权重** | **已开源** | [MiMo-V2.6-Pro-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Pro-RL) · [Flash-RL](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-RL)（MIT） |
| **Distill 复现基线** | **已开源** | [MiMo-V2.6-Distill-Qwen-9B](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B) |
| **技术报告 PDF** | **已公开** | 随 Pro-RL 仓发布 |
| **RL 训练框架** | **已开源** | [XiaomiMiMo/verl](https://github.com/XiaomiMiMo/verl) · [uni-agent](https://github.com/XiaomiMiMo/uni-agent) |
| **RL 任务环境（~7k）** | **已开源** | 报告 §7 / 发布说明「7k+ 高质量 RL 任务环境」 |
| **mini-harnesses** | **已开源** | 发布说明「轻量可组合 Harness」 |

## 架构摘要（报告 §2）

| 组件 | MiMo-V2.6-Pro-RL |
|------|------------------|
| **骨干** | 70 层 hybrid **SWA + GA**；6144 hidden；**384 routed experts / 8 activated** |
| **规模** | **1.02T 总参 / 42B 激活**（稀疏 MoE） |
| **上下文** | **1M tokens** |
| **模态** | Text / Image / Video / Audio（原生全模态） |
| **视觉** | **681M** MiMo ViT（28 层：24 SWA + 4 Full） |
| **音频** | **308M** AudioTokenizer + **127M** patch encoder |
| **投机解码** | 5 层 SWA **MTP** drafter（DFlash 风格，每步预测 7 token） |

Flash 变体参数量约 **310B**（HF 元数据）。

## RL 扩展三维（报告 §4 摘要）

1. **训练算力：** 全异步 **GRPO**；每步 **1,568 prompts × G=16** rollout → **2.7–3.7B training tokens**（上下文至 1M）；Pro / Flash RL 后训练累计成本约 **$2.6M / $0.9M**（报告 Figure 3）。
2. **环境与 Harness：** Code / General / Visual / Cyber 四域 **~7k** 可验证任务；**Multi-Harness Training** 用解耦 **mini-harnesses**（system prompt / tools / context 可重组）提升未见 harness 泛化。
3. **Grader 算力：** **Groupwise Reward Synthesis (GRS)** 离线 rubric + **Groupwise Advantage Redistribution (GAR)** 在线组内排序；配合 reward hacking 防线（冻结 MoE router、对抗筛查、轨迹审计）。

## 训练动态（报告 Figure 1 / §4.1）

- **DeepSWE v1.1 avg@3：** Pro **58.4 → 72.6**；Flash **48.7 → 65.7**（随 RL 成本单调上升）。
- 发布说明补充：Live RL **~6 天 / 各 30 步 / ~75 万轨迹**；任务平均通过率 Pro **+12%**、Flash **+25%**（与报告趋势一致，口径略异）。

## Distill + 开源 RL 实验（报告 §7 / Table 6 摘录）

以 **MiMo-V2.6-Distill-Qwen-9B** SFT 为共同起点，分域 GRPO：

| Benchmark | Qwen3.5-9B | Distill SFT | Distill + RL |
|-----------|------------|-------------|--------------|
| SWE-bench Verified | 60.0 | 61.1 | **66.2** |
| Terminal Bench 2.1 | — | 37.1 | **52.8** |
| MiMo Cyber Bench (mini) | 5.7 | 31.3 | **47.0** |
| MiMo Visual Coding (mini) | — | 64.0 | **72.4** |

## 对 wiki 的映射

- 主实体：[MiMo-V2.6](../../wiki/entities/mimo-v2-6.md)
- 发布说明：[mimo_v2_6_release_2026-09-22.md](../blogs/mimo_v2_6_release_2026-09-22.md)
- 仓库归档：[mimo-v2-6.md](../repos/mimo-v2-6.md)
- 具身评测交叉：[MiMo-Embodied](https://github.com/XiaomiMiMo/MiMo-Embodied)（独立 VLM 评测仓，非 V2.6 本体）
- 小米机器人 VLA 谱系：[Xiaomi-Robotics-1](../../wiki/entities/xiaomi-robotics-1.md)（**不同团队/产品线**）
