# Physical Intelligence 官网技术文章索引

> 来源归档（ingest）；核对日期：2026-09-28。范围为官网 Blog 列出的机器人模型、算法、系统、实验与伙伴部署文章。伙伴篇单独成节，不与模型论文混在技术表里。

- **站点：** <https://www.pi.website/>
- **文章目录：** <https://www.pi.website/blog>；<https://www.pi.website/research>
- **一句话说明：** 从 π₀ 的多本体动作生成，沿动作离散化、泛化、推理时延、经验强化学习、记忆与可控提示，梳理 PI 公开的技术路线。
- **开源核查：** [Open Sourcing π₀](https://www.pi.website/blog/openpi) 明确提供 [openpi 代码和权重](https://github.com/Physical-Intelligence/openpi)；[FAST](https://www.pi.website/research/fast) 提供 [公开 tokenizer](https://huggingface.co/physical-intelligence/fast)。其他文章不能因同属 PI 就推断各自完整训练代码、权重和实验数据已开放；复现前逐篇核对文章资源区和仓库。

## 技术文章（官网日期；每行独立原文入口）

| 日期 | 文章 | 技术摘录与阅读价值 |
| --- | --- | --- |
| 2024-10-31 | [π₀: Our First Generalist Policy](https://www.pi.website/blog/pi0) | 跨机器人、跨任务数据训练 VLA，以连续动作生成控制不同本体；[π₀ 方法](../../wiki/methods/π0-policy.md)。 |
| 2025-01-16 | [FAST: Efficient Robot Action Tokenization](https://www.pi.website/research/fast) | 用 DCT、量化和 BPE 压缩连续动作 chunk 为离散 token；π₀-FAST 把动作预测变成自回归训练，但逐 token 推理有延迟代价。 |
| 2025-02-04 | [Open Sourcing π₀](https://www.pi.website/blog/openpi) | 发布 openpi 的基础权重、微调与推理示例，以及 π₀-FAST；接入新机器人仍需匹配观测、动作和归一化。[π₀ 方法页](../../wiki/methods/π0-policy.md)。 |
| 2025-02-26 | [Teaching Robots to Listen and Think Harder](https://www.pi.website/research/hirobot) | Hi Robot 将高层任务分解与低层动作执行结合，引入人类反馈处理复杂的多阶段指令。[论文实体](../../wiki/entities/paper-hi-robot.md)。 |
| 2025-04-22 | [π₀.₅: A VLA with Open-World Generalization](https://www.pi.website/blog/pi05) | 混合知识来源提升陌生家庭环境中的任务泛化；[论文实体](../../wiki/entities/paper-pi05-open-world-vla.md)。 |
| 2025-05-28 | [VLAs that Train Fast, Run Fast, and Generalize Better](https://www.pi.website/research/knowledge_insulation) | Knowledge Insulation 用 FAST token 训练骨干，动作专家梯度不回传。[论文实体](../../wiki/entities/paper-knowledge-insulation.md)。 |
| 2025-06-09 | [Real-Time Action Chunking with Large Models](https://www.pi.website/research/real_time_chunking) | RTC 在推理期补全下一段动作 chunk，针对大模型延迟造成的停顿。文章还链接 2025-12-08 的训练期后续论文。[论文实体](../../wiki/entities/paper-real-time-chunking.md)。 |
| 2025-11-17 | [π*₀.₆: A VLA that Learns from Experience](https://www.pi.website/blog/pistar06) | RECAP 用真实执行经验和优势条件专精通用策略，针对成功率与任务吞吐。[论文实体](../../wiki/entities/paper-pistar06-recap.md)。 |
| 2025-12-16 | [Emergence of Human to Robot Transfer in VLAs](https://www.pi.website/research/human_to_robot) | 预训练多样性足够后，人视频共微调才会出现迁移；不要理解成无需机器人数据。[论文实体](../../wiki/entities/paper-pi-human-to-robot.md)。 |
| 2025-12-22 | [Moravec's Paradox and the Robot Olympics](https://www.pi.website/blog/olympics) | 用涂花生酱、洗锅、插钥匙等精细任务检验 π₀.₆ 微调后的操作能力与难点。[笔记](../../wiki/entities/pi-robot-olympics.md)。 |
| 2026-03-03 | [VLAs with Long and Short-Term Memory](https://www.pi.website/research/memory) | Multi-Scale Embodied Memory (MEM) 同时利用短期和长期历史来执行跨多个步骤、超过十分钟的任务。 |
| 2026-03-19 | [Precise Manipulation with Efficient Online RL](https://www.pi.website/research/rlt) | RL Token (RLT) 从 VLA 提取便于在线强化学习的表示，针对精密操作与少量真实数据的效率。 |
| 2026-04-16 | [π₀.₇: A Steerable Model with Emergent Capabilities](https://www.pi.website/blog/pi07) | 多模态提示（子任务、元数据、控制模态、视觉子目标）对齐异构经验并在推理时控制行为；[论文归档](../papers/pi07.md)、[方法页](../../wiki/methods/pi07-policy.md)。 |

## 应用叙事（非新模型）

| 日期 | 文章 | 阅读边界 |
| --- | --- | --- |
| 2026-02-24 | [The Physical Intelligence Layer](https://www.pi.website/blog/partner) | Weave / Ultra 自述现场洗衣与打包。数字是伙伴统计，不与实验室论文表横比。[笔记](../../wiki/entities/pi-physical-intelligence-layer.md)。 |

## 技术脉络与边界

1. **动作表示：** π₀ 的连续动作流与 FAST 的离散 token 是两种动作头方案；训练速度、推理延迟需要分别比较。
2. **系统闭环：** π₀.₅ 的陌生环境泛化仍依赖稳定的执行；RTC 解决 chunk 切换处的延迟，MEM 处理跨步骤的历史信息。
3. **经验与可控性：** π*₀.₆ / RLT 强调真实反馈与在线优化，π₀.₇ 则用可组合的条件输入统一异构数据。人视频迁移依赖预训练多样性；Robot Olympics 与 PI Layer 是微调演示和伙伴现场，不是新的训练算法。文章中的实验场景和模型版本各异，不能把性能数字直接横向排序。

## 关联页面

- [VLA 方法总览](../../wiki/methods/vla.md)
- [π₀](../../wiki/methods/π0-policy.md)
- [π₀.₇](../../wiki/methods/pi07-policy.md)
