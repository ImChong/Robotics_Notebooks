---
type: entity
tags: [company, embodied-foundation-model, dataset, scaling, generalist-ai, foundation-policy, cross-embodiment]
title: Generalist AI（机器人）
status: complete
summary: "Generalist AI（2024 年成立）是以超大规模真实交互数据预训练具身基础模型的商业团队；官网 10 篇博文覆盖 2025-06 Research Preview、乐高 one-shot 拼装评测、GEN-0 → GEN-1 → GEN-1.5、物理常识、千手与 2026-06 融资公告，引用应以官网博客为准且注意闭源边界。"
updated: 2026-10-09
related:
  - ./generalist-gen0.md
  - ./generalist-gen1.md
  - ./generalist-gen15-one-shot.md
  - ./generalist-gen1-thousand-hands.md
  - ./physical-commonsense-generalist.md
  - ./skild-s1.md
  - ../concepts/embodied-scaling-laws.md
  - ../concepts/foundation-policy.md
  - ../methods/octo-model.md
  - ../overview/hub-cross-embodiment.md
sources:
  - ../../sources/sites/generalistai-blog-index.md
  - ../../sources/blogs/generalist_research_preview.md
  - ../../sources/blogs/generalist_robots_build_now_too.md
  - ../../sources/blogs/generalist_accelerating_physical_ai.md
  - ../../sources/blogs/generalist_physical_commonsense_2026.md
  - ../../sources/blogs/generalist_gen15_one_shot.md
  - ../../sources/blogs/generalist_thousand_hands.md
  - ../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md
---

# Generalist AI（机器人方向）

## 一句话定义

**Generalist AI**：2024 年成立、聚焦具身智能与通用机器人策略的商业实体（Generalist AI, Inc.，团队位于湾区与波士顿）；对外叙事强调 **海量人类 / 机器人交互数据** 上的预训练、规模定律验证，以及 GEN 系列（GEN-0 → GEN-1 → **GEN-1.5**）向 **多末端接口** 与 **one-shot physical prompting** 的扩展。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GEN-0 / GEN-1 / GEN-1.5 | Generalist Embodied Model 代际 | 公司公开的具身基础模型系列 |
| EFM | Embodied Foundation Model | 具身基础模型；其产品叙事核心 |
| VLA | Vision-Language-Action | 视觉-语言-动作多模态策略方向（对照开源路线） |
| EE | End Effector | 末端执行器；「千手」博文的扩展轴 |
| DoF | Degrees of Freedom | 自由度；Research Preview 中 7-DoF / 6-DoF 机械臂迁移 |
| AGI | Artificial General Intelligence | 通用人工智能；公司使命表述为 physical AGI |

## 与其他「Generalist」用语区分

知识库中另有「通用策略（generalist policy）」泛指任意跨任务模型；本页仅指 **Generalist AI 公司**，避免与 [Octo](../methods/octo-model.md)、[VLA](../methods/vla.md)、[Open X-Embodiment](../concepts/open-x-embodiment.md) 等开源或开放数据路线混淆。

## 为什么重要

- **商业侧 embodied scaling 样本：** 与开源 OXE / Octo 对照，提供「超大规模 in-house 数据 + 闭源模型」的产业叙事锚点。
- **多末端轴补全跨具身图景：** 2026-07「千手」博文把跨具身细化为 **工具/末端接口多样性**，见 [GEN-1 千手](./generalist-gen1-thousand-hands.md)。
- **one-shot 适应叙事：** 2026-08 **GEN-1.5** 宣称 **physical prompting** 与 **1–10 步** 微调即可适应新短程任务，见 [GEN-1.5 一次示范学习](./generalist-gen15-one-shot.md)。闭源对照：[S1](./skild-s1.md) 走 **显式 ICL 预训练**，公开地平线更长。
- **资本与产业信号：** 2026-06 公告新融资 4 亿美元、累计超 5 亿美元（自报），是闭源具身基础模型路线的重要资本样本。
- **引用纪律：** 成功率、小时数、变体数为官方自报；**确认未开源** 代码与数据集，不可当作可复现方法论文。

## 公司概况

| 项 | 内容 | 来源 |
|----|------|------|
| 成立 | **2024 年**（具体月份未见一手披露） | TechCrunch 2026-08-25 |
| 联合创始人 | Pete Florence、Andy Zeng（首席科学家）、Andrew Barry（CTO）；前两位为前 Google DeepMind 研究员，Barry 来自 Boston Dynamics | TechCrunch / SiliconANGLE |
| 地点 | Bay Area（CA）与 Boston（MA） | 官网 About |
| 使命 | 「为物理世界构建通用智能并让所有人可用」；从 **灵巧性** 切入做具身基础模型 | 官网 About（自报） |
| 团队背景 | 自述来自 OpenAI、Boston Dynamics、Google DeepMind 等，参与过 PaLM-E、RT-2、Gemini Robotics、Atlas / Spot / Stretch 等 | 官网 About（自报） |
| 融资 | 2026-06 新融资 $400M、累计 >$500M（自报）；媒体称估值 $2B（Series B），2026-08 又报 8VC 领投近 $200M 延展、估值 $3B（知情人士） | 官方博文 + 媒体 |

## 官方博客时间线（2026-10-09 共 10 篇）

完整列表核查见 [官网 Blog 列表归档](../../sources/sites/generalistai-blog-index.md)。

| 日期 | 博文 | 要点 | 本库入口 |
|------|------|------|----------|
| 2025-06-17 | [Research Preview](https://generalistai.com/blog/research-preview) | 首次公开：端到端 100 Hz 灵巧操作 4 任务；跨 Flexiv / UR5 迁移 | 本页下文「Research Preview」 |
| 2025-09-24 | [The Robots Build Now, Too](https://generalistai.com/blog/the-robots-build-now-too) | 看一眼人搭的乐高结构后复制拼装 | 本页下文「one-shot 乐高拼装评测」 |
| 2025-11-04 | [GEN-0](https://generalistai.com/blog/gen-0) | 物理交互规模化预训练；主张机器人 scaling laws | [GEN-0](./generalist-gen0.md) |
| 2026-01-29 | [Physical Commonsense](https://generalistai.com/blog/physical-commonsense) | 「物理常识 = 机器人暗物质」观点文 | [Physical Commonsense](./physical-commonsense-generalist.md) |
| 2026-03-24 | [The Real Breakthrough Behind Our GTC Demo](https://generalistai.com/blog/the-real-breakthrough-behind-our-gtc-demo) | GTC 现场演示与 GEN-0 带来的快速泛化 | [GEN-0](./generalist-gen0.md)（合并） |
| 2026-04-02 | [GEN-1](https://generalistai.com/blog/gen-1) | 「mastery」阈值叙事；后训练少量机器人数据 | [GEN-1](./generalist-gen1.md) |
| 2026-04-07 | [Going Beyond World Models & VLAs](https://generalistai.com/blog/beyond-world-models) | 解释 GEN-1 为何从零训练、超越 VLA / 世界模型 | [GEN-1](./generalist-gen1.md)（合并） |
| 2026-06-04 | [Accelerating the Next Phase of Physical AI](https://generalistai.com/blog/accelerating-the-next-phase-of-physical-ai) | 融资公告 + 数据飞轮叙事 | 本页下文「2026-06 融资公告」 |
| 2026-07-23 | [Towards Machines with a Thousand Hands](https://generalistai.com/blog/towards-machines-with-a-thousand-hands) | ~9k 末端变体；任务中途换手 | [GEN-1 千手](./generalist-gen1-thousand-hands.md) |
| 2026-08-19 | [GEN-1.5](https://generalistai.com/blog/gen-1.5) | one-shot / few-shot physical prompting；组合示范；sim 提示真机 | [GEN-1.5](./generalist-gen15-one-shot.md) |

## 早期与公司博文摘要

### Research Preview（2025-06）

首篇公开博文，未给模型名称或定量指标（[归档](../../sources/blogs/generalist_research_preview.md)）：

- **形式（自报）：** 全部视频为全自主、实时控制；端到端网络把像素与其他传感映射为 **100 Hz** 动作。
- **四个任务：** 分拣小型紧固件、折纸盒并盘入自行车链锁后合盖（两侧盒舌毫米级对齐）、用刮取 / 纸盘漏斗把 M4 螺丝收回玻璃罐、拆 / 按色分拣 / 抛掷乐高。
- **跨具身（自报）：** 同一模型在 7-DoF Flexiv Rizon 4 与 6-DoF UR5 间迁移；紧固件任务 **未使用 UR5 数据**、在被评测环境内该任务数据为零。
- **边界：** 与后来 GEN-0 的关系博文未说明（推测为其前身，无官方确认）。

### one-shot 乐高拼装评测（2025-09）

博文 *The Robots Build Now, Too*（[归档](../../sources/blogs/generalist_robots_build_now_too.md)）：

- **任务：** 人搭一个小乐高结构，机器人 **只看成品** 即端到端（像素 → 100 Hz 动作）复制搭建；无任务专用工程、无额外指令。
- **能力拆分（作者自述）：** 视觉理解「搭什么」；亚毫米精度、再抓取与对齐瞬间按压；逐块选砖、定向、暂放、安装的序列推理。
- **边界（作者自述）：** 仅测过 4 色、3 块 2×4 砖的结构；作者估算组合空间 99,840 种（估计值，非实测覆盖）；无成功率。「首个端到端拼装乐高的机器人」为 as far as we know 的自述。
- **与 GEN-1.5 区分：** 此处 one-shot 指「看目标结构复制」；GEN-1.5 的 one-shot 指「以示范轨迹作上下文提示」。

### 2026-06 融资公告

博文 *Accelerating the Next Phase of Physical AI*（[归档](../../sources/blogs/generalist_accelerating_physical_ai.md)）：

- **金额（公司公告）：** 新融资 **$400M**，累计 **超过 5 亿美元**。
- **投资方（公司公告）：** Radical Ventures 领投；新进 8VC、Union Square Ventures、Hanabi Capital、Norwest；既有投资方 NVIDIA、Boldstart Ventures、Spark Capital、Bezos Expeditions、NFDG 大幅跟投；新天使 Bin Lin、Fei-Fei Li、Naval Ravikant。
- **技术叙事（自报）：** GEN-0 把机器人带入预训练时代；GEN-1 达到商业可用阈值（多样任务 99% 可靠性、最高约 3 倍于此前 SOTA 的速度、学习复杂新技能、涌现即兴）；GEN-1 两个月后开始形成「更好模型 → 更多有用工作 → 真实业务数据」飞轮。
- **未写入公告：** 估值与轮次名称；媒体（SiliconANGLE）报道估值 $2B，TechCrunch 称该轮为 Series B。

## 数据与就绪度

- **数据 / 重定向就绪度：** 对外强调海量人类可穿戴交互预训练 + 少量机器人后训练；具体数据形态与跨本体适配 **未公开**，不可直接用于重定向或复现实验。
- **开源：** 截至 2026-10-09，公司站与全部 10 篇博文 **未见** GitHub / Hugging Face 训练推理入口。

## 核心原理（对外可核对部分）

公司不公开架构配方；外部读者可核对的主张主要是：

1. **规模化真实交互预训练** 驱动通才物理策略（对照 [Embodied Scaling Laws](../concepts/embodied-scaling-laws.md)）。
2. **端到端高频控制**：自 Research Preview 起一贯表述为像素等传感 → **100 Hz** 动作。
3. **末端多样性** 作为接触物理数据轴（对照 [跨具身知识链](../overview/hub-cross-embodiment.md)）。
4. **与开源 foundation policy** 的选型分工：要复现选 OXE/Octo/π 系开源代码；本页仅作产业对照。

## 工程实践

| 场景 | 建议 |
|------|------|
| 写综述 / 时代叙事 | 可引 GEN 系列作为商业 scaling 样本，并链到 [三个时代 Query](../queries/robot-learning-three-eras-narrative.md) |
| 设计灵巧操作评测 | 借鉴 Research Preview 的多轴任务设计（精度 / 双手协同 / 高频 / 抗扰）与乐高「看结构复制」任务，用自有开源栈实现 |
| 做跨末端研究 | 借鉴「千手」的 **task-vector 诊断** 与 **mid-episode tool swap** 评测设计 |
| 产线选型 | **不要**假设可下载 GEN-1；评估闭源 API/集成需直接对接厂商 |

## 局限与风险

- 营销与技术边界模糊；定量结果缺第三方复现，早期两篇博文甚至没有成功率。
- 「物理 AGI / Cambrian explosion」为愿景修辞，工程上仍受数据、对齐与安全约束。
- 估值、轮次等资本信息多来自媒体，非公司原文。
- 勿把公司名与开源 generalist policy 文献混为一谈。

## 关联页面

- [GEN-0：物理交互规模化预训练](./generalist-gen0.md)
- [GEN-1：Mastery 与超越 VLA / 世界模型](./generalist-gen1.md)
- [GEN-1.5 一次示范学习（Physical Prompting）](./generalist-gen15-one-shot.md)
- [GEN-1 千手：跨末端执行器泛化](./generalist-gen1-thousand-hands.md)
- [Physical Commonsense（Generalist 产业观点）](./physical-commonsense-generalist.md)
- [Embodied Scaling Laws](../concepts/embodied-scaling-laws.md)
- [Foundation Policy](../concepts/foundation-policy.md)
- [跨具身迁移（知识链）](../overview/hub-cross-embodiment.md)
- [Manipulation](../tasks/manipulation.md)
- [Octo](../methods/octo-model.md)
- [S1 / Skild AI](./skild-s1.md) — 另一条闭源 ICL 通才线（显式视频预训练）
- [HOST](./paper-host-one-shot-human-video.md) — 开源单视频 one-shot 对照，不是本公司产品

## 参考来源

- [Generalist AI 官网 Blog 列表核查（2026-10-09，10 篇）](../../sources/sites/generalistai-blog-index.md)
- [Research Preview（来源归档）](../../sources/blogs/generalist_research_preview.md)
- [The Robots Build Now, Too（来源归档）](../../sources/blogs/generalist_robots_build_now_too.md)
- [Accelerating the Next Phase of Physical AI（来源归档）](../../sources/blogs/generalist_accelerating_physical_ai.md)
- [The Dark Matter of Robotics: Physical Commonsense（来源归档）](../../sources/blogs/generalist_physical_commonsense_2026.md)
- [GEN-1.5: Embodied Foundation Models are One-Shot Learners（来源归档）](../../sources/blogs/generalist_gen15_one_shot.md)
- [Towards Machines with a Thousand Hands（来源归档）](../../sources/blogs/generalist_thousand_hands.md)
- [ted_xiao_embodied_three_eras_primary_refs.md](../../sources/blogs/ted_xiao_embodied_three_eras_primary_refs.md)
- GEN-0：<https://generalistai.com/blog/gen-0>
- GEN-1：<https://generalistai.com/blog/gen-1>
- 官网 About：<https://generalistai.com/about>
- 成立年份与创始人：TechCrunch，*Robotics startup Generalist reaches $3B valuation, sources say*（2026-08-25）<https://techcrunch.com/2026/08/25/robotics-startup-generalist-reaches-3b-valuation-sources-say/>
- 估值 $2B：SiliconANGLE（2026-06-04）<https://siliconangle.com/?p=770140>

## 推荐继续阅读

- [Towards Machines with a Thousand Hands](https://generalistai.com/blog/towards-machines-with-a-thousand-hands) — 多末端扩展主文
- [GEN-1 官方博文](https://generalistai.com/blog/gen-1) — mastery 与数据引擎叙事
- [Research Preview](https://generalistai.com/blog/research-preview) — 公司首批灵巧操作演示
