---
type: method
tags: [vla, foundation-policy, deepmind, flow-matching, manipulation]
status: complete
updated: 2026-09-05
related:
  - ../entities/paper-pi0.md
  - ../overview/vla-wm-reading-roadmap-14-papers-technology-map.md
  - ./vla.md
  - ./pi07-policy.md
  - ./diffusion-policy.md
  - ../formalizations/vla-tokenization.md
  - ../formalizations/cross-modal-attention.md
  - ../entities/paper-kai0.md
  - ../entities/paper-spd.md
  - ../entities/paper-dexholdem.md
sources:
  - ../../sources/repos/openpi.md
  - ../../sources/papers/diffusion_and_gen.md
  - ../../sources/papers/chi0_kai0_arxiv_2602_09021.md
summary: "π₀ (Pi-zero) 是由 Physical Intelligence 提出的一种通用的 Vision-Language-Action 模型，通过结合流匹配（Flow Matching）与大规模预训练，实现了对复杂机器人操作任务的高效建模。"
---

# π₀ (Pi-zero) 策略模型

**π₀ (Pi-zero)** 是具身智能大模型（VLA）领域的奠基性工作，由 Physical Intelligence 团队于 2024 年提出（其后继版本见 [π₀.7](./pi07-policy.md)）。它旨在打破“一个机器人一个模型”的限制，通过单一的大型神经网络同时掌控不同形态（如双臂机械臂、灵巧手、移动底座）的机器人执行多样化的复杂任务。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| VLM | Vision-Language Model | 视觉-语言多模态理解模型，VLA 的上游 |
| LLM | Large Language Model | 大语言模型，常作高层任务/语言接口 |

## 主要技术路线

π₀ 的设计融合了语言模型的大规模预训练优势与生成式动作建模的精确性：

1. **流匹配 (Flow Matching) 骨干**：
   有别于 RT-1 等采用的标量离散化路线，π₀ 在动作输出层使用了**流匹配（Flow Matching）**。这是一种基于 [概率流形式化](../formalizations/probability-flow.md) 的高效生成式建模方法，能够以更少的推理步数生成高质量、连续且多模态的动作分布。
2. **视觉语言对齐**：
   π₀ 借用了预训练多模态大模型（如 VLM）的权重，使其天然具备理解自然语言指令（如“把弄脏的毛巾放进篮子里”）并识别图像中复杂物体的能力。
3. **后训练 (Post-training) 范式**：
   类似于 LLM 的指令微调，π₀ 首先在海量的多源机器人数据集上进行基础预训练（Pre-training），随后通过特定任务的高质量演示数据进行对齐微调。

## 为什么是“π₀”？

“π”在强化学习中代表策略（Policy），而“0”则隐喻“从零开始的通用基础”。它的出现标志着机器人控制正经历从“特征工程”到“**Scaling Law (规模法则)**”的范式转移。

## 性能优势

- **多任务泛化**：能够处理从折衣服到整理餐具等完全不同的任务。
- **跨平台通用**：同一模型可以无缝适配不同厂商的机械臂，证明了动作表征的普适性。
- **鲁棒性**：面对视觉背景的变化和物体的轻微位移，展现出了极强的闭环纠错能力。

## HMI 开源主表入口

[openpi](https://github.com/Physical-Intelligence/openpi) 收录于具身智能研究室 [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)。

主表定位：官方仓同时维护流匹配 π0、π0-FAST 与 π0.5，并提供检查点、数据配置、微调与推理服务。接入新本体时关键是动作归一化、字段映射与推理频率对齐。本库以本方法页（及 [π0.7](./pi07-policy.md)）承载 openpi 入口，不另建重复实体。

覆盖核对见 [HMI 开源项目主表覆盖索引](../queries/hmi-opensource-projects-coverage.md)。

## 关联页面
- [π₀ 论文实体](../entities/paper-pi0.md) — arXiv:2410.24164 canonical 节点
- [VLA / 世界模型 14 篇阅读路线](../overview/vla-wm-reading-roadmap-14-papers-technology-map.md)
- [VLA (Vision-Language-Action Models)](./vla.md)
- [π₀.7（Pi-zero 0.7）通才 VLA](./pi07-policy.md) — 同一 π 系路线在「多模态提示 + 异质数据对齐」上的后继公开版本（2026）
- [Diffusion Policy](./diffusion-policy.md)
- [Action Tokenization (动作分词)](../formalizations/vla-tokenization.md)
- [Cross-modal Attention (跨模态注意力)](../formalizations/cross-modal-attention.md)
- [LWD（Learning while Deploying）](./lwd.md) — 其 QAM 组件正是为 flow-based 动作头（如 π₀）设计的策略抽取方法
- [STEAM（自监督时序集成 advantage 建模）](../entities/paper-steam-advantage-modeling.md) — 以 π₀ 为策略骨干，用 CFGRL 做离线后训练提纯
- [FM-VLA（力觉记忆 VLA）](../entities/paper-fm-vla.md) — 以 π₀.₅ 为骨干，在 flow-matching 动作专家后缀注入力觉长程记忆 token
- [χ₀ / kai0](../entities/paper-kai0.md) — 在 openpi π₀.₅ 上做 MA/SA/TDA 分布对齐的协同叠衣配方（代码已开源）
- [Sergey Levine：表达力更强的连续动作策略](../overview/sergey-levine-diffusion-expressive-policies.md) — flow/扩散动作头抬升 IL 与 offline RL 的讲者坐标（Simons 2026）
- [SPD](../entities/paper-spd.md) — 灵巧真机消融：π0 风格单帧长 chunk 弱于历史+短 chunk（CoRL 2026）
- [DexHoldem](../entities/paper-dexholdem.md) — ShadowHand 扑克真机：π₀ / π₀.₅ SPSR 并列 47.5%，芯片 pull 仍弱

## 参考来源
- [SPD 论文归档](../../sources/papers/spd_corl_2026.md) — 灵巧真机上 π0 风格单帧长 chunk 对照
- Black, K., et al. (2024). *π₀: A Vision-Language-Action Flow Model for General Robot Control*.
- [Physical Intelligence Blog](https://www.physicalintelligence.company/blog/pi0).
- [sources/papers/pi07.md](../../sources/papers/pi07.md) — π₀.₇ 后继工作与多模态提示条件（若只关心 π₀ 本体的历史语境可略读摘录节）
- [sources/papers/chi0_kai0_arxiv_2602_09021.md](../../sources/papers/chi0_kai0_arxiv_2602_09021.md) — χ₀ / kai0 后训练与部署对齐
- [sources/courses/sergey_levine_diffusion_rl_robotics_simons_youtube.md](../../sources/courses/sergey_levine_diffusion_rl_robotics_simons_youtube.md) — Levine：flow/扩散作为大规模控制模型动作头的上游读法
