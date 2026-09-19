---
type: entity
tags: [curated-list, locomotion, humanoid, whole-body-control, reinforcement-learning, sustech]
status: complete
updated: 2026-09-19
related:
  - ../tasks/loco-manipulation.md
  - ../tasks/locomotion.md
  - ../concepts/whole-body-control.md
  - ./awesome-legged-locomotion-learning.md
  - ../../sources/repos/awesome-humanoid-robot-learning.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/repos/awesome-legged-robot-learning-clearlab.md
summary: "SUSTech ClearLab 维护的腿足机器人学习 arXiv 精选：Locomotion、Loco-Manipulation/WBC、Physics-Based Character Animation 三线索引。"
---

# Awesome-Legged-Robot-Learning（ClearLab @ SUSTech）

[`clearlab-sustech/Awesome-Legged-Robot-Learning`](https://github.com/clearlab-sustech/Awesome-Legged-Robot-Learning) 是南方科技大学 **ClearLab** 维护的 **腿足机器人学习** 论文精选，覆盖 RL、Sim2Real 与人形 **全身控制 / loco-manipulation** 前沿 arXiv 条目。

## 一句话定义

**2024–2025 腿足学习 arXiv 快讯索引** — 三线分区（Locomotion / WBC / Character Animation）追踪 parkour、HOMIE/TWIST 与 MaskedMimic 等热点。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WBC | Whole-Body Control | Loco-Manipulation 分区核心 |
| RL | Reinforcement Learning | Locomotion 主线 |
| AMP | Adversarial Motion Priors | Character Animation 区代表 |
| Sim2Real | Simulation to Real | 与 locomotion 论文交叉 |
| H2H | Human-to-Humanoid | 遥操作/影子学习条目 |

## 为什么重要

- **人形 WBC 密度高：** Loco-Manipulation 区集中 HOMIE、TWIST、HOVER、BeyondMimic、SONIC 等 2024–2025 代表工作，适合跟踪 **全身技能** 而非仅四足 trotting。
- **与 YanjieZe 人形清单互补：** Acknowledgements 明确引用 [awesome-humanoid-robot-learning](https://github.com/YanjieZe/awesome-humanoid-robot-learning)；ClearLab 版更短、更新 arXiv 尾部。
- **Character Animation 桥：** DeepMimic / AMP / MaskedMimic 区帮助把 **图形学 motion prior** 与腿足 RL 文献对齐。

## 核心结构

| 分区 | 代表方向 |
|------|----------|
| Locomotion | DreamWaQ、Parkour、Multi-Loco、LocoFormer |
| Loco-Manipulation & WBC | GR00T N1、ASAP、LangWBC、KungfuBot |
| Physics-Based Character Animation | AMP、MaskedMimic、TokenHSI、BFM |

## 局限与使用注意

- **体量较小（~27 stars）：** 依赖 ClearLab 手动维护；遗漏项可对照 [gaiyi7788 腿足清单](./awesome-legged-locomotion-learning.md) 与 [awesome-humanoid-robot-learning](../../sources/repos/awesome-humanoid-robot-learning.md)。
- **仅 arXiv 链接：** 无代码/开源 badge；复现前须打开各论文项目页（见 ingest 步骤 2.5）。
- **MIT 列表、条目各自许可：** 列表文本 MIT；论文与代码遵循原作者。

## 关联页面

- [Loco-Manipulation](../tasks/loco-manipulation.md)
- [Whole-Body Control](../concepts/whole-body-control.md)
- [awesome-legged-locomotion-learning](./awesome-legged-locomotion-learning.md)
- [Locomotion](../tasks/locomotion.md)

## 参考来源

- [sources/repos/awesome-legged-robot-learning-clearlab.md](../../sources/repos/awesome-legged-robot-learning-clearlab.md)

## 推荐继续阅读

- GitHub：<https://github.com/clearlab-sustech/Awesome-Legged-Robot-Learning>
- [awesome-humanoid-robot-learning](https://github.com/YanjieZe/awesome-humanoid-robot-learning)
