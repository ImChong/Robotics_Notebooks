# light-loco-parkour.github.io（Light-Loco-Parkour / LightParkour 项目页）

- **标题：** Growing Humanoid Parkour Skills through Real2Sim2Real — LightParkour（站内叙事名）；论文题 *Light-Loco-Parkour: Versatile Perceptive Whole-Body Locomotion via Multi-Skill Distillation*
- **类型：** site / project-page
- **URL：** <https://light-loco-parkour.github.io/>
- **PDF：** <https://light-loco-parkour.github.io/paper.pdf>
- **视频：** <https://youtu.be/96Rfm7OmHjY>
- **机构：** Light Origins（光原点）
- **平台：** Lightbot 0（自研 90 cm / 18.9 kg / 21 DoF 人形）
- **配套论文归档：** [`sources/papers/light_loco_parkour_light_origins_2026.md`](../papers/light_loco_parkour_light_origins_2026.md)
- **入库日期：** 2026-08-04

## 一句话摘要

Light Origins 人形跑酷项目页：用 **Real2Sim2Real** 把稀疏人类动作种子扩成地形配对参考，再经多专家蒸馏 + 转移学习压成**单一机载深度策略**，在 Lightbot 0 上零样本执行攀爬 / vault / 稀疏落足等全身技能，且**无技能标签、无运行时运动图**。

## 开源状态（步骤 2.5，截至 2026-08-04）

| 资源 | 状态 |
|------|------|
| 项目页 + PDF + YouTube | **已发布** |
| arXiv | **未挂**（页上/BibTeX 无编号） |
| 训练 / 推理代码 | **未列出**（页上无 Code 按钮；GitHub 组织仅有 `light-loco-parkour.github.io` 站点仓） |

**结论：确认未开源（无可运行官方实现）。** wiki「源码运行时序图」标 **不适用**。

## 公开信息要点

- **日期标头：** 3 August 2026。
- **叙事主线：** Real → Sim（物理修复接触 + 课程抬障 45→75 cm）→ Real（深度策略部署）。
- **硬件卖点：** Lightbot 0；机载深度 + 同策略室内外场测。
- **BibTeX：** `@misc{chen2026lightlocoparkour, ... url={https://light-loco-parkour.github.io/}}`。

## 为何值得保留

- 与 [PHP](../../wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md) 同属人形感知跑酷，但强调 **无 one-hot 技能指令 / 无运行时 motion generator**，并给出对 PHP / MGMT 的直接对照表。
- Real2Sim2Real 种子扩张（单种子 → 地形族）是可操作的数据工程配方，值得对照 OmniRetarget / HIL。

## 关联资料

- 论文归档：[`sources/papers/light_loco_parkour_light_origins_2026.md`](../papers/light_loco_parkour_light_origins_2026.md)
- 沉淀实体：[`wiki/entities/paper-light-loco-parkour.md`](../../wiki/entities/paper-light-loco-parkour.md)
