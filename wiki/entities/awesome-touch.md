---
type: entity
tags: [curated-list, tactile, visuo-tactile, vla, world-models, wam, contact-rich, sim2real]
status: complete
updated: 2026-08-10
related:
  - ../overview/sun-awesome-touch-technology-map.md
  - ../overview/hub-tactile.md
  - ../concepts/tactile-sensing.md
  - ../concepts/visuo-tactile-fusion.md
  - ../concepts/contact-rich-manipulation.md
  - ../queries/tactile-feedback-in-rl.md
  - ./paper-touchworld-tactile-foundation-dexterous-manipulation.md
  - ./paper-vitacworld.md
  - ./paper-vt-wam-visuotactile-contact-rich.md
  - ./paper-taco-tactile-sensor-benchmark.md
  - ./awesome-world-models.md
  - ./awesome-real2sim2real.md
sources:
  - ../../sources/repos/awesome-touch.md
  - ../../sources/papers/sun_awesome_touch_catalog.md
summary: "sun254667 维护的 Awesome Touch：2025–2026 触觉操作精选集；站内已节点化为技术地图 + paper-sa 详情页。"
---

# Awesome Touch（sun254667 精选集）

**Awesome Touch**（GitHub：[`sun254667/awesome-touch`](https://github.com/sun254667/awesome-touch)）是一份面向 **接触丰富操作** 的触觉研究 curated 列表：时间窗锁定 **2025.01–2026.07**，强调触觉与 **VLA / World Models / WAM** 的融合，而非泛泛的传感器硬件汇编。

## 一句话定义

**触觉 × 语言–动作模型** 的近期文献索引入口（VTLA、视触觉 WM、Tactile WAM、策略与 Sim2Real）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VTLA | Vision-Tactile-Language-Action | 视–触–语言–动作统一策略 |
| WAM | World Action Model | 世界预测与动作联合建模 |
| WM | World Model | 视触觉前向预测 / foresight |
| Sim2Real | Simulation to Real | 触觉仿真到真机迁移 |
| HOI | Hand–Object Interaction | 接触丰富手物交互场景 |

## 为什么重要

- **视觉有盲区**：插入、滑移、软体接触下触觉是「接触真相」；清单把 2025–2026 的 VLA/WM 融合工作集中可见。
- **与站内触觉链对齐**：可挂到 [触觉知识链](../overview/hub-tactile.md)、[Tactile Sensing](../concepts/tactile-sensing.md)、[Visuo-Tactile Fusion](../concepts/visuo-tactile-fusion.md)。
- **挑战表述可操作**：README 归纳 Tactile Pollution、Modality Collapse、数据稀缺等，适合作为选型时的风险清单。

## 站内节点化

- **技术地图：** [Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md)
- **目录 source：** [sun_awesome_touch_catalog.md](../../sources/papers/sun_awesome_touch_catalog.md)
- 新建索引级实体 `paper-sa-*`；已有同 arXiv canonical `paper-*` 则复用。

## 核心结构（怎么读）

| 区块 | 内容侧重 |
|------|----------|
| Surveys | 多模态传感 taxonomy、人形 loco-manip 中的全身触觉角色 |
| Tactile VLA | UniTacVLA / OmniVTLA / TacMamba 等统一与异步融合 |
| Visuo-Tactile WM | OmniVTA、ViTacWorld、TouchWorld、ContactWorld |
| Tactile WAM | 接触丰富场景下的世界–动作联合 |
| Policy / Sim2Real / Hardware / Data | IL·RL·Diffusion；触觉仿真迁移；传感器与基准 |

## 局限与使用注意

- **时间窗窄**：刻意聚焦 2025–2026；经典 GelSight 系硬件综述需另查站内概念页与更早文献。
- **清单 ≠ 可复现**：多数条目需单独核代码/数据集是否开放。
- **模态塌缩风险**：高带宽视觉易淹没稀疏触觉——工程上优先看「融合时机 / 残差表征 / 异步反射」类设计。

## 关联页面

- [Awesome Touch 技术地图](../overview/sun-awesome-touch-technology-map.md) — 清单论文 → 独立详情节点
- [触觉与力觉（知识链汇总）](../overview/hub-tactile.md)
- [Tactile Sensing](../concepts/tactile-sensing.md) / [Visuo-Tactile Fusion](../concepts/visuo-tactile-fusion.md)
- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md)
- [Tactile Feedback in RL（Query）](../queries/tactile-feedback-in-rl.md)
- [TouchWorld](./paper-touchworld-tactile-foundation-dexterous-manipulation.md) / [ViTacWorld](./paper-vitacworld.md) / [VT-WAM](./paper-vt-wam-visuotactile-contact-rich.md)
- [TacO 触觉传感器基准](./paper-taco-tactile-sensor-benchmark.md)
- [Awesome World Models](./awesome-world-models.md) / [Awesome-Real2Sim2Real](./awesome-real2sim2real.md)

## 参考来源

- [sources/repos/awesome-touch.md](../../sources/repos/awesome-touch.md)
- [sources/papers/sun_awesome_touch_catalog.md](../../sources/papers/sun_awesome_touch_catalog.md)

## 推荐继续阅读

- [GitHub 仓库 README](https://github.com/sun254667/awesome-touch)
- [ViTacWorld（arXiv:2607.22530）](https://arxiv.org/abs/2607.22530) — 清单内视触觉 WM 代表
- [TouchWorld（arXiv:2607.07287）](https://arxiv.org/abs/2607.07287) — 预测–反应触觉 foundation
