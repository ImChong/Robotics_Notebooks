---
type: entity
tags: [curated-list, physical-ai, embodied-ai, vla, world-models, sim2real, robotics-foundation-models]
status: complete
updated: 2026-09-20
related:
  - ../methods/vla.md
  - ../concepts/sim2real.md
  - ../methods/generative-world-models.md
  - ./awesome-physical-ai-aichr.md
  - ../comparisons/awesome-physical-ai-curated-lists.md
  - ../overview/awesome-physical-ai-technology-map.md
  - ./lerobot.md
sources:
  - ../../sources/repos/awesome-physical-ai-natnew.md
  - ../../sources/sites/awesome-physical-ai-natnew-github-io.md
summary: "natnew 维护的工程导向 Physical AI 资源地图：14 canonical 类别、~229 条目、MIT 仓库 + GitHub Pages 导航站。"
---

# awesome-physical-ai（natnew）

[`natnew/awesome-physical-ai`](https://github.com/natnew/awesome-physical-ai) 是一份 **工程导向** 的 Physical AI / Embodied AI 策展地图：用 **14 个 canonical 类别** 组织仿真器、数据集、VLA/RFM、World Models、Sim2Real、安全评估与生产模式，并配套 [在线文档站](https://natnew.github.io/awesome-physical-ai/docs/overview) 做快速定向。

## 一句话定义

**Physical AI 全栈导航 catalog** — README 为真源、docs 站为 Learn/Build/Deploy 路由层，覆盖从 CartPole 入门到 VLA/WM 前沿的 ~229 条资源。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Physical AI | Physical Artificial Intelligence | 感知–推理–行动闭环的物理智能 |
| VLA | Vision-Language-Action | 清单 RFM 区核心范式 |
| RFM | Robotics Foundation Model | 机器人基础模型条目 |
| Sim2Real | Simulation to Real | 独立 canonical 类别 |
| WM | World Model | 与 RFM 并列的 canonical 类 |

## 为什么重要

- **14 类 taxonomy + CI：** CONTRIBUTING 按 docs 站 14 类定义质量栏与 entry-count 检查，比扁平 awesome 更利于 **长期策展**。
- **部署视角完整：** 除 Manipulation/Locomotion 外单列 Safety、Governance、Production Patterns，适合 **研究 → 系统** 读者。
- **Quick start 路径：** README 给出 Gymnasium → MuJoCo → LeRobot → OpenVLA 的 staged 入门，降低「Physical AI 从哪读起」摩擦。
- **与 aichr 同名仓对照：** 见 [Physical AI 策展清单对比](../comparisons/awesome-physical-ai-curated-lists.md)。


## 子节点覆盖（2026-09-20 纵深）

去重后 **384** 条独立详情节点（新建 241，复用 143；两清单同时出现 33）。

完整子节点表见 [Physical AI 技术地图](../overview/awesome-physical-ai-technology-map.md)；并集目录见 [awesome-physical-ai-union-catalog.md](../../sources/repos/awesome-physical-ai-union-catalog.md)。

## 核心结构

| 类别 | 侧重 |
|------|------|
| Simulators / Datasets / Benchmarks | 评测与数据 harness |
| RFM / World Models | VLA、通才策略、世界模型 |
| Manipulation / Locomotion | 任务向方法入口 |
| Sim2Real / Safety / Production | 迁移、鲁棒与运维模式 |
| Courses / Companies + 附录 | 学习路径与行业跟踪 |

**文档站：** Overview 用 Learn/Build/Deploy/Measure/Track/Practice overlay 路由；**docs 14 类优先于 README 扫描节**。

## 局限与使用注意

- **README-first：** 完整条目仍在 GitHub README；docs 不重复全量列表。
- **非 SDK：** 无统一安装；LeRobot/Isaac Lab 等须跟官方仓核版本。
- **条目开源逐条核：** 列表不做项目页核查；复现前执行步骤 2.5。

## 关联页面

- [Physical AI 技术地图](../overview/awesome-physical-ai-technology-map.md)
- [awesome-physical-ai（aichr）](./awesome-physical-ai-aichr.md)
- [Physical AI 策展清单对比](../comparisons/awesome-physical-ai-curated-lists.md)
- [VLA](../methods/vla.md)
- [Sim2Real](../concepts/sim2real.md)
- [LeRobot](./lerobot.md)

## 参考来源

- [sources/repos/awesome-physical-ai-natnew.md](../../sources/repos/awesome-physical-ai-natnew.md)
- [sources/sites/awesome-physical-ai-natnew-github-io.md](../../sources/sites/awesome-physical-ai-natnew-github-io.md)

## 推荐继续阅读

- GitHub：<https://github.com/natnew/awesome-physical-ai>
- 文档 Overview：<https://natnew.github.io/awesome-physical-ai/docs/overview>
