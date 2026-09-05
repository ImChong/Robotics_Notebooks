---
type: entity
tags: [curated-list, world-models, wam, vla, model-based-rl, embodied-ai, autonomous-driving]
status: complete
updated: 2026-09-05
related:
  - ../overview/vla-wm-reading-roadmap-14-papers-technology-map.md
  - ./paper-sa-2505-11528-ladi-wm-a-latent-diffusion-based-world-model-for.md
  - ./paper-sa-2602-11075-rise-self-improving-robot-policy-with-compositio.md
  - ./paper-sa-2601-03782-pointworld.md
  - ../overview/sun-awesome-wm-technology-map.md
  - ../concepts/world-action-models.md
  - ../methods/generative-world-models.md
  - ../methods/model-based-rl.md
  - ../methods/vla.md
  - ../overview/robot-world-models-training-loop-taxonomy.md
  - ../overview/robot-world-models-action-consequence-technology-map.md
  - ./awesome-egocentric-vision.md
  - ./awesome-touch.md
  - ./awesome-real2sim2real.md
sources:
  - ../../sources/repos/awesome-world-models.md
  - ../../sources/papers/sun_awesome_wm_catalog.md
summary: "sun254667 维护的 Awesome World Models：覆盖范式分册、WAM/VLA、Model-Based RL、具身/驾驶/GUI 应用与仿真平台的学习友好型 World Model 策展清单；站内已节点化为技术地图 + paper-sa 详情页。"
---

# Awesome World Models（sun254667 精选集）

**Awesome World Models**（GitHub：[`sun254667/awesome-world-models`](https://github.com/sun254667/awesome-world-models)）是一份 **World Model 研究资源** 的 curated 列表：按 **范式 → 交互/WAM → MBRL → 应用域 → 数据/仿真** 组织论文、教程、数据集与平台入口，并提供中英双语 README。

## 一句话定义

面向 **研究入门与工业部署** 的 World Model **全谱索引入口**（从 Language/Video/3D·4D 范式到机器人与驾驶应用）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测模型（观测/状态演化） |
| WAM | World Action Model | 世界预测与动作生成联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略范式 |
| MBRL | Model-Based Reinforcement Learning | 用学得模型做规划/策略优化的 RL |
| 3D/4D | 3D / 4D World Modeling | 几何与时空占用/场景生成路线 |

## 为什么重要

- **降低检索成本**：把「什么叫 World Model」「怎么接到 VLA/WAM」「机器人 vs 驾驶」拆成可浏览分册，避免只盯单篇综述。
- **与站内 taxonomy 对齐**：§6 WAM & VLA、§7 MBRL、§8.1 Robotics 可直接挂到 [WAM](../concepts/world-action-models.md)、[生成式 WM](../methods/generative-world-models.md)、[机器人 WM 训练闭环](../overview/robot-world-models-training-loop-taxonomy.md)。
- **姊妹清单互补**：同维护者的 [Egocentric](./awesome-egocentric-vision.md) / [Touch](./awesome-touch.md) / [Real2Sim2Real](./awesome-real2sim2real.md) 分别补第一人称、触觉与迁移闭环横切面。

## 站内节点化

清单内可解析论文条目已映射为独立详情节点，见：

- **技术地图（全部分组表）：** [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md)
- **目录 source：** [sun_awesome_wm_catalog.md](../../sources/papers/sun_awesome_wm_catalog.md)
- **命名：** 新建索引级实体为 `wiki/entities/paper-sa-*`；若库内已有同 arXiv 的 `paper-*` canonical 页则复用。

## 核心结构（怎么读）

| 区块 | 内容侧重 |
|------|----------|
| Surveys / Tutorials / Foundation | 定义、多轴 taxonomy、起步实现（如 minWM、StableWM） |
| Core Paradigms | Language / Video / 3D·4D / Latent / Object-Centric / Neuro-Symbolic / Foundation |
| Interactive + WAM & VLA | Cascaded vs Joint；WM 用于 VLA 训练与评测 |
| MBRL + Applications | 经典 Dreamer 系 → 具身操作/导航/loco-manip、驾驶、GUI Agents |
| Data / Techniques / Sim | 基准、记忆与推理加速、物理引擎与可微仿真 |

## 局限与使用注意

- **清单滞后**：awesome 依赖 PR 维护；关键论文以 arXiv / 官方仓为准。
- **非可运行栈**：无训练代码；条目开源状态需逐条核项目页。
- **与 Awesome-WAM 分工**：OpenMOSS [Awesome-WAM](../../sources/repos/awesome-wam-openmoss.md) 深耕 WAM 两条架构线；本清单覆盖更广的 WM 全谱。

## 关联页面

- [VLA / 世界模型 14 篇阅读路线](../overview/vla-wm-reading-roadmap-14-papers-technology-map.md) — LaDi-WM / RISE / PointWorld 已从索引级升格
- [Awesome World Models 技术地图](../overview/sun-awesome-wm-technology-map.md) — 清单论文 → 独立详情节点
- [World Action Models（WAM）](../concepts/world-action-models.md) — 联合世界–动作范式
- [Generative World Models](../methods/generative-world-models.md) — 生成式 / 视频世界模型方法页
- [Model-Based RL](../methods/model-based-rl.md) — 清单 §7 经典与连续控制对照
- [VLA](../methods/vla.md) — 与 WAM/WM-for-VLA 评测交叉
- [机器人世界模型训练闭环](../overview/robot-world-models-training-loop-taxonomy.md)
- [动作后果技术地图](../overview/robot-world-models-action-consequence-technology-map.md)
- [Awesome Egocentric Vision](./awesome-egocentric-vision.md) / [Awesome Touch](./awesome-touch.md) / [Awesome-Real2Sim2Real](./awesome-real2sim2real.md) — 同系列姊妹清单

## 参考来源

- [sources/repos/awesome-world-models.md](../../sources/repos/awesome-world-models.md)
- [sources/papers/sun_awesome_wm_catalog.md](../../sources/papers/sun_awesome_wm_catalog.md)

## 推荐继续阅读

- [GitHub 仓库 README（英文全文）](https://github.com/sun254667/awesome-world-models)
- [中文 README](https://github.com/sun254667/awesome-world-models/blob/main/README.zh-CN.md)
- [OpenMOSS Awesome-WAM](https://github.com/OpenMOSS/Awesome-WAM) — WAM 专题策展对照
