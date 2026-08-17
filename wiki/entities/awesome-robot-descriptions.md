---
type: entity
tags: [tooling, urdf, mjcf, xacro, curated-list, repo, inria]
status: complete
updated: 2026-08-17
summary: "Awesome Robot Descriptions 是 URDF/Xacro/MJCF 的策展目录：按机型族给出上游链接、许可证，以及 visual/inertia/collision 是否齐全；列表 CC0，模型许可证各异。"
related:
  - ./robot-descriptions-py.md
  - ../comparisons/robot-description-catalogs.md
  - ../concepts/urdf-robot-description.md
  - ./mujoco.md
  - ./urdf-studio.md
  - ./fiveages-sim-robot-descriptions.md
sources:
  - ../../sources/repos/awesome-robot-descriptions.md
  - ../../sources/repos/robot-descriptions-py.md
---

# Awesome Robot Descriptions

[Awesome Robot Descriptions](https://github.com/robot-descriptions/awesome-robot-descriptions) 是 `robot-descriptions` 组织维护的 **开源机器人描述策展列表**。它不下载网格、不提供 loader，只回答：「这个机型的 URDF / Xacro / MJCF 在哪、许可证是什么、三项几何字段齐不齐。」

## 一句话定义

一份按臂、人形、四足等分类的 Markdown 目录，用 **格式链接 + 许可证 + visual/inertia/collision 勾选** 帮你决定能不能把某份描述拿去仿真或控制。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| URDF | Unified Robot Description Format | 列表主格式之一 |
| MJCF | MuJoCo XML Format | 常链到 Menagerie 子目录 |
| Xacro | XML Macros | ROS 侧参数化 URDF，需预处理 |
| CC0 | Creative Commons Zero | **列表文本** 的许可证，不含模型文件 |

## 核心信息

| 字段 | 内容 |
|------|------|
| 组织 | [`robot-descriptions`](https://github.com/robot-descriptions)；与 [robot_descriptions.py](./robot-descriptions-py.md) 同组织（Stéphane Caron 等，INRIA 谱系） |
| 机构 | 法国国家信息与自动化研究所（INRIA） |
| 许可 | 列表 **CC0-1.0** |
| 开源 | **已开源**（无运行时代码） |
| 配套 | 大多数条目可被 [robot_descriptions.py](./robot-descriptions-py.md) import |
| Stars | 1,635（2026-08-17 核查） |

## 为什么重要

- **勾选比「有个 URDF 链接」更有用：** 缺惯量的模型能看不能做动力学；缺碰撞的模型能渲染不能做接触。FANUC M-710iC 等行直接标 **Inertias ✖️**。
- **许可证前置：** 列表把 GPL、CC-BY-NC、厂商条款和 ✖️（未给出许可）摊在同一列，避免把 Awesome 当「全部可商用资产库」。
- **与加载器分工：** 发现走本页；Python 实验走 `robot_descriptions.py`；ROS 2 / 国内新机走 [fiveages-sim](./fiveages-sim-robot-descriptions.md)。

## 核心原理

分类与 [robot_descriptions.py](./robot-descriptions-py.md) 对齐：Arms、Bipeds、Dual Arms、Drones、Educational、End Effectors、Humanoids、Mobile Manipulators、Quadrupeds、Wheeled。

同一机型可出现多行（官方 Xacro vs Menagerie MJCF vs 第三方 URDF）。选型时同时看：

1. **格式是否匹配后端**（MuJoCo 优先 MJCF；Pinocchio 优先 URDF）。
2. **三项勾选** 是否覆盖你的任务（纯可视化 vs 动力学 vs 碰撞）。
3. **许可证** 是否允许你的发布场景。

相关列表：Stéphane Caron 的 [Awesome Open Source Robots](https://github.com/stephane-caron/awesome-open-source-robots)、ami-iit 的 [Awesome URDF](https://github.com/ami-iit/awesome-urdf)。

## 工程实践

1. 需要可运行对象时，不要 clone 本列表当数据包，改 `pip install robot_descriptions`。
2. 勾选是人工维护，上游改网格后可能过时；加载失败以 loader CI 与上游 README 为准。
3. NAO 等条目对 meshes 标「需另下」，列表里的 ✔️ 只描述「格式里声明了 visual」，不保证仓内自带 mesh 二进制。
4. 提交新机型走仓库 `CONTRIBUTING.md`，保持与 Python 包分类一致，减少发现层与加载层分叉。

## 局限与风险

- **不是镜像：** 链接会 404；国内部分新机（EngineAI、多数轮式人形）覆盖弱于 fiveages-sim。
- **Gallery 依赖 README 图：** 对本库 wiki 无信息增量，不转存图片。
- **CC0 只覆盖列表编辑作品**，复制表内模型仍受原 LICENSE 约束。

## 关联页面

- [robot_descriptions.py](./robot-descriptions-py.md)
- [机器人描述目录选型](../comparisons/robot-description-catalogs.md)
- [URDF](../concepts/urdf-robot-description.md)
- [MuJoCo](./mujoco.md)
- [URDF-Studio](./urdf-studio.md)
- [fiveages-sim robot_descriptions](./fiveages-sim-robot-descriptions.md)

## 参考来源

- [Awesome Robot Descriptions 归档](../../sources/repos/awesome-robot-descriptions.md)
- [robot_descriptions.py 归档](../../sources/repos/robot-descriptions-py.md)

## 推荐继续阅读

- 列表仓库：<https://github.com/robot-descriptions/awesome-robot-descriptions>
- [Awesome URDF](https://github.com/ami-iit/awesome-urdf)
