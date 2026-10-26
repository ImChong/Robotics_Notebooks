---
type: entity
project_id: robomind
project: https://x-humanoid-robomind.github.io/
arxiv: "2412.13877"
papers: ["2412.13877", "2512.24653"]
tags: [dataset, manipulation, multi-embodiment, imitation-learning, vla, tactile, mobile-manipulation, x-humanoid]
status: complete
updated: 2026-10-26
summary: "X-Humanoid 多本体机器人操作数据集与基准；RoboMIND 1.x 提供跨任务示范，2.0 扩展至双臂移动操作、触觉和仿真数据。"
related:
  - ../tasks/manipulation.md
  - ../methods/vla.md
  - ../entities/x-humanoid.md
sources:
  - ../../sources/papers/robomind-v1-rss21.md
  - ../../sources/papers/robomind-v2-arxiv-2512-24653.md
  - ../../sources/sites/robomind.md
  - ../../sources/repos/open-x-humanoid.md
---

# RoboMIND

**RoboMIND** 是北京人形机器人创新中心（X-Humanoid）发布的多本体机器人操作数据集与基准，提供真实遥操作轨迹、语言任务描述和多视角观测；2.0 扩展了双臂移动操作、触觉和数字孪生数据。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RSS | Robotics: Science and Systems | RoboMIND 1.0 论文发表会议 |
| VLA | Vision-Language-Action | 视觉、语言到机器人动作的策略模型 |
| HDF5 | Hierarchical Data Format 5 | RoboMIND 1.x 轨迹文件格式 |
| MIND-2 | Multimodal Intelligence for Networked Devices 2 | RoboMIND 2.0 配套的分层操作系统 |

## 为什么重要

- 通过统一采集协议整合多个机器人本体的数据，便于研究跨任务和跨本体泛化。
- 数据同时包含观测、机器人状态与语言任务描述，可用于模仿学习和 VLA 微调评测。
- 版本演进把数据规模从多本体操作扩展到触觉接触、移动操作与仿真到真实迁移场景。

## 核心信息

| 字段 | 内容 |
|------|------|
| 发布团队 | 北京人形机器人创新中心（X-Humanoid）；项目页同时列出北京大学与北京智源人工智能研究院作者单位 |
| 主论文 | RoboMIND 1.0，RSS 2025；arXiv:2412.13877 |
| 配套论文 | RoboMIND 2.0，arXiv:2512.24653 |
| 官方项目页 | [RoboMIND](https://x-humanoid-robomind.github.io/) |
| 数据下载 | [RoboMIND 1.x（Hugging Face）](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND)；[RoboMIND 2.0（ModelScope）](https://modelscope.cn/collections/X-Humanoid/RoboMIND20) |
| 相关工具 | [Open-X-Humanoid](https://github.com/Open-X-Humanoid) 组织下的 RoboMIND-dataset-utils 与训练工具链 |

## 版本演进

| 版本 | 论文/数据卡口径 | 任务与本体 | 主要变化 |
|------|----------------|------------|----------|
| 1.0 | 初始数据集约 55K 条轨迹、279 个任务、69 类物体（数据卡版本说明） | 4 种本体；Franka、天工人形、AgileX 双臂、UR5e | 统一遥操作采集协议；包含多视角观测、本体状态、语言指令及失败演示 |
| 1.1/1.2 | 扩展至约 107K 条轨迹、479 个任务、96 类物体 | 延续多本体操作数据 | 1.2 增加 Upright_Cup 任务及帧级语言标注更新 |
| 2.0 | 论文报告超过 310K 条双臂操作轨迹、739 个任务 | 6 种本体 | 纳入约 12K 触觉增强 episode、20K 移动操作轨迹及额外 20K 仿真轨迹 |

论文摘要与数据卡所处的发布版本不同，因此 1.0 论文中使用的规模描述与数据卡「初始 1.0」统计口径并不完全相同；引用规模时应同时标明版本和来源，勿把 1.0、1.2 与 2.0 的数量混用。

## 数据与系统结构

RoboMIND 1.x 以人类遥操作为主，包含多视角图像、本体状态和语言描述。数据卡说明其 HDF5 目录按机器人与相机配置组织；不同子集的图像通道可能分别采用 RGB 或 BGR，使用前需按数据卡检查。Franka、UR 与天工等本体在轨迹长度分布和任务结构上也不同，不能仅按文件格式假定动作空间一致。

RoboMIND 2.0 面向更长程的双臂和移动操作任务，并加入触觉及数字孪生数据。配套 MIND-2 将高层语义规划器 MIND-2-VLM 与低层视觉语言动作执行器 MIND-2-VLA 组合：前者把自然语言任务拆成落地子目标，后者生成结合本体状态的电机动作。MIND-2 是论文提出的配套系统，不等同于数据集本身。

## 工程实践

- 下载前确认目标版本、子集、机器人本体和数据格式；HF 数据卡显示需同意共享账号信息后访问文件。
- 训练或评测时统一核对坐标系、关节顺序、夹爪定义、采样频率和图像颜色通道；跨本体使用需重定向或显式适配动作空间。
- 官方 GitHub 组织已有数据集校验与末端位姿重算工具；这些仓库与数据发布入口分开，使用时应分别核对工具仓 README、依赖和许可证。
- ModelScope 集合页与数据卡可能持续更新，记录实验时应保存使用的集合版本或下载快照。

## 局限与风险

- RoboMIND 1.0、1.1/1.2 和 2.0 的数据规模与内容不同，汇报结果时不能把版本混为同一数据集快照。
- 公开下载入口不代表所有子集采用相同授权或访问条件；复用前应检查对应数据卡与集合条款。
- 轨迹数量本身不能代表任务成功率、标注一致性或跨本体可迁移性；应结合具体任务、本体、传感器和评测切分解释结果。
- 项目页链接到 GitHub，其中一项是项目网站源码；可执行数据工具与训练工具链位于 Open-X-Humanoid 组织的独立仓库，不应将网站源码误认作完整训练实现。

## 关联页面

- [机器人操作](../tasks/manipulation.md)
- [视觉语言动作模型](../methods/vla.md)
- [X-Humanoid 开源生态](../entities/x-humanoid.md)
- [RoboMIND 数据集工具](../entities/cn-os-robomind-dataset-utils.md)

## 参考来源

- [RoboMIND 1.0 RSS 论文归档](../../sources/papers/robomind-v1-rss21.md)
- [RoboMIND 2.0 arXiv 论文归档](../../sources/papers/robomind-v2-arxiv-2512-24653.md)
- [RoboMIND 官方项目页归档](../../sources/sites/robomind.md)
- [Open-X-Humanoid GitHub 组织归档](../../sources/repos/open-x-humanoid.md)

## 推荐继续阅读

- [RoboMIND 1.0 论文（RSS 2025）](https://www.roboticsproceedings.org/rss21/p152.html)
- [RoboMIND 2.0 论文](https://arxiv.org/abs/2512.24653)
- [RoboMIND 1.x 数据集卡](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND)
- [RoboMIND 2.0 ModelScope 集合](https://modelscope.cn/collections/X-Humanoid/RoboMIND20)
