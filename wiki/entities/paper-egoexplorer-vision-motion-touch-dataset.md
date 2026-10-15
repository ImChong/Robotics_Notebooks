---
type: entity
tags: [paper, dataset, egocentric, human-motion, loco-manipulation, wearable-capture, tactile, imitation-learning, shanghai-ai-lab, zhejiang-university, shanghai-jiao-tong-university]
status: complete
updated: 2026-10-15
arxiv: "2610.18872"
project_id: egoexplorer-vision-motion-touch-dataset
project: https://internrobotics.github.io/egoexplorer.github.io/
dataset: https://huggingface.co/datasets/InternRobotics/EgoExplorer
code: https://github.com/InternRobotics/EgoExplorer
related:
  - ../queries/humanoid-robot-data-collection-landscape.md
  - ../tasks/manipulation.md
  - ../tasks/teleoperation.md
  - ./paper-egoverse.md
  - ./eidon-tracker-pov-dataset.md
sources:
  - ../../sources/papers/egoexplorer_arxiv_2610_18872.md
summary: "EgoExplorer（arXiv:2610.18872）：5 个 inside-out tracker + 头戴相机采 2,715 段 / 100 h 人类移动操作数据；C³ 将稀疏身体锚点重建为全身动作，18.0 mm 点误差（评测子集）；WUJI 手套带区域触觉，但常规录制中仅约 41% 含触觉。"
---

# EgoExplorer：无基站可穿戴采集人类移动操作的视觉–动作–触觉数据集

**EgoExplorer**（*A Vision-Motion-Touch Observatory for Human Loco-Manipulation*，[arXiv:2610.18872](https://arxiv.org/abs/2610.18872)，[项目页](https://internrobotics.github.io/egoexplorer.github.io/)）由上海人工智能实验室、浙江大学与上海交通大学团队提出。它关注的是人走动、调整姿态并同时使用双手操作的 **loco-manipulation**，用一套可穿戴设备在未布置外部基站的真实环境里同步记录多种信号。

## 一句话定义

**用第一视角视频、稀疏身体追踪、手指动作和可选触觉共同记录人类“边走边操作”，再以 C³ 从稀疏 tracker 锚点补全并约束全身动作。**

## 英文缩写速查

| 缩写 | 含义 |
|------|------|
| C³ | Calibrate–Complete–Constrain，个体标定、姿态补全与几何约束组成的重建流程 |
| POV | Point of view，第一视角 |
| VIVE | HTC 的 VIVE 品牌追踪器；本文使用 Ultimate Tracker 的 inside-out 跟踪 |
| WUJI | 论文中提供手指动作与区域触觉信号的手套方案 |

## 数据规模与传感器

| 项目 | 论文 / 官方页面报告 |
|------|--------------------|
| 录制规模 | 论文 v1：2,715 段、100 小时；项目页标注 100+ 小时并持续增长 |
| 覆盖 | 10 类任务模板、5 个环境（实验室、工厂/工作空间、家庭等） |
| 视觉 | GoPro HERO13 头戴相机，固定在头部 tracker 上 |
| 全身锚点 | 5 个 VIVE Ultimate Tracker：头、双腕、双踝；inside-out，不需要外部基站 |
| 手指 | MANUS Metagloves 或 WUJI Glove 直接测量手指动作 |
| 触觉 | 仅 WUJI 配置额外提供区域触觉，项目页展示每手 17 个区域；论文附录中常规录制约 41% 含 WUJI 流 |
| 标注 | 统一时钟对齐的视频、动作、手部/触觉流与步骤级描述 |

> “无基站”来自 VIVE Ultimate Tracker 的 inside-out 跟踪方式；触觉来自 WUJI 手套配置。两者是系统中的不同硬件职责。

## 方法：C³ 稀疏追踪全身重建

**Calibrate–Complete–Constrain（C³）** 将人体与传感器安装几何、未观测关节补全和物理几何约束拆开处理：

- **Calibrate：** 用规定校准动作确定个体几何、身体尺度与 tracker 安装变换，并在传感器位置不变时复用。
- **Complete：** 从头、腕、踝五个稀疏测量锚点出发，用时序先验预测未观测的身体关节。
- **Constrain：** 联合优化 tracker 测量一致性、骨架时序连续性、人体表面自穿透与地面穿透约束。

## 实验与评测

- **全身重建精度：** 论文评测子集的身体点误差为 **18.0 mm**，对照五锚点重训 AvatarPoser 基线的 **64.1 mm**，相对降低 **71.9%**。这是论文特定参考集上的重建评测，不应理解为每帧或每个录制场景都有 18 mm 的独立真值精度。
- **下游探索：** 论文另做了多模态生成与人形全身动作重定向的初步仿真研究，只能作为可行性信号，不等于已验证的真机策略或部署栈。

## 与其他工作对比

| 数据集 | 采集方式 | 身体覆盖 | 触觉 | 与 EgoExplorer 的差异 |
|--------|----------|----------|------|------------------------|
| **EgoExplorer** | 头戴 GoPro + 5 个 inside-out tracker + 手套 | 头/腕/踝锚点经 C³ 重建全身 | 仅 WUJI 子集（约 41%） | — |
| [EgoVerse](./paper-egoverse.md) | 联盟式第一视角人示教生态，1,362 h | 双手 21 关键点 + 头/相机 6-DoF 位姿，无全身骨架 | 未列为模态 | 规模与跨实验室生态更大；EgoExplorer 侧重全身移动 + 手指 + 触觉的同步记录 |
| [Eidon Tracker + POV](./eidon-tracker-pov-dataset.md) | 7-IMU 上肢追踪 + POV 视频，1,273.8 h | 上肢 | 无 | 时长更长但只覆盖上肢；EgoExplorer 补上下肢与行走，适合 loco-manipulation |

## 为什么值得收录

- **补足模态耦合：** 相较只有第一视角视频、或只采手部操作的数据，它将移动、全身姿态、手指动作及可用时的触觉放在同一时间轴上。
- **面向真实工作空间：** 跟踪器不依赖房间级基站/光学动捕布置，适合实验室、家庭和工作现场移动采集。
- **可用于机器人学习研究：** 论文报告了基于该数据的多模态生成与人到人形动作迁移仿真探索；证据仍属初步，不能等同于真机学习结果。

## 开源与访问状态（截至 2026-10-15）

- **数据集：** [Hugging Face](https://huggingface.co/datasets/InternRobotics/EgoExplorer) 需申请并接受条款；当时有 18 段、约 66 分钟的一小时 mini release，涵盖十类任务模板并具备四路数据流。完整数据集需加入 waitlist。
- **许可证：** 数据卡标注 CC BY-NC 4.0；下载和使用前核对当前页面的访问及许可条件。
- **工具：** [官方 GitHub 仓](https://github.com/InternRobotics/EgoExplorer)公开，但仓库 README 称采集/重建代码预计 2026 年 11 月初发布；截至核对时仓内主要是说明和演示素材。

## 结论

**EgoExplorer 的核心价值是把可移动的人类移动操作示范做成多模态同步记录，而不是只采一段手部视频或孤立的身体轨迹。**阅读和使用时要同时记住两个边界：全身动作由稀疏 tracker 经模型重建；区域触觉仅在 WUJI 手套子集中出现。

## 关联页面

- [人形机器人数据采集产业地图](../queries/humanoid-robot-data-collection-landscape.md) — 无机器人可穿戴采集路线
- [Manipulation 任务](../tasks/manipulation.md) — 人类示范与移动操作
- [EgoVerse](./paper-egoverse.md) — 联盟级第一视角人类示范数据
- [Eidon Tracker + POV](./eidon-tracker-pov-dataset.md) — 可穿戴身体追踪与第一视角视频数据

## 参考来源

- [EgoExplorer 论文摘录](../../sources/papers/egoexplorer_arxiv_2610_18872.md)
- [作者项目页](https://internrobotics.github.io/egoexplorer.github.io/)
- [Hugging Face 数据集](https://huggingface.co/datasets/InternRobotics/EgoExplorer)
- [官方工具仓](https://github.com/InternRobotics/EgoExplorer)
