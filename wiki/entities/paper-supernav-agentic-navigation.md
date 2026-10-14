---
type: entity
tags: [paper, navigation, vln, objectnav, pointnav, agentic, embodied-ai, mobile-robot, unitree-go2, zju, szu]
status: complete
updated: 2026-10-14
arxiv: "2610.12126"
project_id: supernav-agentic-navigation-any-task-any-scene
project: https://zju3dv.github.io/SuperNav/
related:
  - ../tasks/vision-language-navigation.md
  - ./qwen-robot-nav.md
  - ./paper-uni-lavira.md
  - ./paper-navverse.md
  - ./paper-navwam-goal-conditioned-visual-navigation-wam.md
sources:
  - ../../sources/papers/supernav_arxiv_2610_12126.md
  - ../../sources/sites/zju3dv-supernav.md
  - ../../sources/repos/zju3dv-supernav-code.md
summary: "SuperNav（arXiv:2610.12126）给预训练 MLLM 配置 agent harness，以视觉点目标连接高层决策和可替换运动后端；用 Tools、Skills、任务进度与上下文管理支持多任务导航，不做导航专用微调。"
---

# SuperNav：面向任意任务与场景的 Agentic 导航系统

**SuperNav**（*An Agentic Navigation System for Any Task in Any Scene*，arXiv:[2610.12126](https://arxiv.org/abs/2610.12126)，[项目页](https://zju3dv.github.io/SuperNav/)，[代码](https://github.com/zju3dv/SuperNav)）将预训练 MLLM 用作导航任务的语义理解与高层决策器，通过 agent harness 管理持续交互，再把视觉目标点交给运动执行器。作者不对 MLLM 做导航专用微调，重点研究的是工具接口、运行循环、任务状态和 Skills 如何帮助它把通用能力用于具身导航。

## 一句话定义

**MLLM 决定“该看哪里、接下来去哪里、是否完成”，harness 维护任务与上下文，执行器负责把图像上的目标点变成运动。**

## 核心术语

| 术语 | 含义 |
|------|------|
| Agent harness | 为模型组织工具调用、上下文和任务状态的运行层 |
| Visual-point interface | 模型在观测图像中选择方向视图和归一化像素点，作为移动目标 |
| Navigation Skills | 可按需读取的过程指南，覆盖搜索、核验、恢复 |
| Geo-based Executor | 将像素目标结合深度、相机标定和位姿映射为几何目标并规划路径 |
| Learned Executor | 根据目标点和近期 RGB 历史预测局部位移增量与停止信号 |
| SR / SPL | 成功率 / 按路径长度加权的成功率 |

## 系统结构

### 1. Agent loop：维持长程任务状态

Agent 接收用户请求与四向 RGB 图像后，MLLM 反复选择观测、转向、点目标导航、更新任务进度或终止会话。工具返回的新图像和执行诊断回到上下文，供下一轮推理使用。结构化目标记录帮助模型区分已完成与待办项目；文本历史保留已搜索区域、候选线索和失败尝试。

终止工具只表示模型已报告任务结果并关闭会话，benchmark 会独立判定任务是否成功。因此，长时程任务的状态跟踪和“完成”判断属于评测链路中的独立风险点。

### 2. 视觉点导航：把语言决策接到运动

一次观测包含相对当前朝向的前、右、后、左四幅 RGB 图。模型选择一个视图及归一化坐标 `(u,v)∈[0,1]^2`，执行器尝试到达该像素点对应的可通行位置，并返回新图像、执行状态和简短诊断。模型可重新观察、选择别的目标点并恢复，因此高层导航循环与底层执行后端保持松耦合。

### 3. 互换式运动后端

- **几何执行器**：用深度与相机标定把像素目标投到环境几何，再依据定位和可导航结构规划路径。
- **学习执行器**：基于目标标记和近期 RGB 观测预测局部位移增量与停止信号，在短动作段之间反复重规划。

两类后端沿用同一视觉点交互模式；运动方式不同，但反馈都回到相同的 MLLM 决策循环。

### 4. Tools 与 Skills

工具覆盖启动、观察、转向、视觉点导航、目标进度维护与结束。Skills 以 Markdown 说明何时搜索、怎样复核目标、怎样从阻塞中恢复。Skills 进入模型上下文后影响选择，但不强制工具顺序，也不提供新的物理执行能力。

## 评测结果

| 基准 | SuperNav 结果 | 口径 |
|------|---------------|------|
| Habitat-GS 单物体 | SR 78.00%，SPL 0.4127 | 150 项任务 |
| Habitat-GS 有序多物体 | SR 34.00%，SPL 0.1388 | 150 项任务 |
| AI2-THOR 需求驱动 | SR 59.50%，SPL 0.1801 | 200 项任务 |
| HM3D-OVON 未见验证 | SR 68.33% / 73.33%；SPL 0.3837 / 0.4105 | 0.25 m / 1 m 成功阈值 |
| HM3Dv2 验证 | SR 80.30% / 86.50%；SPL 0.3338 / 0.3638 | 0.2 m / 1 m 成功阈值 |

不同基准的任务构成、成功阈值与协议不同，数字只适合在同一行对应的设置内理解。论文和项目页也展示了 Unitree Go2 实机视频，但不能据此推断仓库已有可复现真机部署栈。

## 开源与复现状态

截至 **2026-10-14**，[官方仓库](https://github.com/zju3dv/SuperNav)提供 agent runtime、导航 Tools 与 Skills、仿真实验配置和评测工具，并公开了 Learned Executor 权重和 Habitat-GS 测试任务。官方 README 将 AI2-THOR 需求驱动数据与 Unitree Go2 真机部署列为待发布；当前仓库尚未包含真机驱动。运行仿真还需自行准备场景数据、模型访问与相应服务配置。

## 阅读判断

- **贡献在 harness 与接口**：不是把某个导航动作头做得更大，而是让通用 MLLM 通过视觉目标点、执行反馈和任务记录持续完成不同请求。
- **运动后端可替换**：几何规划更依赖深度/地图等几何输入；学习执行器更依赖训练数据与权重，二者不应被误认为同一控制器。
- **结果有跨任务证据，但边界需按基准读**：单物体、多物体、需求驱动及 HM3D 类别导航协议不同；SR 不宜混成一个总体指标。
- **真机演示与复现入口分开**：页面有 Go2 演示，代码仓库仍标注真机部署待发布。

## Mermaid：从任务请求到闭环导航

```mermaid
flowchart TD
  request["自然语言任务"] --> mllm["预训练 MLLM：理解与选点"]
  obs["四向 RGB 观测"] --> mllm
  mllm --> harness["Agent loop：上下文与任务进度"]
  harness --> tools["导航 Tools：观察、转向、PointNav"]
  tools --> backend["几何执行器 / 学习执行器"]
  backend --> feedback["执行反馈与新观测"]
  feedback --> mllm
  skills["Navigation Skills：搜索、核验、恢复指南"] --> mllm
```

## 参考来源

- [arXiv v1 原文](https://arxiv.org/html/2610.12126v1)
- [SuperNav 官方项目页](https://zju3dv.github.io/SuperNav/)
- [官方实现](https://github.com/zju3dv/SuperNav)
- [SuperNav 论文摘录](../../sources/papers/supernav_arxiv_2610_12126.md)
- [项目页归档](../../sources/sites/zju3dv-supernav.md)
- [仓库归档](../../sources/repos/zju3dv-supernav-code.md)

## 关联页面

- [视觉–语言导航](../tasks/vision-language-navigation.md)
- [Qwen-RobotNav](./qwen-robot-nav.md) — 多任务导航模型及可控观测协议。
- [Uni-LaViRA](./paper-uni-lavira.md) — 零样本统一导航的分层翻译与多本体方案。
- [NavVerse](./paper-navverse.md) — 场景与跨区域具身导航基准。
- [NavWAM](./paper-navwam-goal-conditioned-visual-navigation-wam.md) — 目标图像条件的导航世界动作模型。
