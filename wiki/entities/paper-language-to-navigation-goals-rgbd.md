---
type: entity
tags: [paper, vln, semantic-navigation, ros2, nav2, rgb-d, mobile-robot, turtlebot, unitree-go2, upo]
status: complete
updated: 2026-08-23
arxiv: "2607.13624"
venue: "arXiv 2026"
related:
  - ../tasks/vision-language-navigation.md
  - ../overview/navigation-slam-autonomy-stack.md
  - ./navigation2.md
  - ./autonomy-stack-go2.md
  - ../methods/vla.md
  - ./paper-da-nav.md
  - ./paper-fsd-vln.md
sources:
  - ../../sources/papers/language_to_navigation_goals_arxiv_2607_13624.md
summary: "Language-to-Navigation-Goals（arXiv:2607.13624，UPO）：ROS 2 模块化 VLM+RGB-D→Nav2 语义导航；TurtleBot3 端到端导航误差 0.70 m；Go2 真机定位误差 0.51 m；代码待论文接收后开源。"
---

# Language-to-Navigation-Goals（RGB-D 语义导航）

**From Language to Navigation Goals: A Vision-Language Approach for Semantic Navigation of Mobile Robots Using RGB-D Perception**（[arXiv:2607.13624](https://arxiv.org/abs/2607.13624)）由巴勃罗·德·奥拉维德大学（UPO）团队提出：用 **模块化 ROS 2 管线** 把自然语言导航请求转为 **Nav2 可执行目标**——远程 **VLM** 在 RGB 图上做语义 grounding 返回 bbox，**RGB-D** 几何投影得到地图系三维目标，再由 **Navigation2** 完成路径规划与避障。

## 一句话定义

**把「去冰箱/邮箱」类语言指令落成 Nav2 目标，关键是 VLM 语义检测 + RGB-D 度量投影，而不是端到端学完整导航策略。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLM | Vision-Language Model | 远程服务：联合图文返回 bbox 与自然语言确认 |
| VLN | Vision-and-Language Navigation | 语言条件导航任务族（本文为工程化分层实现） |
| RGB-D | RGB + Depth | 彩色图 + 深度图联合度量定位 |
| Nav2 | Navigation 2 | ROS 2 标准导航栈（全局/局部规划 + 代价地图） |
| ROS 2 | Robot Operating System 2 | 本文通信与模块编排中间件 |
| TF | Transform Frame | ROS 2 坐标变换树（相机→地图） |

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 巴勃罗·德·奥拉维德大学（Universidad Pablo de Olavide, UPO） |
| **出处** | arXiv:2607.13624（v1，2026-07） |
| **仿真** | Gazebo + TurtleBot3 Waffle + ROS 2 + Nav2 |
| **真机** | Unitree Go2 + Intel RealSense RGB-D |
| **VLM** | 远程 VLM 服务（论文未固定单一 checkpoint 名称） |
| **开源** | **待发布**：正文写 *upon acceptance* 开源；截至 2026-08-23 **无官方仓库/项目页** |

## 为什么重要

- **非专家可用**：用户说「去邮箱」而非坐标/路点；系统用自然语言反馈确认理解（如 “I found a mailbox”）。
- **与端到端 VLN 策略互补**：不学习完整导航策略，而是 **语义感知 → 度量目标 → 经典 Nav2**，便于移植到已有 ROS 2 移动平台。
- **上下文指令**：支持 “My father is next to a mailbox…” 类隐含目标解析，不仅限于直接 “go to X”。
- **跨平台验证**：同一框架适配差速 TurtleBot3 与四足 Go2，仅需配置 topic/action 接口。

## 流程总览

```mermaid
flowchart LR
  user["用户自然语言"] --> comm["Communication 模块"]
  robot["机器人 RGB-D"] --> comm
  comm --> vlm["Semantic Perception\n远程 VLM"]
  vlm --> bbox["bbox + 语言确认"]
  bbox --> geom["深度邻域最小值\n+ 针孔投影 + TF"]
  geom --> goal["地图系导航目标"]
  goal --> move["Movement 模块"]
  move --> nav2["Nav2 规划/避障"]
  nav2 --> cmd["cmd_vel"]
  cmd --> robot
  vlm --> comm
  comm --> user
```

## 核心原理

### Semantic Perception（VLM + RGB-D）

1. 将用户指令与机器人 RGB 图发送至 **远程 VLM**，得到目标 **bbox**（\(\mathbf{p}_{tl},\mathbf{p}_{br}\)）与自然语言回复。
2. 取 bbox 中心 \(\mathbf{p}=(p_x,p_y)\)，在深度图邻域 \((2r+1)^2\) 内取 **最小有效深度** \(d\)，降低背景污染。
3. 应用深度偏移 \(d'=d-\delta\)（按目标类别调节停靠距离），针孔模型反投影得相机系 \(\mathbf{P}_c\)。
4. 经 ROS 2 **TF** 变换到全局地图系 \(\mathbf{P}_m\)，作为 Nav2 目标。

### Movement（Nav2）

- 全局/局部代价地图来自 onboard 传感器（LiDAR、深度等）；路径规划与避障 **委托 Nav2**，本框架只负责 **语义目标生成**。

## 评测与结果

### 实验 1：仿真语义感知（公交站，仅感知）

| 指标 | 结果 |
|------|------|
| 语言变体 | 4 种表述 + 4 初始位姿 |
| 平均定位误差 \(e_{goal}\) | **0.68 m**（\(\delta=0.6\) m，相对真值含预期偏移） |

### 实验 2：仿真端到端（人 / 邮箱，Nav2 执行）

| 指标 | 结果 |
|------|------|
| 试验数 | 6（含上下文指令） |
| 平均导航误差 \(e_{nav}\) | **0.70 m** |
| 平均行程 | **6.22 m** |
| 平均耗时 | **26.0 s** |

### 实验 3：真机语义感知（Go2 + RealSense）

| 目标 | 试验 | 定位误差 [m] |
|------|------|-------------|
| 微波炉 | 2 | 0.13, 0.33 |
| 椅子 | 2 | 0.71, 0.87 |
| **平均** | 4 | **0.51 m**（\(\delta=0.5\) m） |

## 与其他工作对比

| 维度 | 本文（ROS 2 + Nav2） | 端到端 VLN 策略 | VLMaps 类语义地图 |
|------|----------------------|-----------------|-------------------|
| 导航执行 | **经典 Nav2** | 学习策略/离散动作 | 地图查询 + 规划器 |
| 语言→目标 | VLM bbox + RGB-D 投影 | 隐式在策略内 | 预建语义地图检索 |
| 平台移植 | 改 ROS topic 即可 | 常需重训/仿真对齐 | 需建图与维护 |
| 多步长程 | 单目标点（未来工作） | Room-to-Room 等基准 | 依赖地图覆盖 |

## 结论

**轻量 ROS 2 框架已证明：远程 VLM 语义 grounding + RGB-D 度量投影足以把日常语言导航请求接到 Nav2，而无需端到端重训导航策略。**

- 仿真端到端平均导航误差约 **0.70 m**，上下文指令可正确解析人/邮箱等目标。
- Go2 真机语义定位平均误差 **0.51 m**，与仿真管线可迁移。
- 模块化设计利于 TurtleBot3 / Go2 等多平台，只适配传感与 cmd_vel 接口。
- 深度偏移 \(\delta\) 需按目标类别手工设定；未来工作提到自适应偏移与多步子任务。
- **代码待论文接收后开源**；当前无法复现完整 ROS 2 包。

## 源码运行时序图

**不适用** — 论文写明接收后开源，截至 **2026-08-23** arXiv 无 GitHub/项目页。工程对照可读 [Navigation2](./navigation2.md) 与 ROS 2 Nav2 文档，但 **本文 VLM 通信与感知模块无公开入口**。

## 与其他页面的关系

- [vision-language-navigation](../tasks/vision-language-navigation.md) — VLN 任务族与基准语境
- [navigation-slam-autonomy-stack](../overview/navigation-slam-autonomy-stack.md) — Nav2 在移动机器人栈中的位置
- [Navigation2](./navigation2.md) — 本文 Movement 模块依赖的导航栈
- [Unitree Go2](./autonomy-stack-go2.md) — 真机验证平台
- [DA-Nav](./paper-da-nav.md) — 另一类语言→导航工程路线（城市户外 VLM）

## 参考来源

- [language_to_navigation_goals_arxiv_2607_13624](../../sources/papers/language_to_navigation_goals_arxiv_2607_13624.md)

## 推荐继续阅读

- [arXiv:2607.13624](https://arxiv.org/abs/2607.13624)
- [Nav2 官方文档](https://docs.nav2.org/)
- [VLN 开源复现四范式](../overview/vln-open-source-repro-paradigms.md)
