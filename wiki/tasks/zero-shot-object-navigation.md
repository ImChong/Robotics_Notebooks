---
type: task
tags: [task, objectnav, zero-shot, navigation, embodied-ai, open-vocabulary, habitat, vln]
status: complete
updated: 2026-10-23
related:
  - ./vision-language-navigation.md
  - ../entities/paper-travexplorer.md
  - ../entities/paper-sa-2608-13723-graph-mambanav.md
  - ../entities/paper-zonda.md
  - ../entities/paper-uni-lavira.md
  - ../entities/paper-sam3.md
  - ../entities/paper-dart-sam3-realtime.md
  - ../entities/paper-blip2.md
  - ../entities/paper-rpv-semnav.md
  - ../entities/paper-wildos-open-vocabulary-object-search.md
  - ../entities/paper-mocap-mobile-code-as-policies.md
  - ../entities/habitat-sim.md
  - ../concepts/embodied-semantic-cognitive-map.md
  - ../concepts/vision-language-feature-fusion.md
  - ../concepts/sim2real.md
  - ../overview/quadruped-vln-embodied-workshop.md
sources:
  - ../../sources/courses/quadruped_vln_embodied_workshop_2day.md
  - ../../sources/papers/travexplorer_arxiv_2605_19958.md
  - ../../sources/papers/zonda_arxiv_2607_21025.md
  - ../../sources/papers/rpv_semnav_arxiv_2607_25448.md
  - ../../sources/papers/graph_mambanav_arxiv_2608_13723.md
  - ../../sources/papers/mocap_arxiv_2610_20998.md
summary: "零样本目标导航（ZSON / ObjectNav）：在未见环境中按开放词汇物体目标探索并到达，无需该目标类别的导航微调；常见 Habitat 评测与四足/轮腿真机部署。"
---

# 零样本目标导航（Zero-Shot Object Navigation）

**零样本目标导航**（Zero-Shot Object Navigation，**ZSON**；基准任务常称 **ObjectNav / Object-Goal Navigation**）要求智能体在 **未见地图** 的环境中，根据 **物体类别或开放词汇描述** 探索并导航至目标实例附近，且 **不在该目标导航轨迹上做任务微调**（感知模型可预训练）。

## 一句话定义

**没来过这栋楼、也没为「找椅子」单独训导航策略，只凭物体名字（或短语）自己建图、探索并走到目标跟前。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ZSON | Zero-Shot Object Navigation | 零样本目标导航设定 |
| ObjectNav | Object-Goal Navigation | 物体目标导航任务名 |
| SR | Success Rate | 成功率 |
| SPL | Success weighted by Path Length | 路径效率加权成功 |
| HM3D / MP3D | Habitat-Matterport 3D / Matterport3D | 常用室内基准 |
| VLM | Vision-Language Model | 开放词汇语义与核验 |
| FOV | Field of View | 视野限制驱动主动感知 |

## 为什么重要

- **服务机器人刚需：** 「去把杯子拿来」的前置是找到杯子；相对坐标点导航更贴近语言接口。
- **课程项目锚点：** 实战营 Day2 下午项目即为 **零样本目标导航的仿真 + 四足真机**。
- **与 VLN 分工：** [VLN](./vision-language-navigation.md) 强调逐步语言指令路径；ObjectNav 强调 **目标物体** 与探索效率。

## 任务要素

| 要素 | 说明 |
|------|------|
| 输入 | 目标类别/短语 + egocentric RGB-D（真机可加 LiDAR） |
| 输出 | 到达目标附近并 STOP（或报告失败） |
| 环境 | 室内多房间；进阶含 **多楼层 / 动态行人** |
| 零样本约束 | 不在目标导航专家轨迹上微调端到端策略；可用预训练检测/VLM |
| 指标 | SR、SPL、路径长度、软成功距离阈值 |

## 主流方法骨架

```mermaid
flowchart TB
  GOAL["语言目标"] --> SEM["开放词汇感知<br/>SAM3 · GDINO · VLM"]
  OBS["RGB-D / LiDAR"] --> MAP["几何 + 语义认知地图"]
  SEM --> MAP
  MAP --> EXP["前沿 / 价值探索"]
  EXP --> PLAN["全局路径 + 局部规划"]
  PLAN --> ACT["移动执行"]
  ACT --> VER["多视角核验"]
  VER -->|确认| STOP["成功 STOP"]
  VER -->|拒绝| EXP
```

| 模块 | 常见实现 |
|------|----------|
| 实例提案 | [SAM 3](../entities/paper-sam3.md)（**SAM 3.1** 适合视频 多实例 PCS）、[DART](../entities/paper-dart-sam3-realtime.md)（多类实时框）、[ENEAS](../entities/paper-eneas.md)、Grounding DINO、OWLv2 |
| 语言对齐 | [BLIP-2](../entities/paper-blip2.md)、CLIP、现代 VLM 嵌入 |
| 地图 | 占据 + 语义热力 + 实例记忆（[语义认知地图](../concepts/embodied-semantic-cognitive-map.md)） |
| 跨楼层 | [TravExplorer](../entities/paper-travexplorer.md) 可通行 3D；[ZONDA](../entities/paper-zonda.md) 高度差启发式 |
| 仿真 | [Habitat](../entities/habitat-sim.md) HM3D/MP3D |
| 训练-free 语义 frontier | [RPV-SemNav](../entities/paper-rpv-semnav.md)（room-mediated RPV + geodesic value map） |
| 户外长程搜索 | [WildOS](../entities/paper-wildos-open-vocabulary-object-search.md)：用视觉 frontier / 可通行性评分稀疏几何图，并融合多视角线索定位远处目标 |
| 面向操作的末段站位 | [MoCaP](../entities/paper-mocap-mobile-code-as-policies.md)：用人类视频的空间记忆找物，以接近方向先验引导底盘位姿搜索，再检查机器人自身的可达性与碰撞约束 |
| 真机 | 四足 Go2、轮腿双足等 + MPPI/局部规划 |

- **图关系 + 长时序方法：** [Graph-MambaNav](../entities/paper-sa-2608-13723-graph-mambanav.md) 在 ObjectNav 对象图里用目标关系先验控制 Graph-Mamba 扫描顺序，并以逐物体 Mamba 记忆补长程上下文；AI2-THOR / RoboTHOR 有仿真评测，但真机证据目前是轮式底盘找书演示。

## 工程实践

| 项 | 建议 |
|----|------|
| 先仿真 | Habitat ObjectNav 跑通探索–核验闭环，再迁真机 |
| 假阳性 | 必须多视角/几何一致性核验，勿见类名即 STOP |
| Sim2Real | 离散仿真动作 → 真机连续跟踪；见 [sim2real](../concepts/sim2real.md) |
| 算力 | 重 VLM 离板；机载跑 LIO + 轻量检测（[Orin NX](../entities/jetson-orin-nx.md)） |

## 评价指标与读法

- **SR 高但 SPL 低：** 会找但乱逛——探索策略差。
- **单楼层强、多楼层崩：** 缺楼梯/可通行 3D 表示。
- **静态强、动态崩：** 缺行人预测与局部避障。

### 从找到目标到可操作站位

[MoCaP](../entities/paper-mocap-mobile-code-as-policies.md) 将目标搜索与后续移动操作连接起来：先从被动人类视频提取空间记忆和接近物体的方向，再结合当前观测搜索底盘位姿，通过抓取候选、逆运动学与全臂碰撞检查确认机械臂能否执行。人类站位只提供搜索先验，不能直接替代机器人自身的几何约束。

- **分开报告导航与操作结果：** 论文在单个室内真机设置中报告，视频空间记忆使目标搜索成功率从 7/10 提升至 10/10；两类移动操作任务的成功率分别为 16/20 和 8/10。两组指标回答不同问题，不能把导航成功率当作抓取成功率。
- **保留实验边界：** 来源记录的接近方向实验汇总为 38/45 对照 18/45，但作者指出该汇总差异在当前样本规模下不显著；不可据此推断在任意场景都能稳定找到可抓取站位。
- **明确任务边界：** MoCaP 是移动操作框架，可作为 ObjectNav 之后的操作衔接案例；其结果不能直接与 Habitat HM3D/MP3D 上的 SR、SPL 排名横向比较。

## 局限与风险

- **开放词汇 ≠ 开放世界鲁棒：** 稀有物体、重度遮挡仍难。
- **成功半径争议：** 3 m SR 可能未保证可见/可交互姿态（对照 REALM 对 VLN 末段的批评）。
- **开源参差：** TravExplorer / ZONDA 等方法页需核对代码是否真正可跑。

## 关联页面

- [视觉–语言导航](./vision-language-navigation.md)
- [TravExplorer](../entities/paper-travexplorer.md)
- [ZONDA](../entities/paper-zonda.md)
- [Uni-LaViRA](../entities/paper-uni-lavira.md)
- [WildOS：户外开放词汇目标搜索](../entities/paper-wildos-open-vocabulary-object-search.md)
- [MoCaP：从人类视频学习找物与可操作站位](../entities/paper-mocap-mobile-code-as-policies.md)
- [四足×VLN 实战营总览](../overview/quadruped-vln-embodied-workshop.md)

## 参考来源

- [四足×VLN 实战营课程大纲](../../sources/courses/quadruped_vln_embodied_workshop_2day.md)
- [TravExplorer 论文摘录](../../sources/papers/travexplorer_arxiv_2605_19958.md)
- [ZONDA 论文摘录](../../sources/papers/zonda_arxiv_2607_21025.md)
- [MoCaP 论文摘录](../../sources/papers/mocap_arxiv_2610_20998.md)

## 推荐继续阅读

- [VLN 开源复现四范式](../overview/vln-open-source-repro-paradigms.md)
- Habitat 文档：<https://aihabitat.org/>
