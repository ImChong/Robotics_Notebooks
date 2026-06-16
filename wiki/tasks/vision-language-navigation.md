---
type: task
tags: [vln, navigation, embodied-ai, vision-language, matterport]
summary: "视觉–语言导航（VLN）要求智能体在三维环境中依据自然语言指令执行一系列离散或连续动作到达目标，是连接语言理解与空间运动规划的基准任务。"
updated: 2026-06-07
status: complete
related:
  - ../overview/vln-open-source-repro-paradigms.md
  - ../entities/sceneverse-pp.md
  - ../entities/esi-bench.md
  - ../entities/paper-worldvln-aerial-vln-wam.md
  - ../concepts/3d-spatial-vqa.md
  - ../concepts/world-action-models.md
  - ../methods/vla.md
  - ../entities/paper-homeworld-whole-home-scene-generation.md
  - locomotion.md
sources:
  - ../../sources/blogs/wechat_shenlan_vln_repro_four_paradigms_2026.md
  - ../../sources/repos/sceneverse-pp.md
  - ../../sources/papers/worldvln_arxiv_2605_15964.md
---

# 视觉–语言导航（Vision-and-Language Navigation, VLN）

**VLN**：智能体接收 **自然语言导航指令** 与 **第一人称（egocentric）视觉观测**（渲染视图或真实相机图像），在离散或连续动作空间中决策，最终到达指令描述的目标位置或物体。**语言–视觉接地** 与 **路径效率** 是核心评价维度。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| VLM | Vision-Language Model | 视觉-语言多模态理解模型，VLA 的上游 |
| WAM | World Action Model | 联合世界模型与动作预测的架构 |
| LLM | Large Language Model | 大语言模型，常作高层任务/语言接口 |
| Locomotion | Robot Locomotion | 足式/人形等无轮移动能力的总称 |

## 为什么重要？

- **机器人场景**：家庭或服务机器人需要理解「穿过客厅，在冰箱左侧停下」这类指令；VLN 提供了可复现的 **语言–几何–动作** 闭环基准。
- **与纯导航的区别**：传统导航多依赖地图与坐标目标；VLN 强调 **语义描述**（地标、相对运动），更贴近人类口头指路。
- **与 VLA 的衔接**：高层策略可将 VLN 视作「语言条件下的路径生成」子问题；仿真基准（如 Matterport3D 上的 **R2R / RxR**）与真实视频蒸馏数据（如室内 tour）常混合使用以缓解 **sim–real** 与 **轨迹分布** 差异。通才 VLA 如 [Qwen-VLA](../entities/qwen-vla.md) 在官方 README 中把 **操作与 VLN 基准** 放进 **同一 checkpoint** 联合评测，可作为「导航是否应并入统一 VLA」的工程参照。
- **Agentic 导航基座**：[Qwen-RobotNav](../entities/qwen-robot-nav.md) 以 **可控观测协议 + 任务 mode** 统一 VLN / ObjNav / 跟踪 / NAVSIM 驾驶，并作为 **Qwen3.7-Plus** 等 planner 的导航原语；与 [Qwen-Robot Suite](../entities/qwen-robot-suite.md) 长时程 **EQA / 开放世界寻物** demo 一并阅读。

## 核心要素

| 要素 | 说明 |
|------|------|
| 环境 | 常用大规模室内扫描数据集（如 Matterport3D）构建可导航网格 |
| 观测 | 全景图序列或 pinhole 渲染视图；近年也引入真实行走视频 |
| 动作 | 常见为离散前向/转向步长；需与数据集标注一致 |
| 监督 | 专家轨迹模仿、强化学习、或从网页视频重建的伪轨迹 + VLM 生成指令 |

**分布差异**：仿真中最短路径、朝前行走居多；真实 Room-tour 视频存在停顿、回头与冗余旋转，直接用作监督需要 **轨迹清洗与动作离散化**（SceneVerse++ 论文中描述了面向 R2R 的三阶段管线）。

### 空中 / UAV 子域

- **设定差异：** [WorldVLN](../entities/paper-worldvln-aerial-vln-wam.md) 等 **空中 VLN** 工作在 **连续 3D 航点** 与 **大视角 egocentric 变化** 下闭环执行语言指令；相对 Matterport 离散转向，更强调 **因果记忆、短视界世界预测与真机迁移**。
- **范式对照：** 地面开源栈见 [四范式复现路径](../overview/vln-open-source-repro-paradigms.md)；空中路线可将 **自回归 World Action Model** 与 **导航 VLA** 对照阅读（[WAM 概念页](../concepts/world-action-models.md)）。

## 常见误区

- **误区**：「VLN 做得好就等于机器人能走。」仿真离散动作与真实连续控制、动力学约束仍有鸿沟，通常需要低层控制与碰撞规避模块。
- **误区**：「只用仿真轨迹训练就能覆盖真实室内。」真实视频的引入（含自动指令生成）是为了丰富 **语言风格与行走模式**，但仍需评估在标准基准上的可迁移性。

## 与其他页面的关系

- **复现路径**：[VLN 四范式开源复现策展](../overview/vln-open-source-repro-paradigms.md) — VLFM / NavGPT / NoMaD / Uni-NaVid 由浅入深（模块化→LLM→扩散 e2e→导航 VLA）。
- **空中 WAM**：[WorldVLN](../entities/paper-worldvln-aerial-vln-wam.md) — 潜自回归世界转移 + 航点解码 + Action-aware GRPO；室内外 UAV 基准与真机部署（arXiv:2605.15964）。
- **数据**：[SceneVerse++](../entities/sceneverse-pp.md) 将室内漫游视频转为 R2R 兼容的离散导航数据，并报告在相关基准上的增益。
- **全屋仿真场景**：[HomeWorld](../entities/paper-homeworld-whole-home-scene-generation.md) 从文本生成 **sim-ready 多房间家居**（300K **Chinese Style** 矢量平面图 + 5K furnished 全屋 3D 待开源），面向 **跨房间语言导航与家务** 的 **户型本地化** 数据链——与 Matterport 系扫描 benchmark 互补而非直接替代。
- **空间推理**：[3D 空间 VQA](../concepts/3d-spatial-vqa.md) 侧重问答；VLN 侧重 **时序决策**，二者常共享场景表示与 VLM 骨干。[ESI-Bench](../entities/esi-bench.md) 则在 OmniGibson 上评测 **为看见而行动** 的细粒度空间 QA，与 VLN 的 **轨迹到达** 目标互补。
- **运动基础**：[Locomotion](locomotion.md) 提供低层移动能力；VLN 更多占据 **任务规划与语义接地** 层，可与 VLA 分层结合。
- **模型**：[VLA](../methods/vla.md) 可作为统一骨架，在导航子任务上接入离散动作头或目标点输出。

## 参考来源

- [深蓝具身智能：VLN 四范式新手复现推荐](../../sources/blogs/wechat_shenlan_vln_repro_four_paradigms_2026.md) — Habitat/R2R 可跑通开源栈策展
- [WorldVLN 论文摘录（arXiv:2605.15964）](../../sources/papers/worldvln_arxiv_2605_15964.md) — 空中 VLN · 自回归 WAM
- [SceneVerse++ 原始资料归档](../../sources/repos/sceneverse-pp.md)
- Chen et al., *Lifting Unlabeled Internet-level Data for 3D Scene Understanding* (arXiv:2604.01907) — VLN 数据生成与 R2R 实验
- Anderson et al., *Vision-and-Language Navigation* — R2R 任务经典定义（如需溯源基准起源可查阅原文）

## 关联页面

- [VLN 开源复现：四范式学习路径](../overview/vln-open-source-repro-paradigms.md)
- [WorldVLN（空中 VLN · WAM）](../entities/paper-worldvln-aerial-vln-wam.md)
- [World Action Models（WAM）](../concepts/world-action-models.md)
- [SceneVerse++](../entities/sceneverse-pp.md)
- [HomeWorld](../entities/paper-homeworld-whole-home-scene-generation.md) — 文本到 sim-ready 全屋 3D 与中国住宅平面图数据
- [3D 空间 VQA](../concepts/3d-spatial-vqa.md)
- [Locomotion](locomotion.md)
- [VLA](../methods/vla.md)

## 推荐继续阅读

- [机器人论文阅读笔记：MolmoSpaces](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/MolmoSpaces__A_Large-Scale_Open_Ecosystem_for_Robot_Navigation_and_Manipulation/MolmoSpaces__A_Large-Scale_Open_Ecosystem_for_Robot_Navigation_and_Manipulation.html)
- [机器人论文阅读笔记：Thinking in 360°](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/Thinking_in_360__Humanoid_Visual_Search_in_the_Wild/Thinking_in_360__Humanoid_Visual_Search_in_the_Wild.html)
- [机器人论文阅读笔记：STATE-NAV](https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/08_Navigation/STATE-NAV__Stability-Aware_Traversability_Estimation_for_Bipedal_Navigation_on_Rough_Terrain/STATE-NAV__Stability-Aware_Traversability_Estimation_for_Bipedal_Navigation_on_Rough_Terrain.html)
- Matterport3D / R2R、RxR 等官方基准说明
- NaVILA、RoomTour3D 等「真实视频 + 导航」相关工作（与互联网视频蒸馏路线对照）
