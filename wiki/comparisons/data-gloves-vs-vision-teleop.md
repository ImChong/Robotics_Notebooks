---
type: comparison
tags: [dexterity, data-collection, teleoperation, hardware, sensors]
status: complete
updated: 2026-11-15
related:
  - ../queries/dexterous-data-collection-guide.md
  - ../entities/paper-nestdex.md
  - ../entities/paper-spd.md
  - ../entities/allegro-hand.md
  - ../entities/xyz-deux.md
  - ../entities/twindex.md
  - ../entities/humantouch.md
  - ../entities/paper-notebook-osmo-open-source-tactile-glove-for-human-to-robo.md
  - ../methods/behavior-cloning.md
sources:
  - ../../sources/papers/imitation_learning.md
  - ../../sources/sites/xyzcorp-deux.md
  - ../../sources/sites/x2robot-twindex.md
  - ../../sources/sites/humantouch-xsparkai.md
  - ../../sources/papers/nestdex_arxiv_2608_13362.md
original:
  - ../../sources/papers/imitation_learning.md
summary: "灵巧操作数据采集选型：对比了穿戴式数据手套与基于视觉的遥操作方案，涵盖了精度、成本、遮挡鲁棒性及力反馈等关键维度。"
---

# 数据手套 vs 视觉遥操作 (灵巧数据采集选型)

在训练灵巧手（Dexterous Hand）执行复杂任务时，获取高质量的人类演示数据是第一步。目前，**穿戴式数据手套 (Data Gloves)** 和 **基于视觉的遥操作 (Vision-based Teleop)** 是两条最主流的技术路线。

## 一句话定义

数据手套靠接触式传感（弯曲传感器 / IMU）换来精度与抗遮挡，视觉遥操作靠非接触相机换来低成本与自然动作；二者是灵巧手示教数据采集的两条主流路线，选型取决于任务接触强度、预算与数据一致性要求。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Teleop | Teleoperation | 人遥操作机器人采集演示数据 |
| IMU | Inertial Measurement Unit | 惯性测量单元，提供加速度与角速度 |
| RGB | Red-Green-Blue | 彩色图像通道，常与深度 (RGB-D) 配合 |
| RL | Reinforcement Learning | 通过与环境交互最大化长期回报来学习策略的范式 |
| Retargeting | Motion Retargeting | 将人体/动物动作映射到目标机器人骨架 |

## 为什么重要

灵巧手策略（BC / RL 微调）的上限由人类演示数据的质量与规模决定。采集通道一旦选定，会同时锁定数据精度、可规模化程度、是否带力 / 触觉模态以及重定向误差来源，后期很难补救，因此需要在项目初期做选型。

## 核心原理（选型维度）

| 维度 | 穿戴式数据手套 | 基于视觉的遥操作 (如 Meta Quest/Leap) |
|------|--------------|---------------------------------|
| **测量原理** | 物理接触（弯曲传感器/IMU） | 非接触式光电（RGB/深度相机） |
| **精度与稳定性** | ✅ 极高且稳定，无漂移 | ⚠️ 中等，受光照和背景干扰，存在漂移 |
| **遮挡鲁棒性** | ✅ 免疫遮挡，手放在背后也能测 | ❌ 极差，手指交叠或被物体遮挡时失效 |
| **力反馈 (Haptics)** | ✅ 部分高级型号支持主动力反馈 | ❌ 几乎没有，只能靠视觉观察 |
| **操作自由度** | ⚠️ 受限，手部有异物感，容易疲劳 | ✅ 极高，裸手操作，自然丝滑 |
| **成本** | ❌ 昂贵 (5k - 50k USD) | ✅ 廉价 (100 - 500 USD) |
| **校准复杂度** | 繁琐，每个人的手部尺寸需精确适配 | 简单，通过算法自动重定向 |

## 适用场景分析

### 推荐使用数据手套的场景：
1. **接触极其频繁的操作**：例如在口袋里摸索硬币、盲拧螺丝。这类任务视觉无法介入，且需要手套提供极高频的关节序列。
2. **高保真力控训练**：当你的 RL 奖励函数对手指捏力非常敏感时，带有力反馈的数据手套（如 SenseGlove）是唯一的选择。
3. **长期、大规模工程采集**：对数据一致性（Consistency）要求极高，不能容忍由于摄像头位置变动导致的轨迹偏移。
4. **与特定灵巧手 1:1 绑定的商业采数**：如 [DEUX / Glove X](../entities/xyz-deux.md)（XYZ，2026）用接触点对齐宣称 **zero-shot retarget、免后处理**——换来的是 **embodiment 锁定** 与闭源数据飞轮，适合门店一体机而非跨臂开源数据集。自变量 [TwinDEX](../entities/twindex.md)（2026-09）走同一锁定，但是 **外骨骼–机械手五维共设计**（运动学/接触/外观/精度/同步），并主张 **零真机遥操作数据** 即可部署；同样未开源。

### 推荐使用视觉遥操作的场景：
1. **低预算科研项目**：Meta Quest 配合重定向算法（Retargeting）可以快速搭建起一套可用的 Demo 采集系统。
2. **需要极致自然动作的模仿**：人类在裸手状态下展现出的灵巧度远超戴着沉重手套的状态，采集到的轨迹更符合“专家”定义。
3. **视觉引导的常规操作**：如折衣服、整理桌面、倒水。这类任务遮挡时间短，算法可以通过时序滤波进行平滑。

## 融合趋势：视触觉混合采集

目前顶尖的实验室（如斯坦福 ALOHA 团队）开始探索将两者结合：使用 Quest 进行宏观位姿追踪，而在指尖安装 GelSight 传感器采集微观接触特征，从而兼顾成本、灵巧度与数据精度。规模化人侧全掌压阻路线见 [HumanTouch](../entities/humantouch.md)（MANUS EMF 手姿 + 头/腕 RGB + 手套质控；ModelScope 已发布首批约 100 h 数据）；人机共用开源磁触觉见 [OSMO](../entities/paper-notebook-osmo-open-source-tactile-glove-for-human-to-robo.md)。

第三条通道是 **copilot / 共享自治采数**：操作员不再直接编全部手指关节。[NestDex](../entities/paper-nestdex.md) 用 1-DoF clutch 调节学到的内层手技能，人只管臂与进度；完整示范再训独立外层策略。它不替代手套或视觉重定向（内层技能仍靠多视重定向冷启动），但把「整任务手指协调」从操作员认知负荷里卸掉。

## 工程实践

- **预算有限先走视觉**：Quest 等头显裸手追踪加重定向即可搭起 Demo 采集，再按瓶颈（遮挡、漂移）决定是否补手套。
- **先量化再选型**：用同一批任务各采少量数据，对比重定向后指尖误差与轨迹成功率，而不是只看硬件标称精度。
- **手套类需逐人校准**：手部尺寸适配与校准是主要工程成本；长期采集要固定校准流程以保持一致性。
- **混合与 copilot 通道**：宏观位姿用视觉、微观接触用触觉传感器，或用 clutch 降低操作员手指认知负荷（见 NestDex 段）。

## 局限与风险

- 本页对比维度（精度、成本区间等）为归纳性定性描述，价格与精度数字随产品迭代变化，选型前需以厂商规格或实测为准。
- 商业手套方案（DEUX、TwinDEX）存在 embodiment 锁定与闭源数据飞轮，跨本体复用困难。
- 视觉方案在遮挡、光照变化下失效；手套方案手部有异物感、易疲劳，且不同人手尺寸需重新校准。

## 关联页面
- [灵巧操作数据采集指南](../queries/dexterous-data-collection-guide.md)
- [Allegro Hand 实体](../entities/allegro-hand.md)
- [DEUX / Glove X（XYZ）](../entities/xyz-deux.md) — 商业三指手 + 1:1 手套零样本重定向样本
- [TwinDEX（自变量）](../entities/twindex.md) — 三指外骨骼–同构手共设计；纯 robot-free（闭源）
- [HumanTouch](../entities/humantouch.md) — 全掌压阻 + EMF 手姿多模态人侧采数
- [OSMO 触觉手套](../entities/paper-notebook-osmo-open-source-tactile-glove-for-human-to-robo.md) — 人机共用开源磁触觉
- [NestDex](../entities/paper-nestdex.md) — copilot 嵌套采数：人控臂 + clutch，不直接编 20-DoF 手指
- [SPD](../entities/paper-spd.md) — 仿真用头显手跟踪、真机改 Manus；同一目标本体两条通道
- [Behavior Cloning (行为克隆)](../methods/behavior-cloning.md)

## 参考来源
- [sources/papers/imitation_learning.md](../../sources/papers/imitation_learning.md)
- [xyzcorp-deux.md](../../sources/sites/xyzcorp-deux.md) — Glove X 产品规格与开源核查
- [x2robot-twindex.md](../../sources/sites/x2robot-twindex.md) — TwinDEX 项目页开源核查与共设计摘录
- [nestdex_arxiv_2608_13362.md](../../sources/papers/nestdex_arxiv_2608_13362.md) — NestDex copilot 作为第三条采数通道
- [SPD 论文归档](../../sources/papers/spd_corl_2026.md) — 仿真头显手跟踪 vs 真机 Manus
- [humantouch-xsparkai.md](../../sources/sites/humantouch-xsparkai.md) — HumanTouch 项目页
- Qin, B., et al. (2023). *AnyTeleop Framework*.

## 推荐继续阅读

- [具身数据纵深路线](../../roadmap/depth-embodied-data.md)
- [灵巧操作数据采集指南](../queries/dexterous-data-collection-guide.md)
- [Meta Quest 遥操作](../entities/oculust-quest-teleop.md)
- [NestDex](../entities/paper-nestdex.md)
