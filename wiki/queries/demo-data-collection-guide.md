---
type: query
tags: [imitation-learning, teleoperation, data-collection, manipulation, behavior-cloning]
status: complete
summary: 用模仿学习做操作任务时，高效收集高质量人类演示数据的完整指南，覆盖硬件选型、数据质量保障和常见陷阱。
sources:
  - ../../sources/papers/imitation_learning.md
  - ../../sources/papers/teleoperation.md
related:
  - ../tasks/teleoperation.md
  - ../entities/oculust-quest-teleop.md
  - ../concepts/embodied-data-cleaning.md
  - ../methods/imitation-learning.md
  - ../methods/behavior-cloning.md
  - ../tasks/bimanual-manipulation.md
---

# 操作任务演示数据收集指南

> **Query 产物**：本页由以下问题触发：「用模仿学习做操作，怎么高效收集人类演示数据？」
> 综合来源：[Teleoperation](../tasks/teleoperation.md)、[Imitation Learning](../methods/imitation-learning.md)、[Behavior Cloning](../methods/behavior-cloning.md)、[Bimanual Manipulation](../tasks/bimanual-manipulation.md)

## TL;DR 快速结论

- **首选系统**：ALOHA / ACT 系统（双臂、低延迟、开源）适合多数操作任务
- **数据量基准**：简单抓取 ~50 条，复杂双臂任务 ~200–500 条，精细操作可能需要 1000+
- **最重要原则**：宁可少而精，不要多而杂——一条高质量演示胜过五条不稳定演示

---

## 硬件选项对比

| 系统 | 自由度 | 延迟 | 成本 | 适用任务 | 优缺点 |
|------|-------|------|------|---------|--------|
| **ALOHA**（领导者-跟随者） | 14DoF（双臂各 7DoF） | 极低（直接关节映射） | 中（~15k USD DIY） | 双臂精细操作、桌面任务 | 沉浸感高、延迟低；需要专用硬件 |
| **[VR 手柄](../entities/oculust-quest-teleop.md)**（Quest / Vive） | 6DoF + 手指 | 低–中（20–50ms） | 低（~500 USD） | 远程操作、大范围运动 | 灵活通用；手指细节难捕捉 |
| **SpaceMouse** | 6DoF 末端速度控制 | 低 | 低（~150 USD） | 单臂、简单拾取放置 | 设置简单；不直观，学习曲线高 |
| **动觉示教**（Kinesthetic） | 全关节直接引导 | 零（直接物理接触） | 无额外硬件 | 精细任务、接触丰富操作 | 最直觉；需要机器人支持反向驱动 |
| **手套 / 手部追踪**（Leap / MediaPipe） | 手指 + 手腕 | 中 | 低–中 | 灵巧手操作 | 手指精度高；腕部绝对位置漂移 |

**选型原则：**
- 任务需要双手协调 → ALOHA 或双臂 VR 方案
- 资源有限、快速验证 → SpaceMouse（单臂）或动觉示教
- 灵巧手操作（多指） → 手部追踪 + 专用末端执行器

---

## 数据质量检查清单

### 演示前检查

- [ ] 机器人标定（末端执行器位置误差 < 2mm）
- [ ] 相机位置固定，确保任务执行过程中目标物体始终在视野内
- [ ] 物体初始位置设定方式一致（使用固定夹具或标记，而非目测）
- [ ] 录制系统时间戳同步（RGB 相机、深度相机、关节状态）
- [ ] 数据格式确认：`action`（关节位置/末端姿态）、`obs`（图像 + 本体感知）对齐

### 演示中标准

- [ ] 每条演示使用**相同策略**（避免在同一数据集里混入不同操作风格）
- [ ] 演示成功率 > 90%（失败演示原则上剔除，除非训练 DAgger）
- [ ] 任务完成时间标准差 < 20%（过高说明示教者操作不稳定）
- [ ] 关键节点动作清晰（抓取、插入等关键帧动作要明确，避免犹豫）

### 演示后验收 (详见 [具身数据清洗](../concepts/embodied-data-cleaning.md))

- [ ] 可视化回放：用 `rerun.io` 或 `rviz` 逐帧检查轨迹
- [ ] 剔除异常轨迹（关节速度突变、force/torque 异常峰值）
- [ ] 数据分布检查：用 PCA 或 t-SNE 可视化 action 分布，确认无明显离群簇
- [ ] 分辨率确认：图像分辨率、关节采样频率满足算法要求（通常 ≥ 50Hz）

---

## 提升数据效率的实践技巧

### 1. 任务分解（Task Segmentation）

将复杂任务拆分为子任务，分段收集演示，再用层次策略或子目标条件策略训练。每段复杂度低，示教者更容易保持一致性。

### 2. 数据增强（Augmentation）

- **颜色抖动**：对 RGB 图像做颜色扰动，提升对光照变化的鲁棒性
- **随机裁剪**：对图像做随机裁剪，提升对相机轻微偏移的鲁棒性
- **镜像翻转**：对左右对称任务使用翻转增强（需同时翻转 action）
- 注意：**不要**对接触状态敏感的动作序列做过强增强

### 3. 示教者筛选与训练

- 首次演示前给示教者看成功案例视频，统一操作策略
- 用前 10–20 条演示做成功率统计，不达标的示教者需重新培训
- 多个示教者数据混用时，用 `demonstrator_id` 标记，便于后续分析

### 4. 在线数据收集（DAgger 风格）

策略训练一轮后，让策略尝试执行，在策略失败处（分布外状态）补充人类修正演示，逐轮迭代改善分布覆盖。

---

## 常见陷阱

| 陷阱 | 后果 | 规避方法 |
|------|------|---------|
| 演示风格不统一（有时慢慢放，有时快速放） | BC 策略学到均值，两种风格都做不好 | 固定操作节奏，或用多模态策略（Diffusion Policy） |
| 物体初始位置随意放置 | 策略只学到特定初始位置 | 使用夹具或在固定范围内随机化初始位置 |
| 数据量不足就开始训练调参 | 浪费调参时间，最终发现是数据问题 | 先用 50 条验证 BC 能过拟合，再扩充数据 |
| 时间戳不对齐 | action 和 obs 不对应，策略学到错误映射 | 用硬件触发同步或统一 NTP 时钟 |
| 忘记记录末端执行器状态（open/close） | 策略无法学习抓取时机 | 确保 gripper 状态在 obs 和 action 中都有记录 |
| 演示过于依赖特定颜色/纹理的物体 | 换物体后策略失效 | 用多种颜色/纹理的同类物体收集演示 |

---

## 一句话记忆

> 「数据质量 > 数量；先保证成功率 > 90%，再追求多样性——脏数据用再多也训不好 BC。」

---

## 参考来源

- [Imitation Learning 源文档](../../sources/papers/imitation_learning.md)
- [Teleoperation 源文档](../../sources/papers/teleoperation.md)
- Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT/ALOHA, 2023)
- Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (2023)
- Ross et al., *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger, 2011)

## 关联页面

- [Teleoperation](../tasks/teleoperation.md) — 遥操作任务定义与系统设计
- [Imitation Learning](../methods/imitation-learning.md) — 模仿学习方法全景
- [Behavior Cloning](../methods/behavior-cloning.md) — BC 算法与局限
- [Bimanual Manipulation](../tasks/bimanual-manipulation.md) — 双臂操作任务挑战
