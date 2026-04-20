---
type: query
tags: [manipulation, contact-rich, impedance-control, bimanual, assembly, force-control]
status: complete
updated: 2026-04-20
summary: "接触丰富操作实践指南：聚焦装配、插拔、拧紧、擦拭等任务中最关键的接触建模、柔顺执行、数据采集与调试策略。"
sources:
  - ../../sources/papers/contact_planning.md
  - ../../sources/papers/contact_dynamics.md
  - ../../sources/papers/imitation_learning.md
related:
  - ../tasks/manipulation.md
  - ../concepts/contact-rich-manipulation.md
  - ../tasks/bimanual-manipulation.md
  - ../concepts/impedance-control.md
  - ../concepts/hybrid-force-position-control.md
---

# Query：接触丰富操作实践指南

> **Query 产物**：本页由以下问题触发：「做接触丰富的操作任务（装配/拧螺丝），有哪些实践要点？」
> 综合来源：[Manipulation](../tasks/manipulation.md)、[Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md)、[Bimanual Manipulation](../tasks/bimanual-manipulation.md)

## TL;DR 决策表

| 场景 | 首选执行层 | 关键观测 | 高风险失败模式 | 第一优先修复 |
|------|-----------|---------|---------------|-------------|
| 插孔/装配 | 阻抗控制 + 末端位姿目标 | 接触力、法向误差、插入深度 | 卡住、侧向顶死 | 降低刚度，做搜索动作 |
| 拧螺丝/旋钮 | 力位混合控制 | 轴向压力、扭矩、角位移 | 滑牙、空转、过力 | 增加轴向预紧，限制扭矩 |
| 擦拭/打磨 | 柔顺轨迹跟踪 | 法向力、接触面积 | 力过大、接触丢失 | 先稳法向力，再提速度 |
| 双手装配 | 双臂协调 + 内力约束 | 两臂相对位姿、内力 | 物体被拉扯、闭链冲突 | 统一物体坐标系规划 |

**总原则**：接触丰富操作优先解决“接触是否稳定”，再优化“动作是否快”。

---

## 1. 先分清你在解决哪一种接触问题

接触丰富任务经常被笼统地当成“更难的 manipulation”，但工程上最好先拆成三类：

1. **接触建立**：末端第一次碰到环境，重点是避免碰撞过猛。
2. **接触维持**：接触已经发生，重点是保持法向力和摩擦条件稳定。
3. **接触切换**：从一个接触模态切到另一个接触模态，例如“贴住 -> 滑动 -> 插入”。

如果没有先分阶段，策略会把所有困难都混在一起：既要找位置，又要调力，还要处理卡住后的恢复。实践里通常至少要有一个显式阶段机，或者让策略输出 action chunk 后交给低层控制器做阶段内执行。

---

## 2. 低层控制不要只用纯位置控制

对自由空间抓取，位置控制往往够用；但对装配、擦拭、拧紧一类任务，纯位置控制会把感知误差直接放大成硬碰撞。更稳的做法通常是：

- **末端位姿目标 + 阻抗控制**：最通用，适合插孔、擦拭、沿边移动。
- **力位混合控制**：一部分方向控位置，一部分方向控力，适合拧螺丝、按压、沿表面滑动。
- **WBC/TSID + 接触约束**：适合人形或双臂场景，接触力、平衡和多任务要一起算。

一个经验法则是：只要任务成功依赖“碰到以后怎么用力”，就至少要有阻抗或力控缓冲层。

---

## 3. 观测里要显式保留接触信息

常见失败并不是策略不会到达目标，而是它不知道“已经碰上了什么”。因此观测设计里最好显式保留：

- 力/力矩传感器读数，或可替代的电流/估计接触力
- 末端相对目标的位姿误差
- 接触法向、滑移迹象、插入深度等任务特征
- 最近若干步动作和观测历史，用于判断“碰撞后是否在振荡”

如果没有直接力传感器，也要在调试层记录：电机电流峰值、速度骤降、末端误差是否突然增大。这些都可以作为接触事件代理信号。

---

## 4. 数据采集要覆盖“失败边界”

对接触丰富任务，最有价值的示范往往不是完美成功轨迹，而是**接近失败边界但又被恢复**的轨迹。因为真实部署时，大多数问题都发生在：

- 起始位姿有几毫米偏差
- 接触角度轻微错误
- 摩擦系数与仿真不一致
- 两臂时序不同步

因此采集数据时建议主动加入：

- 小范围位姿扰动
- 不同接触角度/不同接触顺序
- 失败恢复示范，例如“插不进去后如何搜索重试”
- 双臂任务中的一手固定、一手调整的协调过程

如果全是完美演示，策略往往只学会“理想情况如何成功”，而学不会“偏一点时如何救回来”。

---

## 5. 调试顺序：先力，再几何，再策略

真机失败时，不要一上来怀疑大模型。接触丰富任务更高效的排查顺序通常是：

1. **低层力学参数**：阻抗刚度是不是过大，是否缺少力矩限幅。
2. **几何与坐标系**：末端、相机、夹爪、物体坐标是否一致。
3. **接触阶段机**：是不是在错误阶段执行了错误动作。
4. **策略输出稳定性**：动作是否高频抖动，是否需要 action chunk buffer。

尤其对插孔类任务，很多“策略失败”本质上是坐标系偏了 3~5 mm，或者刚度太硬导致一碰就弹开。

---

## 6. 什么时候要上双臂协调

如果任务满足以下任一条件，通常就不该把它看成两个独立单臂任务：

- 两只手同时抓同一刚体
- 一只手的动作会改变另一只手的接触条件
- 任务成功依赖内力平衡或闭链一致性

此时应尽量：

- 用统一的物体坐标系来规划两臂参考
- 显式限制内力或抓持力分配
- 在策略层输出协同目标，而不是两臂完全独立动作

---

## 常见误区

- **误区 1：把 contact-rich 当成视觉定位问题。**  
  视觉只决定“去哪里”，接触控制决定“碰上以后能不能成功”。
- **误区 2：刚度越高越稳定。**  
  过高刚度会让误差直接转成冲击力，反而更不稳定。
- **误区 3：双臂任务可以分别训练再拼起来。**  
  共同操持物体时会出现闭链约束和内力问题，必须协调建模。

## 参考来源

- [sources/papers/contact_planning.md](../../sources/papers/contact_planning.md) — 接触序列组织、接触约束与装配规划
- [sources/papers/contact_dynamics.md](../../sources/papers/contact_dynamics.md) — 接触力、摩擦锥、力位约束基础
- [sources/papers/imitation_learning.md](../../sources/papers/imitation_learning.md) — ALOHA / ACT / Diffusion Policy 等操作任务数据与策略路线
- Mordatch et al., *Contact-Invariant Optimization for Hand Manipulations*

## 关联页面

- [Manipulation](../tasks/manipulation.md) — 操作任务总览，帮助区分非接触与接触丰富任务
- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md) — 本 Query 的概念基础页
- [Bimanual Manipulation](../tasks/bimanual-manipulation.md) — 双臂装配与内力协调的代表场景
- [Impedance Control](../concepts/impedance-control.md) — 接触任务最常见的柔顺执行层
- [Hybrid Force-Position Control](../concepts/hybrid-force-position-control.md) — 装配、擦拭、拧紧任务中的典型方向拆分方法
