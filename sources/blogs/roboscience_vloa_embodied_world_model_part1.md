# Embodied World Model — Introducing the VLOA Large Model (Part 1)

> 来源归档（blog / RoboScience 官方英文站）

- **标题：** Embodied World Model — Introducing the VLOA Large Model (Part 1)
- **类型：** blog
- **作者：** RoboScience
- **原始链接：** <https://en.roboscience.co/jishu_detail/18.html>
- **发表日期：** 2026-03-23
- **入库日期：** 2026-07-19
- **抓取方式：** 官方技术博文页面直接抓取（WebFetch）
- **一句话说明：** RoboScience 披露 VLOA 双引擎之一 **具身世界模型**：以语言 + 视觉为输入，输出 **物体中心 3D 点云轨迹**（含位姿、时间步、置信度），经 world-causal Transformer 建模动态三维环境，强调物理约束、多解扩散、长时一致与硬件解耦。

## 核心摘录（归纳，非全文）

### 问题重框

- 领域常见两路：**2D 视频生成**（缺三维空间理解）与 **3D 静态重建**（不能预测时序运动）。
- RoboScience 走第三路：**3D dynamic world modeling** — 预测物体在三维空间中的连续轨迹。
- 在 VLOA 中，世界模型负责 **理解物理世界 + 模拟未来轨迹**；与 **General Manipulation Model** 形成闭环。

### 表示与架构

| 要素 | 内容 |
|------|------|
| **输入** | 自然语言指令 + 单/多视角图像 |
| **输出** | 带时间戳的 **3D 点云轨迹**（位置、姿态、时间步、置信度） |
| **骨干** | RGB + 3D 点云先验 + 指令编码 → **world-causal Transformer** → 解码 3D flow；可选分支生成未来操作视频 |
| **为何点云** | 可解释可视化、满足几何约束、可直接馈送下游操作模型 |

### 四项核心能力（博文归纳）

1. **跨物体泛化** — 不同材质/形状/尺寸（洗发水瓶、棉签盒、饮料盒等）无需逐物体重训。
2. **动态过程建模** — 倒水、放置杯子等涉及流体与细粒度接触的过程。
3. **指令跟随与实例区分** — 同类物体不同语义指令（洗衣篮分拣等）。
4. **四技术属性** — 物理约束满足、扩散式多解建模、长时时空一致、**硬件解耦**（臂/人形/灵巧手共享轨迹计划）。

### 数据与 Scaling 叙事

- 互联网视频自动标注清洗：**>100 万小时** 高维多模态操作视频（数千万 clip），**周增数十万小时**；2026 年底目标 **>1000 万小时**。
- 操纵侧（引述 Part 2）：RoboMirage 仿真已积累 **100 亿条** 全空间物体操纵数据，2026 目标 **1 万亿条**。
- 训练曲线：Content Alignment、Subjective Quality、Photometric Consistency、Motion Smoothness 随迭代提升（博文 Figure 1–2）。

### 对 wiki 的映射

- 实体页：[wiki/entities/roboscience-vloa.md](../../wiki/entities/roboscience-vloa.md)
- 方法对照：[wiki/methods/generative-world-models.md](../../wiki/methods/generative-world-models.md)、[wiki/entities/molmo-motion.md](../../wiki/entities/molmo-motion.md)（同为语言条件 3D 轨迹）
- 级联架构：[wiki/overview/world-models-route-01-cascade.md](../../wiki/overview/world-models-route-01-cascade.md)
