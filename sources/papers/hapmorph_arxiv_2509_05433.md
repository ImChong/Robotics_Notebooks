# HapMorph: A Pneumatic Framework for Multi-Dimensional Haptic Property Rendering

> 来源归档（ingest）

- **标题：** HapMorph: A Pneumatic Framework for Multi-Dimensional Haptic Property Rendering
- **类型：** paper（可穿戴触觉 / 气动软体执行器 / VR·遥操作力反馈）
- **原始链接：**
  - arXiv abs：<https://arxiv.org/abs/2509.05433>
  - arXiv HTML：<https://arxiv.org/html/2509.05433v1>
- **作者：** Rui Chen, Domenico Chiaradia, Antonio Frisoli, Daniele Leonardis
- **机构：** Institute of Mechanical Intelligence, Scuola Superiore Sant'Anna (SSSA), Pisa, Italy
- **入库日期：** 2026-06-29
- **一句话说明：** 基于 **拮抗式织物气动执行器（AFPA）** 的 **HapMorph** 可穿戴框架，通过双腔压力 $P_1$/$P_2$ **解耦** 调节交互界面 **尺寸（50–104 mm）** 与 **刚度（最高约 4.7 N/mm）**；可穿戴部件仅 **21 g**；10 人感知实验在 9 种尺寸×刚度组合上达 **89.4%** 识别准确率。

## 核心论文摘录（MVP）

### 1) 问题：可穿戴触觉难以同时渲染几何与力学属性

- **链接：** arXiv HTML §1
- **摘录要点：** 触觉接口需在便携形态下同时调制 **尺寸、形状、刚度** 等多维物理属性；移动机器人/shape display 几何保真高但不可穿戴；织物气动与可穿戴外骨骼便携但常只能单维或离散切换；Takizawa 等可兼顾尺寸+刚度但需固定安装与复杂电机；颗粒 jamming 体积大。现有单腔气动系统普遍存在 **尺寸–刚度耦合**。
- **对 wiki 的映射：**
  - [HapMorph 实体页](../../wiki/entities/paper-hapmorph-pneumatic-haptic-render.md) — 问题定位与相对 encountered-type / jamming / 单腔气动对照。

### 2) AFPA 结构与双腔解耦控制

- **链接：** §2 Results — HapMorph System Design；Fig. 1
- **摘录要点：**
  - 三个 **pouch motor**（TPU 涂层织物，CNC 热封）缝合成单侧膨胀执行器；两个执行器经交叉拉力带构成 **AFPA**：右侧为 **morphing actuator**（用户接触面），左侧为 **modulating actuator**。
  - 双腔压力 $P_1$（调制腔）与 $P_2$（形态腔）经拮抗力学平衡：**$P_2$ 主导感知刚度**，**$P_1$ 连续调节 $H_2$ 高度（尺寸）**；气路经腕带引至远端气源与比例阀（SMC ITV0010 + ESP32 闭环）。
  - 可穿戴绑带（掌+腕）总质量 **21 g**；目标场景：**VR 与遥操作** 中模拟交互物体尺寸与刚度。
- **对 wiki 的映射：**
  - 同上 — 「系统结构」与 Mermaid 流程。

### 3) 尺寸与刚度表征

- **链接：** Fig. 2–3；Supplementary
- **摘录要点：**
  - **尺寸：** 虚拟功原理建模；$H_2$ 在 **50–104 mm**；阶跃响应约 **2 s**（受供气速度限制）；存在加载/卸载滞回。
  - **刚度：** 准静态压缩（1.5 mm/s）；$H_2=15$ mm 时最高约 **4.7 N/mm**、力达 **72 N**；四组代表压力组合可实现 **0.12–0.46 N/mm**（约 4 倍跨度）。
  - **解耦：** 在 52 / 73 / 96 mm 三高度下，经 $P_1$/$P_2$ 配平可维持约 **0.1±0.015 N/mm** 相近刚度——相对单腔气动是核心贡献。
- **对 wiki 的映射：**
  - 同上 — 表征数据表与解耦机制。

### 4) 扩展架构与感知实验

- **链接：** Fig. 4–5；§4 Discussion
- **摘录要点：**
  - **扩展：** 单 pouch 调制器（更线性但滞回更大）；内嵌约束线实现圆柱↔长方体 **形状变形 + 刚度**；双调制腔 + 四层形态腔实现 **高度与侧向位移** 等多 DOF。
  - **用户研究：** 10 人（6M/4F）；盲fold + 降噪；9 状态 = 3 尺寸 × 3 刚度；每状态 10 次共 90 trial；**总准确率 89.4%**（单状态 78%–98%）；平均响应 **6.7 s**（含约 2 s 形态过渡与记录）；**尺寸辨别优于刚度**；trial 推进中准确率下降（疲劳/状态多样性）。
  - **局限：** 外部气源限制便携；~2 s 响应；滞回需补偿算法；纹理/温度未覆盖。
- **对 wiki 的映射：**
  - 同上 — 扩展架构、感知结果与局限。
  - [Teleoperation 任务页](../../wiki/tasks/teleoperation.md) — 操作员侧可穿戴力反馈与 VR 渲染交叉引用。
  - [触觉专题汇总](../../wiki/overview/topic-tactile.md) — 可穿戴 haptic rendering 入口。

## 当前提炼状态

- [x] arXiv HTML §1–4 + 方法节已摘录
- [x] wiki 映射：`wiki/entities/paper-hapmorph-pneumatic-haptic-render.md`
- [x] 交叉更新：`wiki/tasks/teleoperation.md`、`wiki/overview/topic-tactile.md`
