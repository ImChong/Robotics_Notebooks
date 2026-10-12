# CLOTHO: Canonicalizing IMUs from Loose Inertial Garments for Accurate Human Motion Tracking

> 来源归档（ingest）

- **标题：** CLOTHO: Canonicalizing IMUs from Loose Inertial Garments for Accurate Human Motion Tracking
- **类型：** paper
- **机构：** 厦门大学（Xiamen University）；麻省理工学院（MIT）；卡迪夫大学（Cardiff University）；卡内基梅隆大学（Carnegie Mellon University）
- **venue：** ACM Transactions on Graphics（SIGGRAPH Asia 2026）
- **DOI：** <https://doi.org/10.1145/3842534>
- **项目页：** <https://clotho-mocap.github.io/>
- **项目页列出的源码/数据链接：** 截至 2026-10-12 未能核实可访问的代码仓库或数据下载地址；项目页表示将开放软件、权重、硬件和数据。
- **入库日期：** 2026-10-12
- **一句话说明：** 面向松散穿戴惯性服装，把受服装滑移、材料、贴合度与传感器布局影响的异构 IMU 观测 canonicalize 为漂移校正、空间完整的标准 10-IMU 表示，再交给下游网络恢复全身运动。

## 核心论文摘录（策展）

### 1) 问题：宽松服装让惯性测量与人体运动脱耦

- **摘录要点：** 传统惯性动捕通常假设传感器固定在预定身体位置。服装宽松或穿戴位置变化时，IMU 测到的是服装介导的运动；材料、松紧、传感器数目和布局都会改变观测，误差会随时间累积并产生漂移。CLOTHO 将任务拆成「先修复传感表示，再估计人体运动」，避免让姿态网络同时承担穿戴域偏移与重建。
- **对 wiki 的映射：** [paper-clotho-loose-inertial-garments](../../wiki/entities/paper-clotho-loose-inertial-garments.md) — 问题设定与设计动机。

### 2) 方法：配置不变的 latent + 条件残差 canonicalization

- **摘录要点：** 输入为不同服装的稀疏/异构 IMU 序列与传感器掩码；Masked IMU Encoder 从不完整观测学习配置不变 latent，Latent-Conditioned IMU Canonicalizer 根据该 latent 预测残差校正，将输入映射到标准 10-IMU 表示，最后由下游运动重建器输出全身运动。关键设计是把可穿戴传感器几何/时序误差与 pose reconstruction 解耦。
- **对 wiki 的映射：** [paper-clotho-loose-inertial-garments](../../wiki/entities/paper-clotho-loose-inertial-garments.md) — 方法流程与主干图。

### 3) 数据与结果：五种服装、跨布局零样本与长时跟踪

- **摘录要点：** 项目页报告数据规模为约 **12.5M IMU 帧 / 116 小时**，覆盖 5 种服装设计、材料、布局、传感器数量和硬件。Green-10 / Taichi-10 / White-10 使用低成本 BNO085；Purple-6 使用 Xsens MTi-3；T-Shirt-4 使用 Noitom PN3。项目页称在各测试服装上位置误差最低，相对 FIP/LIP 的位置误差降低 **42%**，并对未见服装/传感器数配置零样本泛化；展示最长 **8 小时**跟踪。其消融还报告 7.07 小时配对标签可将 OME 从 16.05° 降至 13.78°，并指出真实少量数据胜过大量合成服装数据。
- **对 wiki 的映射：** [paper-clotho-loose-inertial-garments](../../wiki/entities/paper-clotho-loose-inertial-garments.md) — 数据集、基线与评测读法。

### 4) 复现边界与来源状态

- **摘录要点：** 论文 DOI 为 `10.1145/3842534`。截至入库日项目页宣称将公开软件、模型权重、硬件与数据，但网页未能确认到可访问的仓库或数据下载入口；因此目前不能据此复现实验或判断许可、预处理和训练细节。卡迪夫 ORCA 记录论文 In Press 元数据及受限全文状态。
- **对 wiki 的映射：** [paper-clotho-loose-inertial-garments](../../wiki/entities/paper-clotho-loose-inertial-garments.md) — 工程实践和开源状态。

## 对 wiki 的映射

- [paper-clotho-loose-inertial-garments](../../wiki/entities/paper-clotho-loose-inertial-garments.md)
- [Human Motion 论文索引](../../wiki/overview/paper-notebook-category-14-human-motion.md)
- 相关实体：[EgoExoMoCap](../../wiki/entities/paper-egoexomocap.md)、[AvatarPoser](../../wiki/entities/paper-notebook-avatarposer-articulated-full-body-pose-tracking.md)
