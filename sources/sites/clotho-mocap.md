# CLOTHO 项目主页

> 来源归档（ingest）

- **项目：** CLOTHO: Canonicalizing IMUs from Loose Inertial Garments for Accurate Human Motion Tracking
- **类型：** project site
- **链接：** <https://clotho-mocap.github.io/>
- **论文：** ACM Transactions on Graphics / SIGGRAPH Asia 2026；DOI <https://doi.org/10.1145/3842534>
- **机构：** 厦门大学、麻省理工学院、卡迪夫大学、卡内基梅隆大学
- **核查日期：** 2026-10-12
- **开源状态：** 项目页宣称软件、权重、硬件和数据将开放；页面可见 Code 入口，但当前未能确认可访问的源码仓库或数据集 URL。故记录为「发布计划已声明，实际可获取内容待核实」，不归档猜测的仓库地址。
- **页面摘要：** CLOTHO 先将受宽松服装影响的 IMU 数据修复为漂移校正、空间完整的标准 10-IMU 表示，再进行人体运动恢复。页面报告采集 12.5M IMU 帧、约 116 小时、5 种服装设计，强调跨未见服装零样本泛化与最长 8 小时跟踪。

## 页面摘录

- **问题定义：** 松散惯性服装的传感器与人体之间存在时变耦合，服装、材料、松紧度、布局和 IMU 数量共同改变测量分布。
- **架构：** Masked IMU Encoder 编码部分观测；Latent-Conditioned IMU Canonicalizer 预测残差校正；标准表示交给下游 motion reconstruction。
- **传感器配置：** 三套 10 IMU BNO085 服装、一套 6 IMU Xsens MTi-3 服装、一套 4 IMU Noitom PN3 T 恤。
- **项目页报告：** 相较 FIP/LIP 位置误差降低 42%；可在未见服装配置上零样本泛化；长时测试展示 8 小时跟踪。
- **开放计划：** 页面说明软件、权重、硬件和数据将开放；截至核查日期没有确认到可获取链接。

## 对 wiki 的映射

- [CLOTHO 论文实体页](../../wiki/entities/paper-clotho-loose-inertial-garments.md)
- [论文来源摘录](../papers/clotho.md)
