# SE(3)-LIVOM: Adaptive Multi-LiDAR-Inertial-Visual Odometry and Mapping

- **类型：** paper
- **作者：** Gunhee Shin、Seungjae Lee、Daebeom Kim、Seoyeon Jang、Hyun Myung
- **机构：** 韩国科学技术院（KAIST）
- **日期：** 2026-11-17
- **arXiv：** <https://arxiv.org/abs/2611.22047> v1
- **PDF：** <https://arxiv.org/pdf/2611.22047>
- **HTML：** <https://arxiv.org/html/2611.22047>
- **代码：** <https://github.com/url-kaist/se3-livom-comfort>（[源码归档](../repos/se3_livom_comfort.md)）
- **入库日期：** 2026-11-24
- **核对范围：** v1 全部 3 页，摘要、I–IV 节、图 1–2、表 I–II、参考文献；官方 README 与运行脚本。
- **通读：** 已完成

用户给出的 2603.16118 与 https://se3-lio.github.io/ 是前序 SE(3)-LIO（论文参考文献 [6]），不是本篇的论文编号或项目页。

## 摘录 1：前端（II-A）
双 LiDAR 时间排序合并，保留传感器噪声模型；SE(3) 联合位姿传播与不确定性感知去畸变；LiDAR 几何更新之后进行稀疏直接光度更新。以内点目标 2000 自适应调整体素边长，范围 0.1–0.5 m。
**对 wiki 的映射：** [SE(3)-LIVOM](../../wiki/entities/paper-se3-livom.md)。

## 摘录 2：后端（II-B）
每 25 扫描生成子图，每新增 4 子图做全局平面 BA；iSAM2 增量融合里程计、静止重力及子图间相对位姿因子，10 Hz 输出。
**对 wiki 的映射：** [SE(3)-LIVOM](../../wiki/entities/paper-se3-livom.md)。

## 摘录 3：评测与边界（III–IV）
GrandTour 六测试场景平均 ATE RMSE：FAST-LIVO2 2.16 cm，前端 1.66 cm，完整 1.40 cm。前端均值约 28–42 ms；CON-4 后端最长 19.4 s，10 帧前端超过 100 ms。各场景 IMU 时间偏移独立估计。IROS 2026 COMFORT Challenge 总冠军为作者报告。
**对 wiki 的映射：** [SE(3)-LIVOM](../../wiki/entities/paper-se3-livom.md)。
