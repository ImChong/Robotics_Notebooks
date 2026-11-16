# FAST-LIVO（arXiv:2203.00893）

> 来源归档（paper）

- **标题：** FAST-LIVO: Fast and Tightly-coupled Sparse-Direct LiDAR-Inertial-Visual Odometry
- **类型：** paper
- **作者与机构：** Chunran Zheng*、Qingyan Zhu*、Wei Xu、Xiyuan Liu、Qizhi Guo、Fu Zhang（香港大学机械工程系；* 同等贡献）
- **arXiv：** <https://arxiv.org/abs/2203.00893>（本次读取 v1，2022-03-02）
- **PDF：** <https://arxiv.org/pdf/2203.00893v1>
- **代码：** https://github.com/hku-mars/FAST-LIVO
- **发表：** 论文 PDF（v1）正文未写会议名；仓库 README 写明「accepted to IROS2022」。
- **入库日期：** 2026-11-11
- **核对范围：** arXiv v1 PDF 全文（摘要、I 引言、II 相关工作、III 系统总览、IV 状态估计、V 地图管理、VI 实验、VII 结论与参考文献，正文无附录 / 补充材料）逐段读完，Fig. 1–7 的图注与 Table I–III 的数字已核；Fig. 5–7 的图像本身只读图注，未逐像素审图。仓库只读了 README 前 60 行（简介、论文引用、依赖开头），未读其余 README 与全部源码；未读 v2 及以后版本（若存在）；未运行任何代码。
- **通读：** 已完成
- **一句话说明：** 在 FAST-LIO2 的直接法 LIO 上增加稀疏直接法 VIO 子系统，二者共用同一份 LiDAR 点云地图，在同一个误差状态迭代卡尔曼滤波（ESIKF）里联合更新。

## 开源状态

- 代码仓库存在（README 读到）：https://github.com/hku-mars/FAST-LIVO；README 新闻栏称后续版本为 FAST-LIVO2（本库另有独立节点，勿混淆）。

## 核心摘录（均出自论文，括号为章节 / 表号）

1. **总体结构（III，Fig. 1）：** 两个子系统。LIO 先做后向传播去运动畸变，再算帧到地图的点到面残差；VIO 从视觉全局地图取当前视场内的子图，剔除遮挡 / 深度不连续点，再做稀疏直接法图像对齐。LiDAR 点到面残差、图像光度误差与 IMU 传播在一个 ESIKF 中紧耦合。传感器频率图注为 LiDAR 10~100 Hz、相机 10~50 Hz、IMU 20~150 Hz（Fig. 1 图中数字，文字抽取顺序不完全确定，按图注读）。
2. **状态与时间假设（IV-B）：** 状态在 SO(3)×R^15（维数 18），含 IMU 姿态、位置、速度、陀螺 / 加计零偏与重力向量；假设三传感器时间偏移已知、外参已预标定。LiDAR 与图像不要求同时到达，任一到达即触发一次更新（IV-C）。
3. **两类测量模型（IV-D）：** LiDAR 取最近 5 个点拟合平面，残差为点到面距离（式 4，沿用 FAST-LIO2 的 ikd-Tree）；视觉为帧到地图的稀疏直接对齐，在图像金字塔三层由粗到细最小化光度误差（式 5），参考 patch 取观测角度与当前帧最接近的那一块。式 5 含乘子 Ai，正文未给出其定义（本页不推断）。
4. **耦合方式（II-B、IV-E）：** 视觉直接复用 LiDAR 点作为深度，无需视觉特征提取、三角化或滑窗后端；式 7 在 MAP 目标里把先验、LiDAR 残差、光度残差相加，收到 LiDAR 时令 mc=0，收到图像时令 ml=0。
5. **视觉地图（V-B）：** 地图点放在等大轴对齐体素的哈希表中，每点附带多个来源图像的 patch 金字塔与相机位姿；新 patch 为 8×8 像素；40×40 像素网格内保留最低深度点；用最近一帧 LiDAR 点在 9×9 邻域内检查遮挡；距上次加 patch 超过 20 帧，或像素位移超过 40 像素，则新增 patch；新增地图点时取每个 40×40 网格内梯度最大者，跳过高曲率边缘点。
6. **NTU-VIRAL 九条序列（VI-A，Table II）：** 对 FAST-LIO2、SVO2.0、R2LIVE、去回环的 DVL-SLAM 比 RMSE（米）。FAST-LIVO 在 8/9 序列最优，仅 sbs_01 为 0.29，高于 FAST-LIO2 的 0.25；作者解释为 UAV 快速运动使图像极度模糊，融合低质量图像无助于精度。SVO2.0 在 eee_01、eee_02、sbs_02、sbs_03 失败。
7. **私有数据（VI-B）：** 平台为 DJI Manifold-2c（i7-8550U、8 GB）、MV-CA013-21UC 工业相机、Livox Avia，10 Hz 硬同步。LiDAR 退化场景（约 30 m 长墙）端到端漂移最低 0.05 m；视觉挑战场景（室内外切换、两次剧烈运动、无纹理白墙）总路程 79.52 m，端到端误差 0.04 m；还实时重建了 HKU 校园 RGB 着色点云（Fig. 7）。
8. **耗时（VI-C，Table III）：** 桌面机 Intel i7 上 VIO 子系统合计 10.23 ms、LIO 26.52 ms（合计 36.75 ms）；Qualcomm RB5（Kryo585 ARM）上 VIO 13.82 ms、LIO 51.51 ms。对比 R2LIVE 在同一批私有数据上均耗时 45.16 ms（前端）+ 59.27 ms（VIO 后端滑窗），FAST-LIVO 全流程 36.75 ms。作者结论为两种处理器均可实时并有显著计算余量。
9. **作者自述的支持范围（摘要、贡献 3）：** 同时支持多线旋转式与固态 LiDAR（扫描模式完全不同）。

## 对 wiki 的映射

- [paper-fast-livo](../../wiki/entities/paper-fast-livo.md)
