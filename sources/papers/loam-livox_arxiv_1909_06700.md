# Loam_livox（arXiv:1909.06700）

> 来源归档（paper）

- **标题：** Loam livox: A fast, robust, high-precision LiDAR odometry and mapping package for LiDARs of small FoV
- **作者 / 机构：** Jiarong Lin, Fu Zhang（香港大学机械工程系，论文首页署名 Department of Mechanical Engineering, Hong Kong University）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/1909.06700>（v1，2019-09-15）
- **PDF：** <https://arxiv.org/pdf/1909.06700v1>
- **代码：** https://github.com/hku-mars/loam_livox（论文脚注 1；GPLv2，README 末节）
- **发表信息：** arXiv v1 首页与全文未标注会议；任务说明称其为 ICRA 2020 论文，本归档未在原文中核实
- **关联论文：** Lin & Zhang，*A fast, complete, point cloud based loop closure for LiDAR odometry and mapping*，[arXiv:1909.11811](https://arxiv.org/abs/1909.11811)（v1，2019-09-25），同一仓库的回环部分，见下文「关联论文核对」
- **入库日期：** 2026-11-11
- **一句话说明：** 面向小视场、非重复扫描固态激光雷达（以 Livox MID40 为例）的 LOAM：点级筛选 + 反射率边缘特征 + 分段运动补偿 + 迭代位姿优化与离群点剔除 + 并行实现，不用 IMU，里程计与建图均 20 Hz。
- **核对范围：** 通读 arXiv:1909.06700v1 全文（共 6 页：摘要、§I 引言、§II 相关工作、§III 点筛选与特征提取、§IV 迭代位姿优化含 Algorithm 1、§V 结果含 Fig. 8–10 与 Table I、§VI 结论、参考文献）；该版本无附录 / 补充材料。另通读关联论文 arXiv:1909.11811v1 全文（§I–§VIII、Table I）以判断综述所指；读官方仓库 README（静态，未读源码、未运行代码、未下载 rosbag）。图中的轨迹与点云效果只看了图注与正文描述，未做独立数值复核。
- **通读：** 已完成

## 综述所指论文的判断

综述原话：「Loam_livox 由香港大学 Lin 提出，针对 Livox 新型固态激光雷达扫描特性，对有效点筛选和特征提取部分进行优化」。「有效点筛选 + 特征提取」对应 1909.06700 的 §III（Points selection and feature extraction）；1909.11811 是回环（2D 直方图 + NCC），不涉及点筛选与特征提取。故综述指 **1909.06700**，本库只建这一节点，1909.11811 作为关联说明。

## 核心摘录（均出自 1909.06700v1）

1. **问题（§I）：** 固态激光雷达的三个难点：小视场（Livox MID40 为 38.4 度锥形）、非重复的玫瑰花瓣式扫描（特征不易稳定跟踪）、连续扫描造成的运动模糊。仅用 LiDAR，不用 IMU / GPS / 相机，将当前帧注册到局部地图得到位姿。
2. **点筛选（§III-A）：** 逐点计算深度 D、偏转角 φ、强度 I = R / D²、入射角 θ；剔除四类点：视场边缘（MID40 取 φ ≥ 17 度）、强度过大或过小（MID40 取 I ≤ 7×10⁻³ 或 I ≥ 1×10⁻¹）、入射角接近 0 或 π（θ ≤ 5 度或 ≥ 175 度）、被遮挡点（|Pe − Pd| ≥ 0.1|Pe| 且 |Pe| > |Pd|）。上述阈值是论文给出的 MID40 例值。
3. **特征提取（§III-B）：** 沿用 LOAM 的局部平滑度提取平面 / 边缘特征（引用 [12]），额外把反射率当第 4 维：某点反射率与邻域差异显著则视为边缘特征（材质变化边缘），用于对着关闭的门窗墙面等退化情形。
4. **迭代位姿优化（§IV）：** 边特征点取 5 个近邻，协方差最大特征值须大于次大特征值的 3 倍才认为成线（式 6）；面特征点取 5 个近邻，最小特征值须比次小特征值小 3 倍才认为成面（式 7）。KD 树由另一线程在上一帧注册完成后立即重建。
5. **运动补偿（§IV-C）：** 两种方法：分段处理（把一帧切成 3 个子帧，各自独立与同一地图配准，每个子帧时间间隔为原帧 1/3，可多线程并行）；线性插值（沿用 LOAM 一类做法，式 8–9）。
6. **离群点剔除 / 动态物体过滤（§IV-D、Algorithm 1）：** 每轮先做 2 次迭代的位姿优化，计算两类残差，去掉最大的前 20% 后再做完整优化。
7. **结果（§V）：** 见页面「核心机制」与「工程实践」两节引用的 Fig. 8、Fig. 9、Fig. 10、Table I。
8. **局限（§VI）：** 作者自述顺序扫描匹配"inherently drifting"，减小漂移的回环与滑窗优化留作未来工作。

## 关联论文核对（arXiv:1909.11811v1）

- **标题：** A fast, complete, point cloud based loop closure for LiDAR odometry and mapping；作者同为 Jiarong Lin, Fu Zhang。
- **方法（§III–§VI）：** 每约 100 帧成一个关键帧（§III）；将 3D 空间划分为固定大小的 cell（哈希表 + 八叉树，§IV）；对平面 / 线 cell 的特征方向做旋转归一化后生成两张 60×60（3 度分辨率）的 2D 直方图（§V）；用直方图的归一化互相关（NCC）度量相似度，阈值例值：平面 0.90、线 0.65（§VI-B）；检测到回环后做边-边 / 面-面地图对齐（平均距离 < 0.1 m 视为对齐，§VI-C），再用 Google Ceres 做位姿图优化（§VI-D）。
- **耗时（Table I）：** 2D 直方图计算 1.18 ms（桌面机）/ 1.48 ms（机载）；地图对齐 621 ms / 931 ms；两图相似度 13 µs / 16 µs；在 HKUST 大规模数据集上测得。
- **实验（§VII、Fig. 7）：** 4 个 Livox MID40 数据集，评价主要是轨迹 / 点云回环前后对比和与谷歌地图对齐的目视，**没有给出 ATE 等量化误差**。
- 该文**不改变**前端的点筛选与特征提取，故不是综述所指。

## 仓库 README 核对（静态）

- 仓库自述为"robust, low drift, and real time odometry and mapping package for Livox LiDARs"；开发者 Jiarong Lin；列出两篇 arXiv 论文（1909.06700、1909.11811）；声明参考了 LOAM、LOAM_NOTED、A-LOAM；包含回环、地图保存与重载等工具；Ubuntu 16.04 / 18.04 + ROS Kinetic / Melodic，依赖 Ceres 与 PCL（建议 PCL 1.9）；许可证 GPLv2。
- 本归档**未核实**：源码是否与论文逐项一致、Mid-100 / 其他 Livox 型号的实际表现。

## 对 wiki 的映射

- [paper-loam-livox](../../wiki/entities/paper-loam-livox.md)
