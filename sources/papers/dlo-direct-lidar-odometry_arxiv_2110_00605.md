# Direct LiDAR Odometry: Fast Localization with Dense Point Clouds（arXiv:2110.00605）

> 来源归档（paper）

- **标题：** Direct LiDAR Odometry: Fast Localization with Dense Point Clouds
- **类型：** paper
- **作者：** Kenny Chen、Brett T. Lopez、Ali-akbar Agha-mohammadi、Ankur Mehta（UCLA / NASA JPL，Team CoSTAR）
- **发表：** IEEE Robotics and Automation Letters (RA-L) 2022，7(2)：2000-2007，DOI 10.1109/LRA.2022.3142739（DOI 与卷期页取自官方仓库 README 的 BibTeX；论文 PDF 页眉写明 2022 年 1 月录用）
- **arXiv：** <https://arxiv.org/abs/2110.00605>（v1 2021-10-01；v3 2022-01-07 为 RA-L 录用稿）
- **PDF：** <https://arxiv.org/pdf/2110.00605v3>
- **代码：** https://github.com/vectr-ucla/direct_lidar_odometry （README 称 MIT 许可）
- **入库日期：** 2026-11-11
- **一句话说明：** 面向算力受限机器人的轻量激光里程计前端：直接使用仅轻度预处理的稠密点云，以「关键帧空间」子图构建 + 自研 NanoGICP 求解器（kd 树与协方差跨模块复用）在低 CPU 占用下实时定位。
- **核对范围：** 读了 arXiv v1（8 页，2021-10-01）与 v3（8 页，RA-L 录用稿）的全部正文（摘要、引言、相关工作、方法 II-A 至 II-F、算法 1、实验 III-A 至 III-C、结论、致谢），逐处核对了 v1 与 v3 的数字（表 I 至 III、各毫秒与 CPU 占比、阈值参数）一致；图 2 看了图本身，其余图表主要读图注与表格文字；参考文献列表仅扫读，未逐条核对被引文献；没有附录或补充材料可读（论文本身无）；官方仓库只读了 README，未读源码。
- **通读：** 已完成

## 版本差异（v1 与 v3）

- v1 结论里写的未来工作是「更紧的 IMU 融合（后端）与增加回环」，v3 改为「更紧的 IMU 融合与运动畸变校正」。
- v1 预处理处写「假设运动畸变可忽略（帧率高）」，v3 改为「不校正运动畸变，因为非刚体变换计算量大」。
- v3 的表 III 在 Alpha / Beta 标题处加了路径长度（Alpha 757.4 m，Beta 631.5 m）。
- v3 摘要新增「Field robotics in perceptually-challenging environments require fast and accurate state estimation」一句，并补充「novel keyframing system」。

## 核心摘录

1. **定位（摘要，I 引言）：** 轻量级前端激光里程计（frontend），面向算力受限平台，使用稠密、最小预处理的点云实时估计位姿；不含后端和回环（结论 IV 只写未来工作为更紧的 IMU 融合与运动畸变校正）。
2. **预处理（II-B）：** 仅两个滤波器：围绕原点 1 m³ 的箱形滤波去除机身点，0.25 m 体素栅格下采样；平均每帧约 10,000 点；不做运动畸变校正，不提取特征；输入为 Ouster OS1（20 Hz）或 Velodyne VLP-16（10 Hz）一类 360 度激光雷达。
3. **双阶段 GICP（II-C，算法 1，图 2）：** 先做 scan-to-scan（S2S）得到相邻帧相对变换，再把它传到世界系作为 scan-to-map（S2M）初值；GICP 协方差用 10 个最近邻估计；S2S 带 RANSAC 外点剔除；可选 IMU 提供旋转先验（II-D），仅陀螺积分，无平移先验；无 IMU 时初值取单位阵。
4. **关键帧子图（II-E，图 3、4）：** 关键帧与其点云存于键值字典；子图由 K 个最近邻关键帧与 L 个凸包关键帧的点云拼接（实验取 10 + 10）；在「关键帧空间」检索而非在点空间按半径检索。
5. **自适应关键帧阈值（II-E2）：** 用点云到原点的中位距离的指数平滑值 m_k（α=0.95，β=0.05，经验选取）把平移阈值取为 10 / 5 / 1 / 0.5 m 四档，旋转阈值固定 30 度。
6. **实现层面（II-F）：** 子图法线 / 协方差由各关键帧已算协方差拼接近似，不逐次重算；kd 树与协方差跨 S2S 与 S2M 两个模块复用（表 I）；NanoGICP = FastGICP + NanoFLANN，两个实例间共享数据结构。
7. **组件实验（III-A，Alpha Course，Urban Circuit，VLP-16，60 分钟，4 核 i7 1.30 GHz）：**
   - 子图：半径法（r = 10 m）74.2 ms / 37.5% CPU；关键帧静态阈值 21.6 ms / 10.2%；自适应 19.1 ms / 9.1%。
   - 复用：全无复用 69.8 ms 且约 10%（表 II 为 9.37%）帧丢失；仅复用 kd 树丢 4.51%；仅复用协方差 0%；全复用 21.9 ms / 9.5% CPU，0% 丢帧。
   - NanoGICP 对齐两帧 100 次平均收敛时间：42.53 ms，对 FastGICP 72.88 ms、PCL GICP 178.24 ms（图 7；所有算法用单位阵先验；未与体素化 FastGICP 比较，理由是其降低精度）。
8. **基准（III-B，表 III，数字取自 LOCUS 论文 [13]）：** Alpha（757.4 m）APE max / mean / std 与 ME rmse：DLO 0.40 / 0.18 / 0.06 / 0.19 m；LOCUS 0.63 / 0.26 / 0.18 / 0.28；LOAM 4.33 / 1.38 / 1.19 / 0.60。Beta（631.5 m）：DLO 0.50 / 0.16 / 0.09 / 0.19；LOCUS 1.20 / 0.58 / 0.39 / 0.48。CPU 核数（max / mean）：DLO 0.92 / 0.62，BLAM 1.14 / 0.93，Cartographer 1.75 / 0.88，LIO-Mapping 1.80 / 1.53，LOAM 1.65 / 1.41，LOCUS 3.39 / 2.72。LIO-SAM 与 LVI-SAM 因标定敏感与输入要求严格而未测。
9. **野外实验（III-C）：** 空中平台 Ouster OS1，Spot 平台 Velodyne VLP-16，VN-100 IMU，Intel NUC7i7DNBE 1.9 GHz；洛杉矶废弃地铁三层 850 m 以上端到端漂移 10 cm（手动遥操作，图 9 说明 856 m）；肯塔基州 Lexington 矿洞自主飞行 348 m 稳定跟踪。这两处为定性 / 单点结果，论文没有给出 10 cm 的真值来源。
10. **背景（I 引言，相关工作）：** 在 DARPA 地下挑战赛中 DLO 是 JPL Team CoSTAR 空中机器人的主要状态估计组件，属于 NASA JPL NeBula 框架（v1 与 v3 引言均写明）。

## 论文自己承认或可见的局限

- 仅前端，无回环、无后端（v1 结论列为未来工作，v3 改写为 IMU 融合与畸变校正）。
- 不做运动畸变校正；IMU 仅用陀螺旋转先验，松耦合，无加速度计与平移先验。
- 基准数字来自 LOCUS 论文，并非作者复现同一环境下各基线；LIO-SAM、LVI-SAM 未参与比较。
- 组件级与基准评测数据集（Urban Circuit Alpha / Beta）均来自同一个地下挑战赛场景族；野外试验未给出外部真值。

## 对 wiki 的映射

- [DLO（Direct LiDAR Odometry）](../../wiki/entities/paper-dlo-direct-lidar-odometry.md)
- [激光雷达里程计融合](../../wiki/methods/lidar-odometry-fusion.md)
