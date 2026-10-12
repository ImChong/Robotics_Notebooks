# RAGNAROK 论文来源归档

## 元数据

- **题目：** RAGNAROK: Radar-Aided Gravity-Normalized Alignment for Robust Open Keyframe-based Radar-Visual-Kinematic-Inertial SLAM
- **作者：** Hanjun Kim, Chiyun Noh, Sangwoo Jung, Jaehyung Jung, Simon Boche, Cedric Le Gentil, Stefan Leutenegger, Ayoung Kim（通讯：Ayoung Kim）
- **单位：** 首尔大学机械工程系 RPM Lab（Kim/Noh/Jung/Kim）；慕尼黑工业大学 Mobile Robotics Lab 与 MIRMI（Jaehyung Jung、Boche）；苏黎世联邦理工 Mobile Robotics Lab（Le Gentil、Leutenegger）
- **预印本：** https://arxiv.org/abs/2610.11531
- **HTML：** https://arxiv.org/html/2610.11531v1
- **PDF：** https://arxiv.org/pdf/2610.11531
- **版本：** v1，2026-10-08，cs.RO；GitHub README 标注 IEEE RA-L 2026（2026-10-07 录用）
- **项目页：** https://ragnarok-rvki-slam.github.io/RAGNAROK/（归档见 [sources/sites/ragnarok-rvki-slam.md](../sites/ragnarok-rvki-slam.md)）
- **代码：** https://github.com/hanjun815/RAGNAROK（归档见 [sources/repos/ragnarok.md](../repos/ragnarok.md)）
- **数据集：** 通过 Google Form 申请下载（README Dataset 一节）

## 论文要解决的问题

腿式机器人在粗糙地形、湿滑地面、陡楼梯、暗光和强眩光下需要可靠的状态估计。纯本体感知（腿运动学 + IMU）在打滑与滚动接触时速度估计偏差大；近年的方法加入毫米波雷达多普勒速度，但雷达的偏航可观性有限、仍会漂移；而在恶劣条件下把雷达、相机、腿运动学和 IMU 一起融合的系统此前研究很少。论文称 RAGNAROK 是首个面向腿式机器人的雷达–视觉–运动学–惯性（RVKI）SLAM。

## 方法与训练信息

这是一套经典优化式 SLAM，没有网络训练，仅复用 OKVIS2-X 自带的深度/分割模型。整体结构为「雷达辅助本体估计器（B 样条）→ 在相机关键帧时刻取相对运动约束 → 基于 OKVIS2-X 的外部感知视觉惯性后端」。

1. **滑移/滚动接触感知腿速因子：** 支撑腿接触点 p_C = p_F − r·n（r 为足端球半径、n 为融合后的接触法向），接触雅可比 J_c = J_p + ⌊r·n⌋× J_ω；协方差按支撑时长、关节力矩不连续性和腿间不一致性膨胀；在融合接触法线的切平面内嵌入 2D 滑移偏置。
2. **运动学感知雷达因子：** 两台正交安装的雷达（一台朝前、一台绕前向轴 90°翻转）提供多普勒残差 r_R = −dᵀRᵀ(v + ω×p) − ṽ；多普勒方差 σ_d² = σ_vd² + v_Rᵀ Σ_d v_R，把载体运动与视轴不确定性计入权重；用 Cauchy 鲁棒核替代 RANSAC 去动态外点。
3. **在线外参标定：** 对雷达–IMU、机体–IMU 外参设置带指示项的先验残差。
4. **空间特征感知图像增强：** AGCWD 与 CLAHE 输出按熵和均值加权融合，参数由贝叶斯优化在线调节，目标效用（SFA-EWG）奖励梯度、熵、BRISK 特征数与空间均匀性；仅当原图效用变化超过阈值时才触发 BO。
5. **重力对齐视觉前端：** BRISK 描述子用估计的重力方向定向；近静止区间抑制关键帧；与本体相对位姿对极几何不一致的匹配被拒绝。
6. **自适应加权：** 视觉质量分数 Q_i 把本体权重缩放为 α_i = 1/max(Q_i, 0.1)；视觉回退标志触发 γ_P^δ 与 γ_E^(−δ) 缩放。
7. **回环：** DBoW2 检索 + 极坐标雷达 BEV 图像的 FFT 相关分数（受 SHeRLoc 启发），候选用 3D-2D RANSAC 验证。
8. **优化图：** 重投影、IMU、位姿图、地图对齐与本体残差联合优化；预测深度融合进子图用于占据建图。

平台：Boston Dynamics Spot；传感器（README）：RealSense D455 双目 15 Hz、2×FLIR Blackfly S 单目 15 Hz、MicroStrain 3DM-GV7-AHRS IMU 100 Hz、2×DesignCore RS-1843AOPU 雷达 15 Hz、Ouster OS1-32、腿运动学 150 Hz；真值由 Leica RTC360 地面激光扫描地图 + PALoc + ICP 精化。计时硬件 Intel i7-14700 + RTX 4080 16 GB。

## 论文报告的结果

指标为 APE RMSE（APEt 单位 m，APEr 单位度），取 5 次运行中位数。

- **数据：** 自采 15 条序列（暗、昏暗、眩光、室外、室内、楼梯、自相似、高程变化），以及公开的 Co-RaL、GaRLILEO（后两者无相机，仅评测 RKI 本体部分）。
- **基线：** x-RIO、OpenVINS、OKVIS2-X、DRIFT、Holistic Fusion、RRxIO、RadVIO、Pronto、MUSE、Co-RaL、GaRLILEO。
- **室外平均（Table III）APEt/APEr：** x-RIO 6.915/11.884；DRIFT 5.909/7.909；Holistic Fusion 9.800/19.233；Co-RaL 2.624/4.708；GaRLILEO 1.445/3.089；RAGNAROK-VKI 1.651/2.058；RAGNAROK-RKI 1.079/2.216；RAGNAROK-Odom 0.481/1.550；RAGNAROK-SLAM 0.387/1.457。
- **室内与混合平均（Table VII）：** RAGNAROK-Odom 0.276/2.547，RAGNAROK-SLAM 0.265/2.589；表中平均 APEt 最强的基线为 Pronto + Radar（2.931）。
- **公开数据集（仅 RKI）：** Co-RaL 平均 APEt 1.541（GaRLILEO 2.828，Co-RaL 2.641）；GaRLILEO 数据集平均 APEt 0.488（GaRLILEO 0.575）。
- **消融（Table VI，TerraceLoop/Overpass/Atrium 平均 APEt/APEr）：** 完整 0.272/1.817；去掉滑移偏置 0.282/1.864；去掉图像增强 0.675/2.608；去掉重力对齐 BRISK 0.302/1.898；去掉运动不一致拒绝 0.329/1.980；去掉自适应加权 2.220/8.354；去掉双雷达 0.288/1.851；去掉在线标定 0.353/1.935。论文据此称图像增强带来 59.7% 的平均 APEt 降低、自适应加权带来 87.7%。Co-RaL 上去掉滚动接触建模 / 运动学感知协方差，平均 APEt 由 1.541 升至 1.576 / 1.574。
- **运行时间：** 本体优化 7.3 ms（Terrace）/ 8.2 ms（FloorGyre）；图像增强 1.1 / 0.8 ms；回环优化 110.7 / 589.3 ms（异步）；整帧 RAGNAROK-SLAM 49.5 / 64.8 ms；Terrace 端到端输出延迟 105.5 ms，15 Hz 下丢 1 帧。
- 项目页还写“平均 RMSE ATE 由 1.172 m 降至 0.379 m（−67.7%）”，这一汇总口径在论文表格里未逐项复核，引用时以论文表格为准。

## 复现与解释边界

- 论文没有单列“局限”一节；下列为读表所见：视觉融合并不总是更好（RKI 在部分暗/眩光序列上与 Odom 相当或更好）；大回环代价高（FloorGyre 589.3 ms）；权重参数 w_τ、w_e、w_d、κ 为经验设定；只在 Spot 一个平台上评测，真值依赖 TLS 先验地图；部分基线表格存在缺项。
- 代码在 2026-10-08 首发（README News），需要 Docker + CUDA 12.1 + ROS 2 Humble；mvs-model.pt（279 MB）首次构建时从 OKVIS2-X 服务器下载。
- 项目页文字仍写“code and dataset will be released upon acceptance”，与已上线的仓库不一致，以仓库实际内容为准。
- 仓库许可 BSD 3-Clause；arXiv 论文本身 CC BY-NC-ND 4.0。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-ragnarok-radar-visual-kinematic-inertial-slam.md
- 汇总页：wiki/overview/frontier-humanoid-quadruped-2026-10-05-10-09.md
- 概念入口：wiki/concepts/state-estimation.md
- 任务入口：wiki/tasks/locomotion.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.11531>
- arXiv HTML：<https://arxiv.org/html/2610.11531v1>
- 代码仓库：<https://github.com/hanjun815/RAGNAROK>
- 项目页：<https://ragnarok-rvki-slam.github.io/RAGNAROK/>
