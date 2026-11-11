# R3LIVE++（arXiv:2209.03666）

> 来源归档（paper）

- **标题：** R3LIVE++: A Robust, Real-time, Radiance reconstruction package with a tightly-coupled LiDAR-Inertial-Visual state Estimator
- **类型：** paper
- **作者 / 机构：** Jiarong Lin, Fu Zhang（香港大学机械工程系，HKU MARS 实验室）
- **arXiv：** <https://arxiv.org/abs/2209.03666>（v1，2022-09-08，cs.CV）
- **PDF：** <https://arxiv.org/pdf/2209.03666v1>
- **代码：** https://github.com/hku-mars/r3live
- **入库日期：** 2026-11-11
- **一句话说明：** 在 R3LIVE（FAST-LIO2 式 LIO + 光度式 VIO，共享一张辐射度地图）上加入相机光度标定（CRF + 渐晕）与曝光时间在线估计，使地图点存真实辐射度而非像素颜色。
- **核对范围：** 通读 arXiv v1 PDF 正文第 1–8 节（含 Table 1–7、Fig. 1–19、公式 1–26、参考文献列表）；未读：GitHub 仓库里的补充材料 PDF（`supply/r3live_plus_plus_supplementary_material.pdf`，仅见正文引用）、配套 YouTube 视频、R3LIVE-dataset 原始数据、仓库源码（仅静态读了 README，未读核心源码）；正文无附录。图表中的曲线（Fig. 13/14）只读了正文给出的数值与描述，未逐点核对。
- **通读：** 已完成

## 开源状态

- 仓库 hku-mars/r3live 存在，README 为 R3LIVE 时期的文字（标题 R3LIVE，引用列表列的是 R3LIVE / FAST-LIO2 / R2LIVE 等），未单独成段介绍 R3LIVE++（静态读 README，2026-11-11）。论文摘要写明释放代码、硬件设计与数据集。
- 许可（README）：GPLv2，仅允许个人与学术免费使用，商用需联系作者。
- 数据集：R3LIVE-dataset 在 github.com/ziv-lin/r3live_dataset；手持设备设计在 github.com/ziv-lin/rxlive_handheld（README 链接，未逐一访问）。

## 核心摘录（均出自论文，章节为论文章节号）

1. **摘要 / §1：** R3LIVE++ 由 LIO 与 VIO 两个子系统组成，二者紧耦合在同一个 ESIKF 中，增量式共同构建 3D 辐射度地图；LIO 重建几何结构，VIO 恢复辐射度。作者自称是首个能在普通 CPU（无 GPU / 硬件加速）上实时运行的辐射度地图重建框架。
2. **§2.1：** LIO 子系统“largely based on FAST-LIO2”。**§2.3：** 与 LVI-SAM、R2LIVE 等“特征点法”LIVO 不同，R3LIVE++ 在 LIO 与 VIO 中都用直接法；LIO 与 VIO 共享并共同维护同一张辐射度地图。
3. **§3.3 光度模型：** 像素强度 I_i(ρ)=f_i(τ·V(ρ)·γ_i)，f 为各通道相机响应函数（CRF），V 为渐晕因子，τ 为曝光时间，γ 为点辐射度。CRF 与 V 用 [45] 的方法离线标定，曝光时间在线估计。公式 (4) 由像素反算辐射度。
4. **§3.4 状态：** x = (R_I, p_I, v, b_g, b_a, g, R_C, p_C, t_C, 1/τ, φ)，即 IMU 位姿/速度/偏置/重力、相机外参、IMU-相机时间偏移、逆曝光时间、相机内参 (fx,fy,cx,cy)；外参、内参、时间偏移与曝光时间均在线估计。
5. **§4 LIO：** IMU 反向传播去畸变；原始点到 ikd-tree 中最近 5 点拟合平面的点到平面残差（式 7）；ESIKF 迭代更新；新点以 1 cm 分辨率入图，辐射度初始为 0。
6. **§5 VIO：** 先对图像做 CRF/渐晕校正（式 13）。两步：(a) 帧到帧——Lucas-Kanade 光流跟踪上一帧中的地图点，最小化 PnP 重投影误差得到粗估计（式 14–15）；(b) 帧到地图——最小化地图点辐射度 γ_s 与观测辐射度 Φ_s=τ̌·Γ_k(ρ̌) 之差（式 19），残差对曝光时间有非零雅可比，因此曝光时间在这一步被估计。残差只在稀疏的单个像素上构成（不用图像块）。随后对地图点辐射度做贝叶斯更新（式 24–26），并把光照缓变建模为随机游走（式 20）。
7. **§6.1 NCLT（Table 3）：** 25 条序列（共 27 条，剔除 2012-03-17 与 2012-08-04 两条存在 LiDAR 比 IMU 延迟约 100 ms 的时间同步问题的序列），总长约 138 km、33 h 34 min。对比 R2LIVE、LVI-SAM、FAST-LIO2、LIO-SAM 与自身 LIO 子系统。论文关闭了 LIO-SAM / LVI-SAM 的回环；NCLT 无光度标定，故令 V(·)=1、f(·)=1 关闭光度标定模块。表中平均 APE：本系统 8.51 m、R2LIVE 10.58、LVI-SAM 15.03、Our-LIO 10.75、FAST-LIO2 9.59、LIO-SAM 15.39。LIO-SAM 与 LVI-SAM 各有 5 条序列失败（标“—-”），这些序列不计入它们的平均值。
8. **§6.5 鲁棒性（R3LIVE-dataset）：** degenerate seq 00/01 面向楼梯 / 墙面造成 LiDAR 退化，三种仅 LiDAR 的系统（Our-LIO、FAST-LIO2、LIO-SAM）均失败，R3LIVE++ 回到起点，漂移 4.1 cm 与 4.6 cm；degenerate seq 02 为 LiDAR 与视觉纹理同时退化（白墙），终点漂移 1.62° / 4.57 cm（ArUco 标志板提供终点真值）。
9. **§6.6.1 曝光时间（Table 4）：** 与 Tum-cali [44] 比较（唯一已知可在线估计曝光时间的工作），真值取自相机 API；各序列平均误差 R3LIVE++ 0.189–3.460 ms，Tum-cali 0.341–7.082 ms。作者归因：R3LIVE++ 最小化 scan-to-map 辐射度误差而 [44] 只用连续帧。
10. **§6.6.2 辐射度地图（Table 5）：** 地图点颜色回投影到每帧图像后的平均光度误差（像素值量纲，论文未给单位说明）：baseline（以最近一帧给每个 LiDAR 帧上色，[52][53]）38.60，R3LIVE 23.58，R3LIVE++ 18.01；R3LIVE++ 在全部 14 条序列均最低。
11. **§6.7 耗时（Table 6/7）：** i7-9700K + 64 GB，纯 CPU。NCLT：每 LiDAR 帧均值 34.3 ms、每图像 16.6 ms，10 Hz LiDAR + 5 Hz 相机合计每秒 426 ms；R3LIVE-dataset：22.7 ms / 16.2 ms，10 Hz + 15 Hz 合计 470 ms，作者称“快于实时两倍”。
12. **§7 应用：** 以不同曝光时间渲染再合成 HDR；CGAL + openMVS 离线 Delaunay + graph cut 建网格并贴点色纹理；导出 pcd / ply / obj；导入 Unreal Engine，做 AirSim 车辆仿真与游戏 demo。
13. **§8.2 未来工作（作者承认的局限）：** 地图是点云（当前 1 cm 间距），渲染高分辨率图像受点密度限制；提高点密度会增加耗时，也受 LiDAR 原始点密度限制；未来要探索更充分利用高分辨率图像。

## 实验设置（论文 §6.2）

- 手持设备（Fig. 7）：DJI Manifold-2C（i7-8550U，8 GB）、FLIR Blackfly BFS-U3-13Y3C 全局快门相机、Livox AVIA；相机 FoV 82.9°×66.5°，LiDAR FoV 70.4°×77.2°。
- R3LIVE-dataset：HKU 与 HKUST 校园 13 条序列（Table 2 汇总行 Total 为 7354 s / 8447.6 m；正文称 8.4 km、2.4 h），含 3 条退化序列。

## 对 wiki 的映射

- [paper-r3live-plus-plus](../../wiki/entities/paper-r3live-plus-plus.md)

## 与综述文章《激光雷达 SLAM 算法最新综述》的对照（核对记录）

文章（刘铭哲等，公众号「新机器视觉」转载，2026-11-11）关于 R3LIVE / R3LIVE++ 的原话：「……作者在改进版R3LIVE中对视觉系统进行重构，使用光流法代替原本的特征点法进行帧间跟踪，通过最小化帧到地图的光度误差融合视觉数据，并为地图渲染RGB颜色。实验证明，R3LIVE实现目前最先进的多传感器融合算法精度和鲁棒性。R3LIVE++在R3LIVE基础上，考虑了相机光度校准和对相机曝光时间的在线估计，进一步提高了定位和建图的精度。」

- 「在 R3LIVE 基础上加入光度校准与曝光时间在线估计」：与论文 §1 / §2.3 一致（论文列出的扩展点 (1)(2) 即此）。
- 「进一步提高定位和建图精度」：**建图（辐射度误差）有直接对照**（Table 5，R3LIVE 23.58 → R3LIVE++ 18.01）；**定位精度相对 R3LIVE 的提升论文未直接给出**——Table 3 没有 R3LIVE，且 NCLT 上光度标定被关闭；可比较的只有 R3LIVE++ 与其自身 LIO 子系统（8.51 vs 10.75 m）。因此「提高定位精度」一句在本文内是旁证式支持，而非 R3LIVE vs R3LIVE++ 的直接消融。
- 「R3LIVE 视觉系统用光流替代特征点、最小化帧到地图光度误差」：论文 §5 描述的是 R3LIVE++ 的 VIO（光流只用于帧到帧粗估计，帧到地图才是光度/辐射度误差）；R3LIVE 本身的论文（ICRA 2022）不在本次核对范围内，仓库 README 对其描述为“VIO 渲染点的颜色”，未核对其残差细节。
- 「R2LIVE 通过因子图进一步优化」：R3LIVE++ 论文 §2.3 只写 R2LIVE 是 on-manifold 迭代卡尔曼滤波下的紧耦合；未提因子图，本次未读 R2LIVE 原文，故这一句**未核实**。
- 「R3LIVE 实现目前最先进的精度与鲁棒性」：属于对 R3LIVE 论文的转述；R3LIVE++ 论文自己的结论是“在 NCLT 的 25 条序列上总体精度最高”（§1、§8.1），但见 wiki 页「局限」：该“最高”是均值口径，逐序列并非全胜。
