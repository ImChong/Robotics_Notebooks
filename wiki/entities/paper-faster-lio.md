---
type: entity
project_id: faster-lio
tags:
  - paper
  - lidar
  - lio
  - slam
  - ivox
  - tsinghua
status: complete
updated: 2026-11-11
doi: 10.1109/LRA.2022.3152830
code: https://github.com/gaoxiang12/faster-lio
original:
  - ../../sources/papers/faster-lio_doi_10_1109_lra_2022_3152830.md
related:
  - ./fast-lio.md
  - ./lio-sam.md
  - ./lego-loam.md
  - ./paper-fast-livo.md
  - ./paper-loam-livox.md
  - ./point-lio-unilidar.md
  - ./cartographer.md
  - ../methods/lidar-odometry-fusion.md
  - ../concepts/state-estimation.md
  - ../concepts/lidar-sensing.md
  - ../comparisons/lidar-slam-lio-vio-selection.md
sources:
  - ../../sources/papers/faster-lio_doi_10_1109_lra_2022_3152830.md
summary: "Faster-LIO（Bai 等，IEEE RA-L 2022）：在 FAST-LIO2 上用增量稀疏体素 iVox（哈希 + LRU + 近似 k-NN）替代 ikd-Tree 并做点级并行；论文称相对 FAST-LIO2 约 1.5 到 2 倍 FPS 且精度同级，实际倍数随数据集与 CPU 差异大。"
---

# Faster-LIO（IEEE RA-L 2022）

**Faster-LIO**（*Faster-LIO: Lightweight Tightly Coupled Lidar-Inertial Odometry Using Parallel Sparse Incremental Voxels*，Chunge Bai、Tao Xiao、Yajie Chen、Haoqian Wang、Fang Zhang、Xiang Gao；清华大学与驭势科技 Idriver+；[DOI 10.1109/LRA.2022.3152830](https://doi.org/10.1109/LRA.2022.3152830)；代码 [gaoxiang12/faster-lio](https://github.com/gaoxiang12/faster-lio)）。本次核对未找到该文的 arXiv 编号。

## 一句话定义

**基于 FAST-LIO2 的轻量 LIO：用「哈希稀疏体素 + LRU 淘汰 + 近似 k-NN」的 iVox 取代 ikd-Tree 管理局部地图，并以点级并行加速点到面配准；论文报告固态雷达约 1000 到 2000 Hz、32 线机械雷达 200 Hz 以上，精度与 FAST-LIO2 同级。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| LIO | LiDAR-Inertial Odometry | 激光-惯性里程计 |
| iVox | incremental Voxels | 本文提出的增量稀疏体素地图结构 |
| PHC | Pseudo Hilbert Curve | 伪希尔伯特曲线，iVox 体素内部的可选存储方式 |
| k-NN | k Nearest Neighbors | k 近邻搜索，点到面配准的主要开销之一 |
| ikd-Tree | incremental k-d Tree | FAST-LIO2 使用的增量 k-d 树 |
| IEKF | Iterated Extended Kalman Filter | 迭代扩展卡尔曼滤波，论文里的状态估计器 |
| LRU | Least Recently Used | 最近最少使用缓存淘汰策略，用于删除旧体素 |
| APE / RPE | Absolute / Relative Pose Error | 绝对 / 相对位姿误差，表 II 的精度指标 |
| FPS | Frames Per Second | 每秒处理的激光帧数，论文的速度口径 |

## 为什么重要

- 作者论点（§I）：LIO 不是系统里唯一的模块，算力要与其他模块共享，更快更轻的 LIO 更稳，也可用作离线建图前端。
- 作者论点（§I）：LIO 有 IMU 给的粗略初值，不需要严格 k-NN；k-d 树的构建、平衡、删除是开销，体素没有这些问题。
- 结果：在 [FAST-LIO](./fast-lio.md) 系列已很快的基础上进一步提速，代价是近邻搜索变为近似，对体素尺寸敏感（见局限）。
- 在本库定位：FAST-LIO2 的数据结构与工程层改进，不引入新的状态估计器，不含回环与全局优化。

## 核心原理

作者设计（论文 §III、§IV）：

1. **体素哈希（§III-A）**：点按体素尺寸 s 量化，三个整数坐标分别乘大素数、异或后对哈希表大小取模（式 1）；只存非空体素。
2. **体素内存储**：linear iVox 用向量存点，k-NN 为 O(n)；iVox-PHC 把体素再切成 (2^K)^3 个小立方（文中 K=6），每个存质心，k-NN 走一维有序索引，O(log n)（§III-A、§IV-C）。
3. **近似 k-NN（§III-B）**：只查询点所在体素加邻近 6、18 或 26 个体素，每个体素内取至多 K 个，再合并选最优 K 个；搜索范围有预设上限，不保证严格最近。
4. **插入（§III-C）**：沿用 FAST-LIO2 式体素滤波，新点若比邻域内已有点更靠近体素中心才插入；论文称 0.5 m 的 leaf size 在多数数据集上效果好。
5. **删除（§III-C）**：不做按视场的主动删除，用 LRU 链表加哈希表记录最近访问的体素，超出上限就淘汰最久未用者，插入和删除均为 O(1)。
6. **PHC 近似误差（§IV-B，式 4）**：d_AB ≤ l_AB ≤ √3·d_AB，PHC 找到的最近邻在最坏情况下可能比真实最近邻远约 √3 倍。

```mermaid
flowchart LR
  A[IMU 与激光帧] --> B[预处理 去畸变 降采样]
  B --> C[点级并行 近邻查询]
  M[iVox 局部地图 哈希 加 LRU] --> C
  C --> D[点到面残差 IEKF 迭代更新位姿]
  D --> E[按体素滤波规则 插入新点]
  E --> M
  M --> F[超过容量时淘汰最久未用体素]
```

## 论文结果与关键数字

均出自论文，数字带位置：

| 项 | 内容 | 位置 |
|----|------|------|
| 速度声明 | 固态雷达 1000 到 2000 Hz，32 线机械雷达 200 Hz 以上（摘要）；引言贡献处写固态 1500 Hz 以上 | 摘要、§I |
| 相对 FAST-LIO2 | 作者：典型为 1.5 到 2 倍 FPS，固态和机械雷达数据集均如此 | §V-B |
| 机械雷达增益 | NCLT 与 UTBM 上 IEKF 步骤省约 40% 到 75% 时间，整体 FPS 提升「200% 到 320%」（原文 increased by 200% to 320%） | §V-B |
| 固态雷达增益 | AVIA1、AVIA2 上提升不明显 | §V-B |
| 表 I 的 Spd inc | 9 个序列为 1.52 到 3.49（取两个变体中较快者，见来源归档核算） | 表 I |
| 精度 | 与 FAST-LIO2 同级，典型平移漂移 0.5 到 1.5% | §V-C，表 II |
| 集成 LeGO-LOAM | 建图模块单帧 69.41 ms 降到 51.706 ms，作者说明 NN 搜索不是其瓶颈 | §V-D，表 III |
| 数据结构对比 | 小中型地图上 iVox 的 k-NN 远快于其他结构，但耗时增长更快；追求高召回率时 iVox 不合适 | §V-A，图 6 与图 7 |

## 与综述说法的核对

对象：综述称「Faster-LIO 在 FAST-LIO2 基础上使用 iVox 代替 ikd-tree，可有效降低点云配准时间，不会大幅影响精度，能达到 FAST-LIO2 的 1.5~2 倍速度」。

- **机制与出处**：对得上，且 1.5 到 2 倍与论文 §V-B 的作者原话、仓库 README（about 1.5-2x speed increase）一致。
- **口径是 FPS 之比，不是配准耗时之比**：作者在 §V-B 用 FPS 表述，对比对象是原版 FAST-LIO2 默认参数（局部地图 L=1000 m，降采样 l=0.5 m）。「降低点云配准时间」是综述的因果解读；论文图 9 和图 10 显示节省主要在 IEKF 加 ICP 步骤。
- **「典型」不等于「处处」**：论文表 I 的相对提升在 9 个序列上为 1.52 到 3.49；图 1 柱状图（目测，非原文数字）显示 AMD 平台 AVIA1 与 AVIA2 上只有约 1.05 到 1.15 倍，Intel 平台多数序列在 2 倍以上，NCLT 的 Intel 平台约 3 倍。结论依平台与数据集差别大，综述的 1.5 到 2 倍只是作者的典型概括。
- **提速不全是 iVox**：§V 明说 Faster-LIO 还把 OpenMP 换成 TBB 与 C++17、删掉特征提取、删掉地图 box 删除与检测点缓存、调整残差计算。论文没有做「只换数据结构」的消融，因此 1.5 到 2 倍应归于整套实现，不能全算作 iVox 的功劳（作者对 iVox 的单独对比在 §V-A 的合成数据实验）。
- **「不会大幅影响精度」**：表 II 与作者表述基本一致（互有胜负），但在 UTBM 多数序列的 APE 上略差（如 utbm_2 为 14.48 m 对 12.73 m）。所谓精度同级基于 4 个数据集共 9 个序列，机器与 FAST-LIO2 对齐，无多次运行方差。

## 工程实践

仓库 `gaoxiang12/faster-lio`（核对提交 `8038fba`）：

- 依赖 ROS（Melodic 或 Noetic）、glog、Eigen、PCL、yaml-cpp，C++17 与 TBB；Ubuntu 18.04 需升级 g++ 并用仓库自带 TBB（README）。
- iVox 类型在编译时选择：默认 linear，`-DWITH_IVOX_NODE_TYPE_PHC=ON` 切到 PHC（README）。
- `config/avia.yaml` 关键参数：`ivox_grid_resolution: 0.5`（注释写 default=0.2）、`ivox_nearby_type: 18`（可选 6、18、26）、`filter_size_surf: 0.5`、`max_iteration: 3`；`Options` 里默认容量 1000000 个体素。
- 离线 `run_mapping_offline` 与在线 launch 两种运行方式；结束时打印各步骤平均耗时与平均 FPS（README 示例输出）。
- 本库另有 [Lite3 SLAM 页](./cn-os-lite3-slam.md)使用了该仓库的 faster-lio 子包，可看其源码时序图。
- 源码仅静态读，未运行，未复现论文数字。

## 局限

论文自己承认或可从论文读出：

- 近邻是近似的：作者在 §V-A 写明要精确邻居时 iVox 不是好选择；PHC 最坏误差见式 4。
- 对体素尺寸敏感：README 的 Known Issues 要求不稳定时加大体素；论文称 leaf size 是速度与精度的关键折中参数。
- 地图规模：iVox 的 k-NN 耗时随局部地图点数增长更快（§V-A），作者建议用中小局部地图；论文未在大规模建图场景验证。
- 无回环、无全局优化（§V-C 因此关闭 LIO-SAM 与 LiLi-OM 的回环做比较），只是里程计与局部建图。
- 对其他系统的收益有限：集成到 LeGO-LOAM 仅从 69.41 ms 降到 51.706 ms（表 III）。

我的未验证之处（判断，非作者自述）：

- 无「仅替换数据结构」的消融，见上节；Spd inc 列只给最佳变体，也没有多次运行的方差。
- 精度实验仅 9 个序列且多为车载场景；对手持、足式或强抖动平台的表现论文未涉及，足式机器人上的效果为推测。
- 综述所称 FAST-LIO2 在大场景下 ikd-Tree 平衡开销使其无法实时建图，论文并没有直接测量；论文实际测的是中小局部地图下 iVox 更快。
- 论文与仓库数字有出入：机械雷达速度论文摘要写 200 Hz 以上，README 写 100 Hz 以上；Xeon 型号正文写 5128，图 1 写 5218（疑为笔误）。

## 关联页面

- [FAST-LIO（本库系列页）](./fast-lio.md)：该页只覆盖 FAST-LIO，且只提到 FAST-LIO2 与 ikd-Tree 的名字，没有 iVox 与 Faster-LIO 的内容，故本页单独成节点。
- [LIO-SAM](./lio-sam.md)：论文表 I 与表 II 的对比基线之一。
- [LeGO-LOAM](./lego-loam.md)：论文 §V-D 的集成对象。
- [FAST-LIVO](./paper-fast-livo.md)：同属 FAST-LIO2 一脉的视觉扩展。
- [Livox LOAM](./paper-loam-livox.md)：固态雷达里程计的前作。
- [Point-LIO](./point-lio-unilidar.md)：另一条改进 FAST-LIO 的路线。
- [Cartographer](./cartographer.md)：带回环与全局优化的 SLAM 对照。
- [激光里程计融合](../methods/lidar-odometry-fusion.md)、[状态估计](../concepts/state-estimation.md)、[激光雷达感知](../concepts/lidar-sensing.md)、[LIO/VIO 选型对比](../comparisons/lidar-slam-lio-vio-selection.md)。

## 参考来源

- Bai 等，*Faster-LIO*，IEEE RA-L 7(2)，4861-4868，2022，DOI 10.1109/LRA.2022.3152830；读的是仓库自带的 `doc/faster-lio.pdf`。
- 官方仓库：https://github.com/gaoxiang12/faster-lio（`README.MD`、`config/avia.yaml`、`include/ivox3d/*`、`src/laser_mapping.cc` 节选）。
- 来源归档与通读范围：[faster-lio_doi_10_1109_lra_2022_3152830.md](../../sources/papers/faster-lio_doi_10_1109_lra_2022_3152830.md)。

## 继续阅读

- 论文引用的 FAST-LIO2（Xu 等，IEEE T-RO 2022）与 ikd-Tree（arXiv:2102.10808，Cai 等）：对照被替换的数据结构，本次未读。
- 体素哈希与空间哈希：论文引用 Teschner 等 2003（式 1 的出处）与 Nießner 等 2013。
- 同类体素化 LIO 地图（如自适应体素的 VoxelMap 一类）：本次未核对，不在本页范围。
