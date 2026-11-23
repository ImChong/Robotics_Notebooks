# ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multimap SLAM（T-RO 2021 / arXiv:2007.11898）

> 来源归档（ingest · 全文通读补档）

- **标题：** ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multimap SLAM
- **缩写 / 系统：** **ORB-SLAM3**
- **类型：** paper / vslam / vio / multi-map / open-source
- **arXiv：** <https://arxiv.org/abs/2007.11898>（v1 2020-07-22；IEEE Transactions on Robotics, Vol. 37, No. 6, 2021）
- **作者：** Carlos Campos, Richard Elvira, Juan J. Gómez Rodríguez, José M. M. Montiel, Juan D. Tardós
- **机构：** Universidad de Zaragoza（I3A Robotics, Perception and Real-Time Group）
- **代码仓：** <https://github.com/UZ-SLAMLab/ORB_SLAM3>（归档见 [`sources/repos/orb_slam3.md`](../repos/orb_slam3.md)）
- **入库日期：** 2026-11-21
- **原文下载/核对日：** 2026-11-21
- **通读：** 已完成
- **核对范围：** 全文 3608 行（§I–§VIII 全文、图 1–6 系统框图与因子图、Table II EuRoC 单会话、Table III–IV TUM-VI、Table V 多会话 Atlas、Table VI–VII Tracking/Mapping/Loop/Merge 计算耗时表）

---

## 一句话说明

**ORB-SLAM3** 提出首个覆盖 **单目 / 双目 / RGB-D**、**纯视觉 / 视觉-惯性（紧耦合 MAP 估计）** 与 **多地图（Atlas）** 的开源 SLAM 库，通过 **短程（共视局部窗口）、中程（当前地图已建点复用）、长程（DBoW2 回环）与跨地图（Atlas 融合与 Welding BA）四类数据关联**，结合 **最大后验快速 IMU 初始化** 与 **针孔 / Kannala-Brandt 鱼眼统一相机模型**，在 EuRoC 与 TUM-VI 上把室内 MAV/手持精度推进到厘米级乃至 sub-cm 级。

---

## 核心论文摘录

### 1) 系统总架构与三类数据关联（§I–§IV）

- **系统四模块：** Atlas 多地图管理器、Tracking 线程、Local Mapping 线程、Loop & Map Merging 线程。
- **相机模型抽象：** 把针孔与 Kannala-Brandt 鱼眼相机投影与 Jacobians 封装在相机对象中，使外极几何、PnP、BA 对两种镜头透明可用。
- **三种尺度数据关联：**
  - **短程（Short-term）**：最近数帧内匹配，易在探索未回访区域累计漂移；
  - **中程（Mid-term）**：利用局部地图（covisible keyframes）中尚未丢弃的既有三维点，无回环也能大幅抑制漂移；
  - **长程与跨地图（Long-term & Multi-map）**：重定位、回环与多会话 Atlas 合并时建立跨时间/跨会话匹配。

### 2) 快速 MAP 视觉–惯性初始化（§V）

- **三步初始化：**
  1. **纯视觉 SfM**：仅单目或双目运行约 2 s（200 Hz IMU 采样下积累初始关键帧轨迹），确定初步尺度（纯单目为 up-to-scale）。
  2. **纯惯性 MAP 优化**：固定纯视觉位姿，对重力方向 $R_{wg}$、陀螺/加速度偏置、速度与尺度 $s$（若为单目）进行联合 MAP 优化（带 IMU 零偏高斯先验）。
  3. **视惯联合 MAP BA**：联合优化相机位姿、路标点与惯性状态；通过高/低阶可观性检验后在数秒内完成尺度与重力收敛（可在后续 5–15 s 二次优化精修尺度和偏置）。

### 3) 回环与 Atlas 多地图融合（§VI）

- **改进地点识别：** 候选帧不再仅以单帧 DBoW2 最高分判定，而是检验 **三张共视关键帧在局部地图点上的三维重投影一致性**（几何一致后再估计 $Sim(3)/SE(3)$ 变换），显著提升召回率同时避免误闭合（地点识别仅约 ~10 ms/keyframe）。
- **Welding BA 与本质图优化：** 当 Active Map $M_a$ 与存储地图 $M_m$ 发生重合时，对齐坐标并组建 **Welding Window**（含重叠区共视关键帧及 IMU 预积分边，固定 $M_m$ 窗外边界关键帧以消去规范自由度），做局部 Welding BA 后跑全局 Essential Graph 位姿图优化传播修正。

### 4) 关键实验指标（§VII / Table II–VII）

- **EuRoC 单会话与 V202 消融（Table II & VI）：** ORB-SLAM3 在 **Stereo-Inertial** 配置下 11 个序列全通，平均 RMS ATE **0.035 m**、尺度误差 **0.6%**；其中 **Table VI（EuRoC V202，752×480@20Hz）** 上 Stereo-Inertial RMS ATE **0.014 m**（Tracking 平均 33.05±9.29 ms，Mapping 平均 196.61±54.52 ms），Monocular-Inertial **0.021 m**、Stereo **0.028 m**、Monocular **0.029 m**。多地图机制使 V103（单目）与 V203（双目）等 ORB-SLAM2 丢失序列也可恢复运行。
- **TUM-VI Room 序列（Table IV，AR/VR 代表小场景）：**
  - Mono **0.039 m**、Stereo **0.068 m**、Mono-Inertial **0.011 m**、Stereo-Inertial **0.009 m**（sub-centimeter）。
- **TUM-VI 长户外（Table III）：** 无回环远点（>20 m 天空特征若不裁剪易引入漂移，实验中户外序列丢弃 >20 m 点）下仍存在 10–70 m 尺度/偏置漂移；结合室内走廊多会话合并后可把轨迹漂移大幅压低。
- **典型失败模式（§VIII）：** 纯低纹理环境提取不到足够 ORB 描述子；纯旋转探索无法初始化单目/单目惯性深度；无滚转/俯仰角变化的平面匀速运动（如平坦路面小车）会导致 IMU 参数可观性不足而难以完成惯性初始化。

---

## 开源状态（步骤 2.5）

- **仓库：** [`UZ-SLAMLab/ORB_SLAM3`](https://github.com/UZ-SLAMLab/ORB_SLAM3)（GPL-3.0）已完整开源（C++ 库 + EuRoC/TUM-VI/KITTI 示例 + ROS/ROS1 包装节点），归档见 [`sources/repos/orb_slam3.md`](../repos/orb_slam3.md)。

---

## 对 wiki 的映射

- 主实体页：[`wiki/entities/orb-slam3.md`](../../wiki/entities/orb-slam3.md)
