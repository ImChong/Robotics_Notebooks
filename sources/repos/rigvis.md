# RigVis：Stereo-Inertial SLAM in C++（官方仓库原文与运行入口核对）

> 来源归档（repo / 官方第一手资料）

- **项目：** RigVis
- **作者：** Ekrem Serdar Öztürk（GitHub `ekrrems`）
- **类型：** 官方开源代码仓库及技术文档
- **代码：** https://github.com/ekrrems/Stereo-inertial-slam
- **原文 README：** https://github.com/ekrrems/Stereo-inertial-slam/blob/main/README.md
- **核对版本：** `3dee4bc0db261bdeb069d9b0407f9c0a2e5e9839`（仓库 `main`，2026-11-22 获取）
- **许可证：** MIT（仓库 `LICENSE` 与 README）
- **入库日期：** 2026-11-22
- **核对范围：** 通读官方 README、`CMakeLists.txt`、`config/rigvis.example.yaml` 和运行入口 `src/main.cpp`；核对 `src/Dataloader.cpp`、`src/StereoFrontend.cpp`（processFrame、dense ICP 调度、关键帧）、`src/VisualInertialFusion.cpp`、`src/Estimator.cpp`、`src/LocalBundleAdjuster.cpp`（ICP 残差）、`src/LoopClosing.cpp`、`src/PoseGraph.cpp`、`src/DenseMapper.cpp`、`src/IcpAligner.cpp`、`src/MapExport.cpp`、`src/Evaluation.cpp` 和 `tools/rigvis_evaluate.cpp` 对应运行流程及主要调用关系。**未下载 KITTI、未编译运行、未实测复现、未逐行审计其余文件。**
- **通读：** 已完成
- **开源状态：** **已开源**；存在 `rigvis_dataset_runner` 和 `rigvis_evaluate` 可执行入口，当前环境未验证可编译通过
- **对应知识节点：** [RigVis](../../wiki/entities/rigvis.md)

## 一手技术核对

1. **前端**：`StereoFrontend::processFrame` 的 ORB、栅格分布、双目行带对应、匀速预测、local-map PnP 与上一帧双目 PnP 回退；按跟踪质量插入关键帧。
2. **后端**：`runLocalBundleAdjustmentOnNewKeyframe` → `runLocalBundleAdjustment`（mono/stereo 残差、Huber、卡方筛外点）。`detectAndCloseLoop` 先按空间距离/视角找候选，再 PnP 验证和连续一致性确认，`correctLoop` 执行 SE(3) 位姿图、重复地图点融合和全局 BA。
3. **回环阈值**：`include/StereoFrontend.hpp` 中 `loopSearchRadiusMeters_ = 50.0`（候选搜索半径），`loopMinInliers_ = 60`，`loopMinKeyframeGap_ = 100`。此前用户摘要“1 米阈值”未找到该定义，不能用作当前代码事实；“8 次回环”也需运行日志证明。
4. **视觉惯性**：`include/VisualInertialFusion.hpp` 明确写 **loosely coupled**。`src/main.cpp` 每帧先 `frontend.process`，后将 VO 相邻帧位姿作为 ESKF 观测；IMU 传播与 ESKF 更新不会反馈视觉前端跟踪或进入 BA 惯性因子。18 维误差状态、相对位姿门限与 2 s 初始化在源码可追溯；**紧耦合预积分因子列在 README Roadmap，尚未实现为当前 BA**。
5. **稠密**：`computeDenseCloud` 用 OpenCV StereoSGBM 生成逐关键帧相机坐标点云，输出阶段将当前关键帧位姿应用到局部点云，并以 0.10 m 体素融合。默认深度范围 1–20 m（README），`map_dense.ply` / `map_sparse.ply` 导出。
6. **ICP 实验开关**：`config/rigvis.example.yaml` 显式 `dense_icp: 0`，默认 **关闭**；开启后异步 `launchDenseIcp`，`collectFinishedIcp` 对弱跟踪帧保留有效约束；`runLocalBundleAdjustment` 将相对位姿因子送入 Ceres BA。**不是默认始终进入 SE(3) 回环位姿图**。
7. **无 ROS 依赖**：`CMakeLists.txt` 依赖 Eigen3/OpenCV/Ceres，官方主要构建和运行示例不依赖 ROS；ROS 2 接口位于未来规划中。

## 实验与可复现性边界

- 官方 README：KITTI raw `2011_10_03_drive_0027`（对应 KITTI Odometry 00），4544 帧，约 3.7 km。
- 作者报告：`t_rel=1.20%`、`r_rel=0.50°/100m`、`ATE(RMSE)=2.81m`、逐帧 RPE = `0.057m / 0.100°`。
- `src/Evaluation.cpp::evaluateTrajectory`：ATE 采用 `Eigen::umeyama(..., false)` **仅刚体无尺度**对齐；`t_rel/r_rel` 按 KITTI 100–800 m 子轨迹计算，RPE 对相邻帧计算；真值由 KITTI OXTS 转换。
- README 中 ORB-SLAM2 指标仅是**作者引用论文作参考**；本次**未运行 KITTI 数据**，不宣称复现实测或对标公平性。
- 用户摘要提到的 `约 570 万点` 和 `8 回环` **未见 README 的稳定评测表**，当前不视为可独立复核结果。

## 编译运行（按 README，未执行）

```bash
git clone https://github.com/ekrrems/Stereo-inertial-slam.git
cd Stereo-inertial-slam
mkdir build && cd build
cmake ..
make -j
# 在仓库根目录复制 config/rigvis.example.yaml 为 config/rigvis.yaml
# 设置 KITTI *_sync 的 dataset_path，以及可选 100Hz *_extract 的 imu_path
./rigvis_dataset_runner
./rigvis_evaluate trajectory_final.txt /path/to/2011_10_03_drive_0027_sync
```

注意：README 要求 CMake ≥3.30、C++17、Eigen3、OpenCV、Ceres。其 CMake 默认指定 OpenCV 5 的模块名，OpenCV 4 需按 README 改为 `opencv_features2d` / `opencv_calib3d`。

## 已阅读源码入口与运行时调用

- `src/main.cpp`：`DataLoader` → 按时间戳分派 `ImuSample` 或 `StereoFrame`。
- IMU 事件：`VisualInertialFusion::addImu` → `Estimator::imuPropagate`。
- 双目事件：`StereoFrontend::process` → `processFrame` → ORB/立体匹配/局部跟踪/PnP → 新关键帧时局部 BA、回环、可选 SGBM/ICP；然后 `VisualInertialFusion::addCameraFrame` 调用相对位姿 ESKF 更新。
- 收尾：`StereoFrontend::finalTrajectory` 重建经回环校正轨迹；`MapExport` 输出点云/相机/关键帧 CSV；`Evaluation` 写出在线 VO、VIO、最终轨迹的 KITTI/ATE/RPE 评价。
- `tools/rigvis_evaluate.cpp` 独立载入保存轨迹、OXTS 真值并调用 `evaluateTrajectory`。

## 注意事项

“通读：已完成”仅指官方 README/运行入口及上述核心实现可核查范围，**不代表已完成基准复现实验或全部源文件的逐行审计**。如修改默认配置、实验指标或源码，应重新对照该归档记录的 commit 与实际代码。
