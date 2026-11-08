# GS-SLAM Modified Gaussian Rasterizer — 官方源码深读

- **标题：** GS-SLAM Modified Gaussian Rasterization (仅公开渲染/梯度内核)
- **类型：** repo
- **链接：** https://github.com/yanchi-3dv/diff-gaussian-rasterization-for-gsslam
- **入库日期：** 2026-11-08
- **核对范围：** 已完整阅读作者 README，并检查 diff_gaussian_rasterization/__init__.py 的 autograd forward/backward、ext.cpp 的 pybind11 导出、rasterize_points.cu 的 CUDA forward/backward 封装；源码快照 `164d243`，不等于完整 3DGS SLAM 系统源码审计。
- **通读：** 已完成

## 唯一官方公开范围

官方项目页面仅发布“Modified Rasterization Code”，**未找到完整可运行的 GS-SLAM 前后端**。本 fork 基于 graphdeco-inria/diff-gaussian-rasterization 改造：
- forward 输出 color/radii/depth/alpha/depth variance；
- backward 支持 Gaussian 参数和 camera quaternion `cam_q_w2c` / translation `cam_t_w2c` 梯度；
- `mapping_mode` 与 `tracking_mode` 为相应梯度计算开关；
- **作者公开限制**：viewmatrix/projmatrix 不对 pose 求导，更新 quaternion/translation 后需手动同步矩阵；pose 的 depth-loss 梯度仍存在已知错误，作者只用 **RGB loss** 更新位姿。

## 代码入口/时序

`GaussianRasterizer.forward(...)` → `_RasterizeGaussians.apply` → `_C.rasterize_gaussians`（pybind11 `ext.cpp`）→ `RasterizeGaussiansCUDA`（`rasterize_points.cu`）→ `CudaRasterizer::Rasterizer::forward` → 输出 RGB/Depth/Alpha；反传 `_C.rasterize_gaussians_backward` → `CudaRasterizer::Rasterizer::backward` → 参数和 camera pose 梯度（仅 `tracking_mode`）。

## 工程边界

README 例子为独立 rasterizer 的 `python setup.py install` / `pip install .`，**不是完整 GS-SLAM 启动命令**。Pose 更新需手工维护矩阵，depth→pose derivative 不稳定，不应以该代码直接宣称复现论文完整 SLAM ATE、mapping FPS。

- [论文深读档案](../papers/vision3d_gs_slam_20261108.md)
- [独立 GS-SLAM 节点](../../wiki/entities/paper-gs-slam-yan.md)
