# MonoGS — 一手资料归档

> 来源层（官方代码/仓库）；视觉与三维重建一手资料专题，2026-11-08。

- **标题：** MonoGS
- **类型：** repo
- **链接：** https://github.com/VSLAM-LAB/MonoGS
- **入库日期：** 2026-11-08
- **一句话说明：** 基于三维高斯的单目以及多模式 SLAM。
- **核对范围：** 已通读作者 README 包含 Monocular/RGB-D/Stereo 的安装、测试、数据集、Realsense 实操与评测入口；核对 slam.py、utils/slam_frontend.py、utils/slam_backend.py 前后端代码、可视化多进程队列；未运行。
- **通读：** 已完成

## 直接一手来源（同一项目不同入口）

- [MonoGS 源码](https://github.com/VSLAM-LAB/MonoGS) **（本归档主类型）**

## 官方源码真实入口

- `python slam.py --config configs/mono/tum/fr3_office.yaml --eval`：使用配置加载 TUM 的数据集、GaussianModel、multiprocessing 队列。
- `slam.py::SLAM.__init__` 初始化 `FrontEnd`、`BackEnd`、GUI 等子系统；`run` 启动 backend 进程并运行 frontend 主循环。
- `utils/slam_frontend.py::tracking` 渲染 map、计算 tracking loss、更新 camera pose；`is_keyframe` 按 Gaussian covisibility 确定新关键帧。
- `utils/slam_backend.py::initialize_map/map` 负责地图高斯初始化、渲染损失、densification 和优化并反馈前端。
- 源码 README 说明 `dev.speedup` 是更快的独立分支，不能把 10 FPS branch 结果当作 main 实测。

## 方法与边界

基于三维高斯的单目以及多模式 SLAM。已核对官方 README/主要 C++ CUDA 依赖说明和 Python 前后端执行入口；无 GPU 运行或 benchmark 实测。

**已核对 README：** 官方以 `python slam.py --config configs/mono/tum/fr3_office.yaml` 演示单目模式，另提供 RGB-D 与双目配置；应分别核对相机模型、数据格式与可视化结果。

## 对 wiki 的映射与待核验事项

- 来源汇总：[视觉三维专题 90 条来源索引](../sites/vision3d_research_primary_sources_2026_11_08.md)。
- 在升格独立 wiki 实体页前，先按论文 ID / 项目页 / 源码查重，避免与既有节点及不同项目同仓实现冲突。
- [x] 原论文 §3–4 与附录核对，来源见 paper 归档。
- [x] 前后端、渲染、关键帧和地图优化源码入口核对，见 wiki。
- [x] 独立深读节点 [MonoGS](../../wiki/entities/paper-monogs.md)，保留运行/复现边界。
