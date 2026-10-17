# RTISSLAM（Cosys-Lab/RTISSLAM）

- **类型：** GitHub 仓库
- **仓库：** <https://github.com/Cosys-Lab/RTISSLAM>
- **源码状态：** 已公开
- **许可证：** BSD-3-Clause
- **核验日期：** 2026-10-17
- **论文：** <https://arxiv.org/abs/2610.18479>
- **知识节点：** [RTISSLAM](../../wiki/entities/paper-rtisslam.md)

## 仓库说明

纯声呐 SLAM、定位与导航（嵌入式 3D 成像声呐）。目录：`rtissim/`（2D 仿真器）、`rtisslam/`（描述子、占据栅格、声学里程计、定位器）、`rtisnav/`（规划与导航）、`batslam/`（BatSLAM 2.0 位姿图及扩展）、`sondec/`（阵列模型与成像）、`ros/`（ROS 2 Jazzy：`rtis_msgs`、`rtisslam_ros`、`rtisslam_bringup`）、`scripts/`、`tools/`、`video/`、`tests/`、`data/`。

安装：`python3 tools/setup_linux.py` 或手动（venv + `requirements.txt` + colcon）；推荐 Ubuntu 24.04 + NVIDIA GPU。运行：`pytest tests/`、`tools/demo.py --world office`、真机 `ros2 launch rtisslam_bringup real.launch.py`。核验时 1 次提交，README 引用"待补"。

## 相关归档

- [论文摘录](../papers/rtisslam_arxiv_2610_18479.md)
