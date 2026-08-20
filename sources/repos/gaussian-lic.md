# Gaussian-LIC / Gaussian-LIC2

> 来源归档

- **标题：** Gaussian-LIC / Gaussian-LIC2
- **类型：** repo
- **链接：** https://github.com/APRIL-ZJU/Gaussian-LIC
- **Stars：** ~600（2026-08）
- **许可证：** GPL-3
- **入库日期：** 2026-08-20
- **一句话说明：** Gaussian-LIC2 官方实现：实时 LiDAR-Inertial-Camera 3D Gaussian Splatting SLAM；Coco-LIC 连续时间里程计前端 + 深度补全 + CUDA 加速增量建图。
- **沉淀到 wiki：** [paper-gaussian-lic2](../../wiki/entities/paper-gaussian-lic2.md)

---

## 核心定位

论文 *Gaussian-LIC2: LiDAR-Inertial-Camera Gaussian Splatting SLAM*（arXiv:2507.04004，IJRR 2026）与会议前作 *Gaussian-LIC*（ICRA 2025）的官方代码仓库。面向 **户外大尺度** 场景的 **实时位姿估计 + 照片级 3D 高斯地图**，同时输出 **RGB 与深度** novel view。

- **项目页：** <https://xingxingzuo.github.io/gaussian_lic2/>
- **论文：** <https://arxiv.org/abs/2507.04004>
- **前作论文：** <https://arxiv.org/pdf/2404.06926>
- **依赖前端：** [Coco-LIC](https://github.com/APRIL-ZJU/Coco-LIC)（连续时间 LIC 里程计）

---

## 运行入口（README 摘要）

| 组件 | 入口 |
|------|------|
| 建图后端 | `roslaunch gaussian_lic fastlivo2.launch`（`~/catkin_gaussian`） |
| 里程计前端 | `roslaunch cocolic odometry.launch config_path:=config/ct_odometry_fastlivo2.yaml`（`~/catkin_coco`） |
| 深度补全 | `ckpt/Large_300.pth` + `setup_spnet.sh` / `export_onnx.sh` / `build_trt.sh` |
| 输出 | `~/catkin_gaussian/src/Gaussian-LIC/result` |

**环境：** Ubuntu 20.04、CUDA 11.7、cuDNN 8.9.7、OpenCV 4.7（CUDA）、LibTorch 2.0.1、TensorRT 8.6.1、ROS catkin。

**支持数据集：** FAST-LIVO / FAST-LIVO2 / R3LIVE / MCD / M2DGR 等 ROS bag。

---

## 待发布（README checklist）

- [ ] 快速后优化
- [ ] 优化版 Coco-LIC
- [ ] Dockerfile
- [ ] 网格工具
- [ ] **Gaussian-LIC2 自采数据集**

---

## 对 wiki 的映射

- 实体页：[paper-gaussian-lic2](../../wiki/entities/paper-gaussian-lic2.md)
- 论文 source：[gaussian_lic2_arxiv_2507_04004.md](../papers/gaussian_lic2_arxiv_2507_04004.md)
- 项目页：[gaussian-lic2.md](../sites/gaussian-lic2.md)
