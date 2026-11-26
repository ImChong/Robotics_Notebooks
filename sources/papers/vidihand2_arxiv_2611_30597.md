# ViDiHand2：融合视频扩散先验与手部几何 | ViDiHand2: Unifying Video Diffusion Priors and Hand Geometry for 3D Hand Motion Reconstruction in the Wild

- **原文：** [arXiv:2611.30597v2](https://arxiv.org/abs/2611.30597v2) · [HTML](https://arxiv.org/html/2611.30597v2) · [PDF](https://arxiv.org/pdf/2611.30597v2)
- **作者：** Yuxi Wang、Yufei Liu 等；首发 2026-11-23，v2 2026-11-24。
- **机构：** 南洋理工大学、上海交通大学、武汉大学、香港中文大学。
- **核对范围：** v2 正文方法、评测、消融、局限及补充材料的数据划分、补充对照、架构和运行时口径；对照[项目页](../sites/vidihand2-github-io.md)和[官方实现](../repos/vidihand.md)，未运行 GPU 模型。
- **通读：** 已完成

| 原文锚点 | 记录 |
|---|---|
| §§3.2–3.4、图 3 | 将绝对深度回归改为相机射线上的几何求解，由 MANO 关节间距、重投影和时间项共同约束腕部轨迹；置信度门控融合射线关节与参数手。 |
| 表 1、补充 B | 六个同域数据集与野外 Gen-HumanEgo；18 个官方实现经同样训练划分重训，另列可获得的官方权重对照。惩罚版指标将漏检计入。 |
| 表 4–8 | 上一段 KV 缓存缓解接缝抖动；野外绝对误差的主要变化来自几何拟合而非直接深度回归。 |
| 补充 F | A100-SXM4 80 GB，400 帧 608×480，57.23 FPS、14.79 GiB 峰值；从已解码帧和内参至关节，不含模型加载、文件 I/O 和相机估计。 |

## 对 Wiki 的映射

- [ViDiHand2 论文节点](../../wiki/entities/paper-vidihand2.md)
