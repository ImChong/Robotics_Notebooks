# gaussian_lic2_arxiv_2507_04004

> 来源归档（ingest）

- **标题：** Gaussian-LIC2: LiDAR-Inertial-Camera Gaussian Splatting SLAM
- **短名：** Gaussian-LIC2
- **类型：** paper
- **来源：** arXiv abs / PDF
- **原始链接：**
  - <https://arxiv.org/abs/2507.04004>
  - <https://arxiv.org/pdf/2507.04004>
- **项目页：** <https://xingxingzuo.github.io/gaussian_lic2/> — 归档见 [`sources/sites/gaussian-lic2.md`](../sites/gaussian-lic2.md)
- **视频：** <https://www.youtube.com/watch?v=SkPnpuCfh88>
- **作者：** Xiaolei Lang*, Jiajun Lv*, Kai Tang, Laijian Li, Jianxin Huang, Lina Liu, Yong Liu†, Xingxing Zuo†（* 同等贡献；† 通讯）
- **机构：** 浙江大学控制科学与工程学院 · 穆罕默德·本·扎耶德人工智能大学（MBZUAI）机器人系
- **版本：** arXiv:2507.04004；会议前作 Gaussian-LIC（ICRA 2025, arXiv:2404.06926）；期刊版 Gaussian-LIC2 被 **IJRR 2026** 接收
- **入库日期：** 2026-08-20
- **一句话说明：** 首个在 **实时** 约束下同时兼顾 **照片级 RGB/深度渲染** 与 **几何精度** 的 LiDAR-Inertial-Camera **3D Gaussian Splatting SLAM**：连续时间紧耦合里程计 + 稀疏深度补全初始化 + LiDAR 深度监督建图 + 高斯地图光度反馈增强退化跟踪。

## 核心摘录

### 1) 问题与动机
- 辐射场 SLAM（NeRF / 3DGS）追求 **位姿 + 照片级地图**，但户外无界场景面临剧烈运动、光照变化、纹理缺失。
- 现有 LiDAR-Inertial-Camera 3DGS-SLAM（Gaussian-LIC、GS-LIVM、LVI-GS、GS-LIVO 等）多 **仅从 LiDAR 点初始化高斯**，在 **稀疏 LiDAR / LiDAR 盲区** 欠重建；ADC 在增量 SLAM 中时效不足。
- 多数方法 **重视觉轻几何**，深度渲染质量不足；**里程计与建图解耦**，未探索增量高斯地图对跟踪的反馈。
- 缺乏支持 **序列内 / 序列外** RGB 与深度 novel view 联合评测的 LIC 数据集。

### 2) 方法要点
1. **连续时间紧耦合 LIC 里程计（前端）：** 基于 B-spline 连续轨迹，在因子图中紧融合 LiDAR、IMU 与两种可选相机因子（光流跟踪 / **高斯地图光度约束**），实现实时位姿估计。
2. **轻量零样本深度补全：** 融合 RGB 外观与稀疏 LiDAR 深度（SPNet 类网络，TensorRT 部署），为 LiDAR 未覆盖像素生成稠密深度，支撑盲区高斯初始化。
3. **增量 3DGS 建图（后端）：** 用精确 LiDAR 深度监督高斯优化；CUDA 加速深度光栅化（tile culling、per-Gaussian、Sparse Adam、分离 SH 等）；按渲染不透明度缩放深度以缓解增量插入导致的深度低估。
4. **高斯地图 → 里程计反馈：** 将增量高斯地图的光度约束纳入连续时间优化，在 LiDAR 退化且纹理缺失时提升跟踪鲁棒性。
5. **下游扩展：** 连续时间轨迹 + 3DGS 支持 **视频帧插值** 与 **快速网格提取**。

### 3) 实验（论文报告摘要）
| 基准 / 场景 | 指标 | 对照 / 前作 | Gaussian-LIC2 | 读法 |
|-------------|------|-------------|---------------|------|
| Bell_Tower_01（150 s，自估位姿） | PSNR ↑ / Depth-L1 (m) ↓ | Gaussian-LIC 25.36 / 0.93 | **25.51 / 0.27** | Depth-L1 相对前作约 **-71%**；论文摘要称跨方法 **Depth L1 -62%** |
| Bell_Tower_01 运行时 | 处理时间 (s)，RTX 4090 | Gaussian-LIC 102 | **90** | 实时照片级建图 |
| 自采数据集（GT 位姿，in-seq NVS） | PSNR | GS-LIVM ~17–23 | **21.78–27.63** | 多序列领先 MM3DGS-SLAM / GS-LIVM |
| FAST-LIVO2 退化序列 | 起终点漂移 | FAST-LIVO2 / LIO-only | **Gaussian-LIC2 (c1)** 厘米级 | 高斯光度约束助 LiDAR 退化场景回环 |
| HKisland03 / HKairport03 | 轨迹长度 | — | **1.95 km / 2.1 km** | 大尺度户外实时定位与渲染 |
| 前端耗时（每 0.1 s LIC 扫描） | ms | — | **41.97 (4090) / 55.03 (3090)** | 跟踪前端 < 0.1 s，满足实时 |

- **消融：** 深度补全对 LiDAR 盲区初始化关键；LiDAR 深度监督与 opacity-scaled depth 显著改善几何；高斯光度因子在 LiDAR 退化场景降低漂移。
- **局限：** 依赖 NVIDIA GPU + CUDA 11.7 + TensorRT + LibTorch 重依赖链；自采评测数据集与 Docker **待发布**（README checklist）。

### 4) 开源核查（步骤 2.5）
- **项目页：** 列 Paper、Pipeline、Demo 视频、BibTeX；链到 GitHub **Code**。
- **GitHub：** [`APRIL-ZJU/Gaussian-LIC`](https://github.com/APRIL-ZJU/Gaussian-LIC) — **已开源**（~600 stars，C++，GPL-3）；含 `roslaunch` 入口、`config/`、`ckpt/`（SPNet 权重需 Google Drive 下载 + TensorRT 导出脚本）。
- **运行栈：** Ubuntu 20.04 + ROS catkin；**Coco-LIC** 作连续时间里程计前端，**Gaussian-LIC** 作建图后端；支持 FAST-LIVO / FAST-LIVO2 / R3LIVE / MCD / M2DGR 等 bag。
- **数据集：** 论文自采 LIC 数据集（GT 位姿、深度、外推轨迹）— README checklist **待发布**。
- **结论：** **已开源**（训练/推理/部署代码 + 深度补全权重）；自采评测集 **待发布**。

## 对 wiki 的映射

- 升格 [Gaussian-LIC2 论文实体](../../wiki/entities/paper-gaussian-lic2.md)
- 交叉 [导航·SLAM 栈](../../wiki/overview/navigation-slam-autonomy-stack.md)、[LiDAR/VIO 选型](../../wiki/comparisons/lidar-slam-lio-vio-selection.md)、[State Estimation](../../wiki/concepts/state-estimation.md)

## 当前提炼状态

- [x] 摘要 + 方法 + 实验表 + 开源边界
- [x] wiki 实体页与交叉引用
- [x] `sources/sites/` + `sources/repos/`
