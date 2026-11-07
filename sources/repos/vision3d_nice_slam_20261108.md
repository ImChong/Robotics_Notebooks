# NICE-SLAM — 一手资料归档

> 来源层（官方代码/仓库）；视觉与三维重建一手资料专题，2026-11-08。

- **标题：** NICE-SLAM
- **类型：** repo
- **链接：** https://github.com/cvg/nice-slam
- **入库日期：** 2026-11-08
- **一句话说明：** 分层隐式表示的可扩展神经 SLAM。
- **核对范围：** 完整阅读作者官方 README 和论文 https://arxiv.org/html/2112.12130 的方法、实验及限制，并核对主程序/Mapper/Tracker/rendering 代码入口：README：python -W ignore run.py configs/Demo/demo.yaml；run.py 读取 config → src/NICE_SLAM.py 分配共享资源与调度 Tracker、Mapper；src/Tracker.py::optimize_cam_in_batch 采样渲染光线计算 RGB-D loss，src/Mapper.py 管理 keyframe overlap/frustum 与局部网格更新。；未独立运行实测。
- **通读：** 已完成

## 直接一手来源（同一项目不同入口）

- [NICE-SLAM 源码](https://github.com/cvg/nice-slam) **（本归档主类型）**

## 官方源码结构与阅读要点

README：python -W ignore run.py configs/Demo/demo.yaml；run.py 读取 config → src/NICE_SLAM.py 分配共享资源与调度 Tracker、Mapper；src/Tracker.py::optimize_cam_in_batch 采样渲染光线计算 RGB-D loss，src/Mapper.py 管理 keyframe overlap/frustum 与局部网格更新。

粗层 grid 表示墙面与未观测区域的几何概貌，中/细层 grid 优化可见局部细节，另用颜色 grid 做 RGB 重建。Coarse/Mid/Fine 几何采用预训练固定解码 MLP，优化的是局部 feature grid。光线采样和深度/颜色体渲染构造跟踪与建图误差；Tracker 独立优化当前 camera pose，Mapper 优化局部网格和关键帧；大型场景使用可见视锥局部 feature 更新与 keyframe overlap 选取。不是原始 iMAP 的单一全局 MLP。

几何预测有训练集先验及归纳偏置，跨类别泛化不等于保证；光线/体渲染和双设备 Tracking/Mapping 有算力需求；RGB-D 深度噪声和动态物体造成重建误差，论文中传统特征 SLAM 的部分轨迹指标仍优。

## 方法与边界

分层隐式表示的可扩展神经 SLAM。本归档已核对原始 README 与源码主要入口，但没有运行 GPU 的 SLAM 性能复现。

**已核对 README：** 官方以 `python -W ignore run.py configs/Demo/demo.yaml` 运行 demo，使用 `visualizer.py` 可视化；README 特别说明 demo 配置不等于论文复现配置。

## 对 wiki 的映射与待核验事项

- 来源汇总：[视觉三维专题 90 条来源索引](../sites/vision3d_research_primary_sources_2026_11_08.md)。
- 在升格独立 wiki 实体页前，先按论文 ID / 项目页 / 源码查重，避免与既有节点及不同项目同仓实现冲突。
- [x] 已按原始论文逐节核对方法、实验与局限。
- [x] 已核对 README/源码运行入口和关键 Tracking/Mapping 关系。
- [x] [NICE-SLAM 深读实体](../../wiki/entities/paper-nice-slam.md) 包含两个 Mermaid 图，健康度待 CI 验证。
