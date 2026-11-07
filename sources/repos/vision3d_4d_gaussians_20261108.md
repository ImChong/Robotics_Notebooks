# 4D Gaussian Splatting — 一手资料归档

> 来源层（官方代码/仓库）；视觉与三维重建一手资料专题，2026-11-08。

- **标题：** 4D Gaussian Splatting
- **类型：** repo
- **链接：** https://github.com/hustvl/4DGaussians
- **入库日期：** 2026-11-08
- **一句话说明：** 动态场景下时空高斯渲染。
- **核对范围：** 已完整阅读作者 README 的环境安装、D-NeRF/HyperNeRF/Neu3D 数据预处理、训练、checkpoint、渲染与 metrics 入口，并核对 master 843d5ac 的 train.py、render.py、scene/gaussian_model.py、scene/deformation.py 关键调用和时间字段；不是全部第三方依赖逐行审计。
- **通读：** 已完成

## 直接一手来源（同一项目不同入口）

- [4D Gaussian Splatting 源码](https://github.com/hustvl/4DGaussians) **（本归档主类型）**
- [4D Gaussian Splatting 原论文](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering_CVPR_2024_paper.html)

## 可执行入口与源码对象

- `train.py`：`scene_reconstruction(..., stage, ...)` 执行 coarse/fine；coarse 针对静态高斯初始化，fine 用动态时间变形。
- `scene/gaussian_model.py`：`GaussianModel` 持有显式高斯参数、`deform_network`、优化器/调度器和稠密化状态。
- `scene/deformation.py`：`Deformation` 使用 `HexPlaneField` 构造查询特征，`deform_network` 包装位置/时变输入与网络参数；实际是否应用 scale/rotation 等分支须据配置审查。
- `gaussian_renderer.render`：融合时间变形与可微高斯渲染，通过 `render.py` 和 `metrics.py` 推理与评测。
- **训练入口**：`python train.py -s data/dnerf/bouncingballs --port 6017 --expname dnerf/bouncingballs --configs arguments/dnerf/bouncingballs.py`；**渲染**：`python render.py --model_path output/dnerf/bouncingballs --skip_train --configs arguments/dnerf/bouncingballs.py`。
- **源码快照**：[`hustvl/4DGaussians@843d5ac`](https://github.com/hustvl/4DGaussians/tree/843d5ac636c37e4b611242287754f3d4ed150144)。

## 方法与边界

动态场景下时空高斯渲染。已核对作者 README 和关键 Python 调用路径，未执行训练、渲染、GPU 测试或第三方依赖完整审计。

## 对 wiki 的映射与待核验事项

- 来源汇总：[视觉三维专题 90 条来源索引](../sites/vision3d_research_primary_sources_2026_11_08.md)。
- 在升格独立 wiki 实体页前，先按论文 ID / 项目页 / 源码查重，避免与既有节点及不同项目同仓实现冲突。
- [x] 论文对应原文详见 [论文档案](../papers/vision3d_4d_gaussians_20261108.md)。
- [x] 读取官方 README 与训练/渲染/高斯和变形模块调用，源码时序图见实体页。
- [x] 新建 [4D-GS](../../wiki/entities/paper-4d-gaussian-splatting-wu.md) 独立节点并标明推理局限、性能条件。
