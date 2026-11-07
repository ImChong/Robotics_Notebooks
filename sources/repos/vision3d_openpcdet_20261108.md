# OpenPCDet / PointRCNN / PV-RCNN — 一手资料归档

> 来源层（官方代码/仓库）；视觉与三维重建一手资料专题，2026-11-08。

- **标题：** OpenPCDet / PointRCNN / PV-RCNN
- **类型：** repo
- **链接：** https://github.com/open-mmlab/OpenPCDet
- **入库日期：** 2026-11-08
- **一句话说明：** 点云三维检测框架，包含不同独立模型实现。
- **核对范围：** 完整阅读官方 README 的架构、模型支持/分数据集实验、安装/使用指引，并核对 docs/GETTING_STARTED.md、tools/train.py、tools/demo.py、pcdet/models/detectors/detector3d_template.py、DatasetTemplate 的执行入口，版本 233f849；未运行 GPU 复现。
- **通读：** 已完成

## 直接一手来源（同一项目不同入口）

- [OpenPCDet（PointRCNN/PV-RCNN）](https://github.com/open-mmlab/OpenPCDet) **（本归档主类型）**

## 原始技术资料深读证据

1. 数据集抽象：`pcdet/datasets/dataset.py::DatasetTemplate` 管理 POINT_CLOUD_RANGE、POINT_FEATURE_ENCODING、DATA_AUGMENTOR、DATA_PROCESSOR 和 batch collate；独立的数据集类提供原始点/标注。
2. 检测网络：`pcdet/models/detectors/detector3d_template.py` 的 `build_networks()` 装配 VFE、3D backbone、BEV 映射/2D backbone、PFE、dense head、point head 或 RoI head，不同配置使用不同模块子集。
3. 训练与推理：`tools/train.py` 从 YAML 构造 loader、model、optimizer/scheduler，再调用 `train_utils.train_utils.train_model`；`tools/demo.py` 构造 DemoDataset 后推理并 Open3D 可视化。
4. 统一几何参数：README 定义 3D box 为 `(x,y,z,dx,dy,dz,heading)`；并提供旋转 IoU/NMS、proposal 细化及多卡训练。
5. **独立算法身份**：PointRCNN（原始点前景分割→3D proposal→二阶段精修）与 PV-RCNN（稀疏体素 backbone+体素到关键点聚合+RoI-grid pooling）不能因共用框架而合并为一个论文节点。

## 工程、实验与局限

官方 README 的 KITTI val 中等难度 Car@R11：PointPillar 77.28、SECOND 78.62、PointRCNN 78.70、PV-RCNN 83.61（官方给定训练环境）。这些数值来自**不同模型配置**，是参考复现结果而不是 OpenPCDet 跨所有数据集的单一性能；Waymo 还区分 20%/100% 训练样本，nuScenes 使用不同的 NDS/mAP。

官方流程以 `docs/GETTING_STARTED.md` 为准：整理 KITTI/Waymo 数据→生成 dataset info 和 gt database→选择 `tools/cfgs/kitti_models/*.yaml`→执行 `python tools/train.py --cfg_file tools/cfgs/kitti_models/pv_rcnn.yaml`，再使用 `python tools/test.py` 或 `python tools/demo.py` 核对预测。安装依赖要按官方 `docs/INSTALL.md`；GPU、spconv、PyTorch 和自定义 CUDA 算子需匹配。**没有在本环境运行训练**。

多模型工具箱并不保证所有数据集或所有训练配置可直接互换；不同传感器 LiDAR ring/坐标、类别映射、label 约定与类别范围必须重配。部分历史 README 的 PyTorch/SpConv 版本只是当时支持范围，不应据此宣称 2026 环境开箱即用。Waymo 许可限制权重再分发。常见失败源是 custom dataset 的标定、box yaw、点云预处理和 checkpoint 与 config 不匹配。

## 方法与边界

点云三维检测框架，包含不同独立模型实现。已针对本官方软件项目完整审读原始 README 和核心技术说明，并核对主执行入口；并未独立运行指标复现。

**已核对 README：** 这是包含多种 LiDAR 三维检测方法的工具箱；PointRCNN 与 PV-RCNN 是独立方法，未来建立 wiki 时需按各自论文身份区分，但共享本仓库实现入口。

## 对 wiki 的映射与待核验事项

- 来源汇总：[视觉三维专题 90 条来源索引](../sites/vision3d_research_primary_sources_2026_11_08.md)。
- 在升格独立 wiki 实体页前，先按论文 ID / 项目页 / 源码查重，避免与既有节点及不同项目同仓实现冲突。
- [x] 完整审读官方 README / 使用文档；本归档是官方工具库，不冒充某一篇论文。
- [x] 检查作者提供的主要执行入口、模块、参数与约束；绘制官方代码时序图。
- [x] 对应独立 [OpenPCDet wiki](../../wiki/entities/openpcdet.md) 已新增；健康度由规则和 CI 复核。
