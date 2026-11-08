# GazeVLA-Data — 人类第一视角数据预处理代码

- **类型：** repo / dataset-tools / egocentric / gaze
- **仓库：** https://github.com/lichy2004/GazeVLA-Data
- **论文：** https://arxiv.org/abs/2604.22615
- **项目页：** https://gazevla.github.io/
- **训练代码：** https://github.com/lichy2004/GazeVLA
- **核对范围：** README 中的安装/示例/数据支持清单与外部依赖说明；仅核对公开入口，未下载原始视频或 MANO 受许可文件。
- **核对日期：** 2026-11-08

## 已公开内容

- README 提供 Python 3.11 环境安装、MANO 依赖和数据处理/可视化入口：`python src/dataset/EgoDex.py`，以及 `python examples/vis.py --file_path data/EgoDex/processed/00000000.hdf5`。
- 说明支持人手动作对齐和第一视角数据的二维/三维可视化，包含统一格式处理。
- TODO 中 **EgoDex 与 HOT3D** 勾选已支持；HoloAssist、OAKINK2、TACO、HOI4D、H2O、xperience-10m、VITRA、EgoScale 等仍未勾选，不能混同为已经公开处理全部 13 数据源。
- 依赖各原始数据集的独立获取许可，并需要用户自行同意 MANO 的许可协议。

## 开源边界

这是**处理脚本仓**，不是论文声称的 150M 帧全部数据的再分发包；是否可逐源重建论文训练集需另行验证。README 顶部许可徽章与末尾 CC BY-NC 4.0 表述也需确认。

## 映射

- [论文全文归档](../papers/gazevla_arxiv_2604_22615.md)
- [项目主页归档](../sites/gazevla_project.md)
- [模型训练代码归档](./gazevla_code.md)
- [项目实体](../../wiki/entities/paper-gazevla.md)
