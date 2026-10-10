# SAM-V（gong208/SAM-V）

> 来源归档（ingest）

- **仓库：** <https://github.com/gong208/SAM-V>
- **类型：** repo / multi-view segmentation / 3D perception / robotics
- **论文：** [SAM-V: Geometry-Aware Segment Anything for Multi-View Instance Segmentation（arXiv:2609.25490）](../papers/sam_v_arxiv_2609_25490.md)
- **作者：** Jiangshan Gong、Yuqun Wu、Qiqian Fu、Yao Xiao、Chuhang Zou、Shenlong Wang、Derek Hoiem
- **代码许可证：** Apache-2.0（仓库代码；第三方依赖与模型权重另有许可）
- **预训练权重：** [Frank-Gong123/SAM-V](https://huggingface.co/Frank-Gong123/SAM-V)，Stage 1 / Stage 2 checkpoint，CC BY-NC 4.0
- **数据：** 仓库不分发 Hypersim、ScanNet++、ScanNet 或 IGGT benchmark；需从数据维护方获取并遵守各自条款。
- **入库日期：** 2026-10-10
- **一句话说明：** 将 VGGT 多视角几何特征与 SAM 的图像/提示编码融合，在同一前向中输出跨视角一致的目标掩码，并提供训练、评测与 Web demo。

## 开源状态与许可边界

**源码与运行入口已公开**：仓库含模型、训练、数据预处理、基准脚本和可选浏览器 demo；README 给出安装、数据准备、Stage 1/2 训练、推理和复现实验命令。SAM-V 自身 checkpoint 可从 Hugging Face 下载。

但“仓库 Apache-2.0”不等于整套管线可任意商业使用：

- **checkpoint 单独为 CC BY-NC 4.0**。
- **VGGT 是必需依赖**，其 Meta research-materials agreement 不是 OSI 许可；README/NOTICE 提醒商业使用前必须审阅其许可与 AUP。
- 数据集不随仓库分发，Hypersim、ScanNet++、ScanNet 与 IGGT benchmark 各有独立条款。
- PanSt3R baseline 使用 NAVER 非商业许可；复现该 baseline 时需另行处理。
- 仓库代码许可证与依赖、权重、数据的许可必须分开判断。

## 仓库入口

| 入口 | 用途 |
|---|---|
| model/sam_vggt_model.py | SamVGGT 模型：SAM 与 VGGT 特征融合、提示融合及 SAM mask decoder |
| training/trainer.py | YAML 驱动的训练入口；Stage 1 / Stage 2 分别使用对应配置 |
| preprocessing/ | instance id、帧存在性、相机姿态及可选离线 SAM embedding 预处理 |
| masks/automatic_mask_generator.py | 从逐帧 proposals 扩展到 every-object 推理 |
| masks/prompt_sampling.py | 在 proposal 内采样点提示 |
| benchmarks/README.md | Table 1 / Table 2 的官方评测入口与指标约定 |
| demos/web/app.py | 可选 FastAPI 浏览器 demo：上传同一场景多帧、点击提示、返回跨视角 mask |
| tools/export_release_checkpoint.py | 导出发布 checkpoint |

## 复现提示

- 运行环境：README 钉定 Python 3.10、Linux/CUDA 12.1、PyTorch 2.3.1。
- 对 SAM-V 主流程，README 建议只初始化 vggt 与 sam-hq 子模块；递归拉取所有 baseline 子模块会额外下载多个 GB。
- frozen SAM ViT-H 与 VGGT-1B 基础权重需要另外下载；SAM-V checkpoint 只包含训练得到的模块，不包含这两个基础模型权重。
- 必须设置包含仓库根目录、sam-hq 和 vggt 的 PYTHONPATH；README 解释 VGGT 子模块自带的 training 包会遮蔽主仓同名模块。
- Table 2 的论文复现命令使用 --nms_iou_type box；仓库新实验默认是 mask。变更该参数会改变后处理口径，不能混比。
- 评测入口会在运行输出中写入 provenance.json 与 git_diff.patch，记录 commit、配置、命令、解释器和 dirty 状态；复现实验应保留 provenance。
- Web demo 是可选研究演示，上传大小、帧数、速率限制和输出清理有配置项；公开部署应启用清理并设置限额。

## 对 wiki 的映射

- [SAM-V 论文实体页](../../wiki/entities/paper-sam-v.md)
- [SAM（Segment Anything）](../../wiki/entities/paper-segment-anything.md)——二维提示分割基础模型
- [VGGT 几何状态综述](../../wiki/overview/vggt-geometric-state-survey.md)——SAM-V 所用前馈多视角几何先验

## 原始来源

- GitHub 仓库：<https://github.com/gong208/SAM-V>
- README：<https://github.com/gong208/SAM-V/blob/main/README.md>
- NOTICE：<https://github.com/gong208/SAM-V/blob/main/NOTICE>
- Hugging Face checkpoint：<https://huggingface.co/Frank-Gong123/SAM-V>
- 论文归档：[arXiv:2609.25490](../papers/sam_v_arxiv_2609_25490.md)
