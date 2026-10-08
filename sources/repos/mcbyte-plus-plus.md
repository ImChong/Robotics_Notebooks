# tstanczyk95/McBytePlusPlus

- **类型：** repo / multi-object-tracking / sports analytics
- **官方仓库：** <https://github.com/tstanczyk95/McBytePlusPlus>
- **官方论文：** <https://arxiv.org/abs/2608.15688>
- **机构：** Inria STARS
- **许可证：** Apache-2.0（仓库 LICENSE）；第三方依赖、检查点与数据许可需分别核对
- **核查日期：** 2026-10-08
- **沉淀到 wiki：** [McByte++ 实体页](../../wiki/entities/mcbyte-plus-plus.md)

## 仓库要点

McByte++ 是 McByte 的更快、增强版，面向体育视频的长时多目标跟踪。官方 README 将其贡献概括为在线 Re-ID、轻量掩码传播和选择性相机运动补偿；论文摘要补充了 tracking-by-detection 框架、不需评测集特定 detector 训练或调参，以及 HOTA / IDF1 的提升。前作 McByte 与此 repo 是两个独立代码仓库，应分开引用。

## 复现入口

| 用途 | 文件 / 命令 |
|------|-------------|
| 在线 Re-ID 跟踪 | `python tools/demo_track__with_reid.py --path <帧目录>` |
| 不用 Re-ID | `python tools/demo_track__no_reid.py --path <帧目录>` |
| 主跟踪器 | `yolox/tracker/mcbyteplusplus_tracker__with_reid.py` |
| 安装 | `INSTALLATION.md` |

可用 `--det_path` 输入预计算检测结果，跳过内置 detector 参数。输出包括可视化帧与 MOT 格式轨迹文本。在线 Re-ID 通过目标外观特征与暂时不可见旧 tracklet 的特征做余弦相似度匹配；默认 `--reid_sim_thresh=0.8`。

## 开源与运行边界

- **已开源：** 官方仓库公开且附 Apache-2.0 LICENSE。
- **环境：** 安装文档以 Linux、CUDA 12.8、GCC 10.5、Python 3.10 为已测试组合；建议遵循其中 PyTorch / 依赖版本。
- **外部模型：** YOLOX 体育检测权重、EdgeTAM 权重与 GTA-link 来源的体育 Re-ID 权重需要从上游单独获取。检测文件也可作为输入。
- **流式限制：** README 当前说明 EdgeTAM 掩码传播实现会先载入所有帧，逐帧处理仍在规划中；所以在线 Re-ID 不等于整个实现已适合实时流。
- **性能读数：** 官方 FPS 在单张 NVIDIA H100 上测量且排除 detector，只反映 tracker 部分；报告部署速度时应单独核算检测阶段。
- **许可边界：** 仓库为 Apache-2.0，不代表各上游依赖、模型权重或数据拥有同一许可。

## 关键入口

- [README](https://github.com/tstanczyk95/McBytePlusPlus#readme) — 方法、演示、指标与阈值说明
- [INSTALLATION.md](https://github.com/tstanczyk95/McBytePlusPlus/blob/main/INSTALLATION.md) — 环境与模型下载
- [with-ReID demo](https://github.com/tstanczyk95/McBytePlusPlus/blob/main/tools/demo_track__with_reid.py)
- [with-ReID tracker](https://github.com/tstanczyk95/McBytePlusPlus/blob/main/yolox/tracker/mcbyteplusplus_tracker__with_reid.py)
- [LICENSE](https://github.com/tstanczyk95/McBytePlusPlus/blob/main/LICENSE)

## 对 wiki 的映射

- [McByte++ 实体页](../../wiki/entities/mcbyte-plus-plus.md) — 论文与代码合并的知识节点
- [McByte++ 论文归档](../papers/mcbyte-plus-plus_arxiv_2608_15688.md) — 摘要与结果摘录
