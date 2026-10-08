# Training-Free Long-Term Multi-Object Tracking for Sports Video Analytics

> 来源归档（arXiv / 项目 ingest）

- **标题：** Training-Free Long-Term Multi-Object Tracking for Sports Video Analytics
- **短名：** McByte++
- **类型：** paper / computer-vision / multi-object-tracking / sports / re-identification
- **arXiv：** <https://arxiv.org/abs/2608.15688>
- **PDF：** <https://arxiv.org/pdf/2608.15688>
- **项目页 / 官方代码：** <https://github.com/tstanczyk95/McBytePlusPlus>
- **代码状态：** **已开源**；官方仓库公开，声明 Apache-2.0
- **提交日期：** 2026-08-16（arXiv v1）
- **作者：** Tomasz Stanczyk、Seongro Yoon、Francois Bremond
- **机构：** INRIA STARS 团队（官方仓库说明 designed and developed at Inria）
- **一句话说明：** 免数据集特定训练的体育 tracking-by-detection 系统，把轻量掩码传播、条件相机运动补偿与在线行人再识别结合，支持目标离开和重新进入场景时尝试恢复原身份。
- **沉淀到 wiki：** [McByte++ 实体页](../../wiki/entities/mcbyte-plus-plus.md)

## 摘要级要点

- **问题：** 体育视频中的遮挡、快速相机运动与球员再次出现会造成短轨迹断裂和身份变化；常规逐帧关联难以在长时间离场后恢复身份。
- **方法：** McByte++ 以检测驱动跟踪为基础，结合轻量掩码传播、条件相机运动补偿（CMC）和在线 Re-ID。Re-ID 将当前新出现目标与目前不可见的旧轨迹进行特征比较；官方演示默认相似度阈值 0.8。
- **训练要求：** 论文摘要称不需要在评测数据集上重新训练 detector 或进行数据集特定调参；实际运行仍依赖预训练检测器、掩码模型与（在线 Re-ID 模式下）身份特征模型。
- **代码入口：** `tools/demo_track__with_reid.py`（在线 Re-ID）与 `tools/demo_track__no_reid.py`（无 Re-ID）；支持帧目录输入，也可通过 `--det_path` 读入预先计算的检测结果。
- **论文摘要主张：** 相比原始 McByte，在线设置最高 +3.0 HOTA、+6.1 IDF1；通过轻量掩码/运动建模实现最高约一个数量级的跟踪速度提升。

## README 报告的基准结果摘录

官方仓库 README 将 FPS 说明为在单张 NVIDIA H100 上测量，**仅计跟踪，不包括检测**；检测结果由外部预先生成后输入跟踪器。表内比较原始 McByte 与 McByte++ 在线 Re-ID：

| 测试集 | McByte HOTA | McByte IDF1 | McByte FPS | McByte++ HOTA | McByte++ IDF1 | McByte++ FPS |
|--------|-------------|-------------|------------|---------------|---------------|--------------|
| SoccerNet-tracking 2022 test | 85.0 | 79.9 | 1.04 | 87.5 | 84.5 | 8.69 |
| SportsMOT test | 76.9 | 77.5 | 3.60 | 79.9 | 83.6 | 14.57 |
| SoccerNet-tracking 2023 challenge | 64.1 | 76.5 | 1.46 | 64.3 | 78.6 | 11.13 |
| DanceTrack test | 67.1 | 68.1 | 2.00 | 64.5 | 67.8 | 20.23 |

DanceTrack 的 Re-ID 收益较小：README 将其归因于人员通常停留在画面内，且使用的 Re-ID 模型主要面向体育场景；速度提升仍明显。离线 GTA-link 全局关联作为单独后处理列在 README 中，不是 McByte++ 在线运行所需组件。

## 工程与开源核查

- 安装文档报告测试环境为 Linux、CUDA 12.8、GCC 10.5、Python 3.10。
- 检测器、EdgeTAM 掩码模型和 Re-ID 预训练权重来自各自上游仓库，需单独下载配置。
- 当前掩码传播基于 EdgeTAM 实现，需要一次性载入全部帧；官方 README 表示后续考虑支持逐帧处理。
- 仓库声明 Apache-2.0；依赖组件、预训练权重与数据的许可不由该仓库声明自动覆盖。

## 对 wiki 的映射

- [McByte++ 实体页](../../wiki/entities/mcbyte-plus-plus.md) — 将论文、方法与官方代码合并为一个项目节点
- [Roboflow Sports](../../wiki/entities/roboflow-sports.md) — 相关的体育 CV 工具节点（不是依赖或代码子模块）
