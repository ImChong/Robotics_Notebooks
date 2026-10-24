# DistScene: Object-to-Scene Distillation for 3D Scene Generation

> 来源归档（ingest）

- **标题：** DistScene: Object-to-Scene Distillation for 3D Scene Generation
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.06960>
- **HTML / PDF：** <https://arxiv.org/html/2610.06960v1> · <https://arxiv.org/pdf/2610.06960>
- **作者：** Kunming Luo, Hongyu Yan, Ken Deng, Chengcheng Zhou, Tianyu Liu, Haipeng Li, Haibin Huang, Xuelong Li, Ping Tan
- **机构：** 香港科技大学（HKUST）；中国电信人工智能（TeleAI）
- **项目页：** <https://coolbeam.github.io/DistScene/>
- **提交日期：** 2026-10-03（arXiv v1）
- **入库日期：** 2026-10-24
- **一句话说明：** 从单张图像联合生成可分解的环境与物体，并通过场景条件下的局部精修恢复物体细节。

## 摘要要点

- **问题：** 单图场景重建需要同时推断环境、物体几何和相互位置；只生成物体集合时，环境几何缺少显式建模，容易出现布局与支撑关系不一致。
- **Scene-Frame Generation：** 在共享场景坐标系中生成独立物体与环境，使背景几何参与物体位置、尺度和关系建模。
- **Object-Centric Refinement：** 将各物体变换至归一化局部空间，在场景潜变量与输入图像条件下精修，再放回原场景位置。
- **Object-to-Scene Distillation：** 用 LLM 生成场景与物体描述，以 TRELLIS.2 分别生成对象和环境，经接触 / 碰撞检查组合并渲染，构造约 125k 个合成训练场景。
- **评测：** 室内 MIDI-test 上场景级 Chamfer Distance 从 3D-Fixer 的 0.1295 降至 0.0877（相对降低 32.2%）；户外 UrbanScene3D 上 CD-L1 为 0.0772、F-score 为 0.722。指标比较只代表论文采用的几何评测口径。

## 对 wiki 的映射

- [DistScene 实体页](../../wiki/entities/paper-distscene.md) — 方法、几何评测、资源开放状态与仿真使用边界。

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2610.06960>
- arXiv HTML：<https://arxiv.org/html/2610.06960v1>
- 官方项目页：<https://coolbeam.github.io/DistScene/>
