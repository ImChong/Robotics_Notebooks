# Go-with-the-Track 项目页（eyeline-labs.github.io）

> 来源归档

- **标题：** Go-with-the-Track: Video Compositing and Motion Control with Point Tracking
- **类型：** site（项目页 + 视频 demo + 消融）
- **URL：** <https://eyeline-labs.github.io/Go-with-the-Track/>
- **论文：** <https://arxiv.org/abs/2606.20891>
- **代码：** <https://github.com/Eyeline-Labs/Go-with-the-Track>
- **模型：** <https://huggingface.co/Eyeline-Labs/Go-with-the-Track>
- **数据集：** <https://huggingface.co/datasets/Eyeline-Labs/Go-with-the-Track>
- **入库日期：** 2026-09-15
- **一句话说明：** Eyeline Labs / Netflix 等 SIGGRAPH 2026 工作官方页：reference-anchored point-tracks 统一视频合成与运动控制；含网格风格化、相机重定向、消融与数据集可视化。

## 开源核查（步骤 2.5）

| 项 | 结论（2026-09-15） |
|----|-------------------|
| 项目页 Code | GitHub + HF model + HF dataset 三链齐全 |
| 开放程度 | **已开源**（推理代码 + checkpoint + eval 数据） |
| 交叉归档 | [go_with_the_track_arxiv_2606_20891.md](../papers/go_with_the_track_arxiv_2606_20891.md)；[go-with-the-track.md](../repos/go-with-the-track.md) |

## 页面结构（维护索引）

| 区块 | 内容要点 |
|------|----------|
| Abstract | 统一 motion control + compositing；reference-anchored tracks |
| Applications | 多参考风格化、网格风格化/合成、关键点合成、静态/动态相机、时序稳定、首帧重建/风格化 |
| Ablations | embedder / adapter / relative position；数据集混合；关键帧数量；迭代 resampling |
| Dataset | 训练样本与增强可视化 |

## 对 wiki 的映射

- 主实体：[paper-go-with-the-track.md](../../wiki/entities/paper-go-with-the-track.md)
- 论文：[go_with_the_track_arxiv_2606_20891.md](../papers/go_with_the_track_arxiv_2606_20891.md)
- 代码：[go-with-the-track.md](../repos/go-with-the-track.md)
