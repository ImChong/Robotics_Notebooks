# Meta SAM 2 项目页

> 来源归档

- **标题：** Introducing Meta Segment Anything Model 2 (SAM 2)
- **类型：** site / project page
- **URL：** <https://ai.meta.com/sam2>
- **论文：** <https://arxiv.org/abs/2408.00714> / Meta 研究页
- **代码：** <https://github.com/facebookresearch/sam2>
- **Demo：** <https://sam2.metademolab.com/>
- **数据集：** <https://ai.meta.com/datasets/segment-anything-video>
- **入库日期：** 2026-07-26
- **一句话说明：** Meta 官方介绍 SAM 2：统一图像/视频可提示分割、流式 memory、SA-V、模型与 demo 下载入口。

## 开源状态（项目页核查，2026-07-26）

| 项 | 状态 |
|----|------|
| Paper / Blog | 已挂链 |
| Code / Model | **已开源** — Download the model → GitHub `facebookresearch/sam2` |
| Dataset | SA-V 开放（CC BY 4.0） |
| Demo | sam2.metademolab.com |
| 复现范围 | 预训练权重 + 训练代码 + demo（Apache-2.0） |

## 页面结构（策展）

- Key capabilities — 任意对象、跨帧 refinement、零样本、实时流式推理
- Model architecture — memory 模块将 SAM 推广到视频；图像时 memory 为空
- SA-V — ~51K 视频、part/遮挡多样性
- Open innovation — 模型 / 数据 / demo / 代码

## 对 wiki 的映射

- 论文归档：[`sources/papers/sam2_arxiv_2408_00714.md`](../papers/sam2_arxiv_2408_00714.md)
- 仓库归档：[`sources/repos/sam2.md`](../repos/sam2.md)
- 沉淀 **[`wiki/entities/paper-sam2.md`](../../wiki/entities/paper-sam2.md)**
