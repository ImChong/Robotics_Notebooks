# LeVJEPA 项目页（levjepa.github.io）

> 来源归档

- **标题：** LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics
- **类型：** site / project page
- **URL：** <https://levjepa.github.io/>
- **论文：** <https://arxiv.org/abs/2608.27395>
- **代码：** <https://github.com/MLO-lab/LeVJEPA>
- **权重：** <https://huggingface.co/galilai-group/LeVJEPA-VideoMix-Large>
- **入库日期：** 2026-09-04
- **一句话说明：** 官方落地页：冻结 ViT-L patch-token PCA 交互演示、目标/架构/token dropping/block-causal 叙述，以及与 V-JEPA 2 / VideoMAEv2 / DINOv2 的对照表；页内互链代码与权重。

## 开源状态（步骤 2.5，2026-09-04）

| 项 | 状态 |
|----|------|
| 项目页 | **有**；演示 + 摘要 + 引用 |
| Code | 页内指向 [MLO-lab/LeVJEPA](https://github.com/MLO-lab/LeVJEPA) |
| Checkpoints | HF `galilai-group/LeVJEPA-VideoMix-Large` |
| 结论 | **已开源**（训练代码 MIT；`module.py` 与权重 CC BY-NC 4.0） |

页内交互演示说明：冻结编码器的 patch token 前三个主成分渲成 RGB；训练目标只监督 clip 级 `[cls]`，物体仍与背景分离。

## 对 wiki 的映射

- 论文：[`sources/papers/levjepa_arxiv_2608_27395.md`](../papers/levjepa_arxiv_2608_27395.md)
- 代码：[`sources/repos/levjepa.md`](../repos/levjepa.md)
- 沉淀 **[`wiki/entities/paper-levjepa.md`](../../wiki/entities/paper-levjepa.md)**
