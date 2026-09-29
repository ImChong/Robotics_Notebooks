# SAM 3 / SAM 3.1（Segment Anything Model 3）官方仓

> 来源归档

- **标题：** SAM 3: Segment Anything with Concepts · SAM 3.1（Object Multiplex）
- **类型：** repo
- **组织：** Meta / facebookresearch
- **链接：** <https://github.com/facebookresearch/sam3>
- **论文：** <https://arxiv.org/abs/2511.16719>
- **项目页：** <https://ai.meta.com/sam3/> · Blog <https://ai.meta.com/blog/segment-anything-model-3/>
- **项目页归档：** [meta-sam3.md](../sites/meta-sam3.md)
- **入库日期：** 2026-08-04（SAM 3）；**2026-09-29**（SAM 3.1 release 再核）
- **一句话说明：** SAM 3 推理与微调官方仓：文本/几何/exemplar 概念提示分割；**SAM 3.1** 增视频 **Object Multiplex**、编译/批处理优化与 `facebook/sam3.1` checkpoint。
- **沉淀到 wiki：** [`wiki/entities/paper-sam3.md`](../../wiki/entities/paper-sam3.md)

## 开源状态

**已开源**：推理、微调、checkpoint（HF `facebook/sam3` / **`facebook/sam3.1`**）、Release Notes（`RELEASE_SAM3p1.md`）与示例 notebook。

## 仓库入口（README / Release 级）

| 组件 | 说明 |
|------|------|
| 安装 / 权重 | README Getting Started；HF [`facebook/sam3`](https://huggingface.co/facebook/sam3) 与 [`facebook/sam3.1`](https://huggingface.co/facebook/sam3.1) |
| 图像 / 视频 PCS | 概念分割与跟踪 notebook |
| SAM 3.1 视频 | [`examples/sam3.1_video_predictor_example.ipynb`](https://github.com/facebookresearch/sam3/blob/main/examples/sam3.1_video_predictor_example.ipynb) — 文本/点提示 + **Object Multiplex** |
| 微调 | 仓内 finetuning 说明 |
| SAM 3.1 变更 | [`RELEASE_SAM3p1.md`](https://github.com/facebookresearch/sam3/blob/main/RELEASE_SAM3p1.md) — 约 7×（128 目标/H100）、VOS/MOSEv2 等 |

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-sam3](../../wiki/entities/paper-sam3.md) | 论文实体（含 3.1 更新节） |
| [sam3_1_release_2026_03](../papers/sam3_1_release_2026_03.md) | SAM 3.1 Release Notes 摘录 |
| [paper-sam2](../../wiki/entities/paper-sam2.md) / [sam2](./sam2.md) | 视频可提示前代 |
| [paper-segment-anything](../../wiki/entities/paper-segment-anything.md) | 静态图奠基 |
| [paper-blip2](../../wiki/entities/paper-blip2.md) | 课程零样本管线常见图文侧 |
| [GO2 SAM 流水线](../../wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md) | 四足语义建图 2D 侧可升级至 SAM3/SAM3.1 |
