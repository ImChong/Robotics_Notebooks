# SAM 3（Segment Anything Model 3）官方仓

> 来源归档

- **标题：** SAM 3: Segment Anything with Concepts
- **类型：** repo
- **组织：** Meta / facebookresearch
- **链接：** <https://github.com/facebookresearch/sam3>
- **论文：** <https://arxiv.org/abs/2511.16719>
- **项目页：** <https://ai.meta.com/sam3/>
- **入库日期：** 2026-08-04
- **一句话说明：** SAM 3 推理与微调官方仓：文本/几何/exemplar 概念提示分割，含 checkpoint 与 notebook。
- **沉淀到 wiki：** [`wiki/entities/paper-sam3.md`](../../wiki/entities/paper-sam3.md)

## 开源状态

**已开源**：推理、微调、checkpoint 下载链接与示例 notebook（以仓库 README / `RELEASE_*` 为准）。

## 仓库入口（README 级）

| 组件 | 说明 |
|------|------|
| 安装 / 权重 | 按官方 README 与 Hugging Face `facebook/sam3` 入口 |
| 推理 | 图像/视频概念分割与跟踪 notebook |
| 微调 | 仓内 finetuning 说明 |

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-sam3](../../wiki/entities/paper-sam3.md) | 论文实体 |
| [paper-sam2](../../wiki/entities/paper-sam2.md) / [sam2](./sam2.md) | 视频可提示前代 |
| [paper-segment-anything](../../wiki/entities/paper-segment-anything.md) | 静态图奠基 |
| [paper-blip2](../../wiki/entities/paper-blip2.md) | 课程零样本管线常见图文侧 |
| [GO2 SAM 流水线](../../wiki/queries/go2-3d-semantic-mapping-sam-pipeline.md) | 四足语义建图 2D 侧可升级至 SAM3 |
