# LAVIS / BLIP-2（Salesforce）

> 来源归档

- **标题：** BLIP-2 in LAVIS
- **类型：** repo
- **组织：** Salesforce
- **链接：** <https://github.com/salesforce/LAVIS>（入口目录 `projects/blip2`）
- **论文：** <https://arxiv.org/abs/2301.12597>
- **HF：** <https://huggingface.co/Salesforce/blip2-opt-2.7b> 等
- **入库日期：** 2026-08-04
- **一句话说明：** BLIP-2 官方实现所在库：模型 zoo、预训练/微调配置与推理示例；亦经 Hugging Face Transformers 使用。
- **沉淀到 wiki：** [`wiki/entities/paper-blip2.md`](../../wiki/entities/paper-blip2.md)

## 开源状态

**已开源**：`blip2_opt` / `blip2_t5` / `blip2` 等架构与多种 checkpoint 类型（pretrain / caption_coco 等）。

## 使用提示（README 级）

| 用途 | 建议 model type |
|------|-----------------|
| 零样本 image-to-text | `pretrained_{LLM}` |
| COCO 风格 caption | `caption_coco_{LLM}` |
| 图文特征 / 检索 | `blip2` 架构 |

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-blip2](../../wiki/entities/paper-blip2.md) | 论文实体 |
| [vision-language-feature-fusion](../../wiki/concepts/vision-language-feature-fusion.md) | Q-Former 对齐概念 |
| [paper-sam3](../../wiki/entities/paper-sam3.md) | 零样本感知管线常见搭档 |
