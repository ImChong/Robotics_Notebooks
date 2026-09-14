# Hugging Face Collection — EgoSuite-Open100K

> 来源归档

- **标题：** EgoSuite-Open100K（Hugging Face Collection）
- **类型：** site / huggingface-collection
- **URL：** <https://huggingface.co/collections/LightwheelAI/egosuite-open100k>
- **机构：** 光轮科技（LightwheelAI）
- **入库日期：** 2026-09-14
- **集合更新：** 约 2026-08（API 显示 25 days ago，相对入库日）
- **一句话说明：** 光轮智能 **最大规模全标注开放 egocentric 人类数据** 官方索引：规划 100k h，首批 10k h，含 EgoStandard / EgoPro / EgoDemo 三条 HF 数据集入口。

## 集合成员（3 项）

| 数据集 | 链接 | 角色 |
|--------|------|------|
| **EgoStandard** | <https://huggingface.co/datasets/LightwheelAI/EgoStandard> | 90k h 规划 · 头戴视角 · 手/身姿态 |
| **EgoPro** | <https://huggingface.co/datasets/LightwheelAI/EgoPro> | 10k h 规划 · 头戴+腕部 · 手/身姿态 |
| **EgoDemo** | <https://huggingface.co/datasets/LightwheelAI/EgoDemo> | 50 h 全 Sub-SKU 小样 + raw 变体 |

## 分发机制

- **EgoStandard** / **EgoPro** 本体数据经 **HF Bucket** 分发（`hf buckets list/sync`）；数据集仓为 **数据卡 + 访问入口**。
- 需 HF 账号登录并通过 access 审批后下载。
- Bucket 前缀与 manifest 为 **真源**（体量大、可变）。

## 对 wiki 的映射

- [EgoSuite-Open100K](../../wiki/entities/egosuite-open100k.md)
- [EgoStandard 数据卡](../datasets/lightwheel-egostandard.md)
- [EgoPro 数据卡](../datasets/lightwheel-egopro.md)
