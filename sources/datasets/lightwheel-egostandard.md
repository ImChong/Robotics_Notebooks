# EgoStandard（LightwheelAI / Hugging Face）

> 来源归档

- **标题：** EgoStandard
- **类型：** dataset / huggingface-bucket
- **链接：** <https://huggingface.co/datasets/LightwheelAI/EgoStandard>
- **机构：** 光轮科技（LightwheelAI）
- **所属集合：** [EgoSuite-Open100K](../sites/hf-egosuite-open100k-collection.md)
- **入库日期：** 2026-09-14
- **一句话说明：** EgoSuite-Open100K 的 **90,000 h 头戴视角主线**：同步 3D 手部姿态，body 子集加全身姿态；数据经 `LightwheelAI/EgoStandard` Bucket 分发，LeRobot v3 与 MCAP 为同一 episode 双格式。

## Sub-SKU

| Sub-SKU | 规划时长 | 相机 | 姿态标注 |
|---------|----------|------|----------|
| EgoStand | 80,000 h | 头戴 | 手部 |
| EgoStand-body | 10,000 h | 头戴 | 手部 + 全身 |

- **不含** 腕部相机（腕部见 [EgoPro](./lightwheel-egopro.md)）。
- 部分子集含 **事件级语义** 标注（complimentary add-on）。

## 数据访问（官方 README 摘要）

```bash
pip install -U huggingface_hub
hf auth login
hf buckets list LightwheelAI/EgoStandard -h -R
hf buckets sync hf://buckets/LightwheelAI/EgoStandard/PREFIX ./LOCAL_DIR
```

## 隐私与许可

- 自动脱敏（人脸、车牌等）+ 人工核验；参与者知情同意。
- 使用受仓库 license 与 access terms 约束；**学术研究与商业训练**（见 HF blog）。

## 对 wiki 的映射

- [EgoSuite-Open100K](../../wiki/entities/egosuite-open100k.md)
