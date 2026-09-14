# EgoPro（LightwheelAI / Hugging Face）

> 来源归档

- **标题：** EgoPro
- **类型：** dataset / huggingface-bucket
- **链接：** <https://huggingface.co/datasets/LightwheelAI/EgoPro>
- **机构：** 光轮科技（LightwheelAI）
- **所属集合：** [EgoSuite-Open100K](../sites/hf-egosuite-open100k-collection.md)
- **入库日期：** 2026-09-14
- **一句话说明：** EgoSuite-Open100K 的 **10,000 h 头戴+腕部主线**：同步头/腕视频与 3D 手部姿态，body 子集加全身；针对接触、抓取、手部出框等头戴视角盲区补强。

## Sub-SKU

| Sub-SKU | 规划时长 | 相机 | 姿态标注 |
|---------|----------|------|----------|
| EgoProStandard | 8,000 h | 头戴 + 腕部 | 手部 |
| EgoProStandard-body | 2,000 h | 头戴 + 腕部 | 手部 + 全身 |

- 部分子集含 **事件级语义** 标注（complimentary add-on）。

## 设计动机（HF blog 归纳）

腕部相机补足头戴视角在 **接触瞬间、手部遮挡、细粒度抓取** 上的像素不足——手在头高处常仅数像素宽。

## 数据访问（官方 README 摘要）

```bash
pip install -U huggingface_hub
hf auth login
hf buckets list LightwheelAI/EgoPro -h -R
hf buckets sync hf://buckets/LightwheelAI/EgoPro/PREFIX ./LOCAL_DIR
```

## 隐私与许可

- 自动脱敏 + 人工核验；参与者知情同意。
- 学术研究与商业训练（见各数据卡 license）。

## 对 wiki 的映射

- [EgoSuite-Open100K](../../wiki/entities/egosuite-open100k.md)
