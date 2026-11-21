# LightNav-ER（Light Origins Tech Blog 2026-09-01）

> 来源归档 — Light Origins Tech Blog ingest（LightNav-0：以规模化 Real2Sim2Real 实现零样本通用导航）

- **标题：** LightNav-ER：具身推理中期训练模型
- **类型：** product / model checkpoint（依托 LightNav-0 论文发表）
- **关联论文：** [LightNav-0 (arXiv:2608.30935)](https://arxiv.org/abs/2608.30935)（§ IV-A、§ IV-C、§ V-A、§ VI-B Table II、§ VI-E Table IX）
- **项目页/引用：** https://www.lightorigins.com/en/blog/lightnav-0
- **Tech Blog：** <https://www.lightorigins.com/en/blog/lightnav-0>
- **入库日期：** 2026-09-21
- **核对范围：** 通读 `arXiv:2608.30935v1` HTML 全文中 ER mid-training 相关章节（§ IV-A 数据配比 5 类数据源、§ V-A 训练超参 `LR=1e-5` / `batch=128` / `10,240 tokens` / `~170 H100 GPU-hours` 与 specialize-then-retain 课程、§ VI-B Table II 8 项基准评测 67.4 avg、§ VI-E Table IX 下游导航消融 +2.4 pt SR / +1.0 pt SPL）及官方 Tech Blog；LightNav-ER 独立权重未发布，未在本地复现评测。
- **通读：** 已完成
- **一句话说明：** LightNav-0 第一阶段 ER 中期训练产物（基于 Qwen3-VL-4B）；8 项具身推理基准平均 67.4（4 第一 / 4 第二），为下游 SFT/RL 提供空间先验（下游 8 设定平均 SR 60.8→63.1）。

## 开源状态

- **未单独发布**（步骤 2.5，2026-09-21）

## 核心摘录

1. **引用上下文：** [lightorigins_lightnav_0_2026-09-01.md](../blogs/lightorigins_lightnav_0_2026-09-01.md)
2. **导读：** 对齐段「先理解空间再学行动」的可操作锚点；MolmoER / Gemini Robotics-ER 同路线对照。

## 对 wiki 的映射

- [lightnav-er](../../wiki/entities/lightnav-er.md)
- [3 篇技术地图](../../wiki/overview/lightorigins-3blogs-technology-map.md)
