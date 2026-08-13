# Flex-π（arXiv:2608.10860）

> 来源归档（ingest）

- **标题：** Flex-π: A Multi-Stream World-Action Model with Compute Flexibility
- **类型：** paper / world-action-models / joint-wam / multi-stream / compute-flexibility / manipulation
- **arXiv abs：** <https://arxiv.org/abs/2608.10860>
- **PDF：** <https://arxiv.org/pdf/2608.10860>
- **HTML：** <https://arxiv.org/html/2608.10860>
- **项目页：** <https://flex-pi.github.io/> — 归档见 [`sources/sites/flex-pi-github-io.md`](../sites/flex-pi-github-io.md)
- **代码：** <https://github.com/geyan21/flex-pi> — 归档见 [`sources/repos/flex-pi.md`](../repos/flex-pi.md)（**待发布**）
- **机构：** 华盛顿大学（UW）；艾伦人工智能研究所（AI2）
- **作者：** Ge Yan\*、Jinghao Liu\*、Yuzhi Fan\*、Lei Cai、Minwen Liao、Jesse Zhang†、Dieter Fox†（\* equal；† equal advising）
- **发表 / 上传：** 2026-08（arXiv:2608.10860）
- **入库日期：** 2026-08-13
- **一句话说明：** 6B 多流 Joint WAM：共享 Wan VAE 编码 RGB+pointmap，DINOv3 语义流，MoT 联合去噪；流 dropout + cross-modality forcing 使单 checkpoint 可在 action-only↔full joint 间切换算力。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | [flex-pi.github.io](https://flex-pi.github.io/) | 真机 2–7× 表、56 流组合交互、方法图 |
| 代码占位 | [geyan21/flex-pi](https://github.com/geyan21/flex-pi) | README 写明 code ready soon |

## 开源状态（步骤 2.5，截至 2026-08-13）

- **部分 / 待发布：** 项目页链到 GitHub，但仓库仅 README，明确 **「The code is ready soon」**；无训练/推理入口、无权重。
- **处理：** wiki 写「代码待发布」；源码运行时序图 **不适用**。

## 摘要级要点

- **问题：** 通用 WAM 几乎只预测 RGB latent，缺乏操纵所需的 3D/语义监督；加模态常意味着新传感器、新先验或更慢推理。
- **主张：** 冻结视频 VAE 几乎无损编码 pointmap；与 DINO 共训可「免费」放大 WAM；部署用 dropout 训练换算力柔性。
- **方法：** RGB/pointmap → Wan-2.2 VAE；DINO tokens；5B MoT trunk + ~1B action expert；\(\mathbf{m}^{\mathrm{in}}/\mathbf{m}^{\mathrm{out}}\) 独立采样；始终对全部未来流算 flow loss（cross-modality forcing）。
- **结果要点：**
  - 真机双臂 YAM：ID avg full joint **83.0%** vs ManiFlow **58.0** / \(\pi_{0.5}\) **52.1**；OOD **76.1** vs **31.5/43.2**
  - action-only ~**60 ms**（快于 \(\pi_{0.5}\)）；full joint ~**193 ms**（RTX 5090）
  - RoboTwin 有限演示约 **1.9×** 最强 WAM；LIBERO 总体最高 **99.2%**
- **局限：** 仍数据饥渴；joint 与 action-only 不能同时兼得最低延迟与最高成功率。

## 对 wiki 的映射

- 新建实体页：[wiki/entities/paper-flex-pi.md](../../wiki/entities/paper-flex-pi.md)
- 交叉：[World Action Models](../../wiki/concepts/world-action-models.md)、[DreamWAM](../../wiki/entities/paper-dreamwam.md)、[FACT](../../wiki/entities/paper-fact.md)、[VLA](../../wiki/methods/vla.md)
