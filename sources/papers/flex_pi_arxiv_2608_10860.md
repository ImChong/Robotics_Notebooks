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
- **一句话说明：** 6B 多流 Joint WAM：共享冻结 Wan VAE 编码 RGB+pointmap，DINOv3 语义流，MoT 联合去噪；流 dropout + cross-modality forcing 使单 checkpoint 可在 action-only↔full joint 间切换算力。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | [flex-pi.github.io](https://flex-pi.github.io/) | 真机 2–7× 表、56 流组合交互、方法图 |
| 代码占位 | [geyan21/flex-pi](https://github.com/geyan21/flex-pi) | README 写明 code ready soon |

## 开源状态（步骤 2.5，截至 2026-08-13 复核）

- **宣称将开源 / 待发布：** 项目页链到 GitHub；API 显示仓 `size=1`、contents 仅 `README.md`（~2.2 KB），明确 **「The code is ready soon」**；无训练/推理入口、无权重、无 SPDX。
- **处理：** wiki 写「代码待发布」；源码运行时序图 **不适用**。

## 摘要级要点

- **问题：** 通用 WAM 几乎只预测 RGB latent，缺乏操纵所需的 3D/语义监督；加模态常意味着新传感器、新先验或更慢推理。
- **主张：** 冻结视频 VAE 几乎无损编码 pointmap；与 DINO 共训可「免费」放大 WAM；部署用 dropout 训练换算力柔性。
- **方法：** RGB/pointmap → Wan-2.2 VAE；DINO tokens（PixelUnshuffle）；5B MoT trunk + ~1B action expert；\(\mathbf{m}^{\mathrm{in}}/\mathbf{m}^{\mathrm{out}}\) 独立采样；始终对全部未来流算 flow loss（cross-modality forcing）。
- **结果要点：**
  - 真机双臂 YAM：ID avg full joint **83.0%** vs ManiFlow **58.0** / \(\pi_{0.5}\) **52.1**；OOD **76.1** vs **31.5/43.2**
  - action-only ~**60 ms**（快于 \(\pi_{0.5}\)）；full joint ~**193 ms**（RTX 5090，\(K{=}4\)）
  - RoboTwin 有限演示约 **1.9–4.5×** 最强对照；满数据两模式均为 **94.6%**
  - LIBERO 柔性 ckpt **98.5%**；固定模式 Flex-π\* **99.2%**；LIBERO-Plus Total full joint **80.9%**
- **局限：** 仍数据饥渴；joint 与 action-only 不能同时兼得最低延迟与最高成功率；真机微调 ≥10 epoch；LIBERO-Plus 落后强 VLM 骨干。

## 核心摘录（面向 wiki 编译）

### 1) 共享 VAE 几乎免费吃 3D

冻结 Wan-2.2 VAE（只在 RGB 上训过）直接编解码 pointmap：PSNR **31.1 dB**、归一化 MSE \(3.1{\times}10^{-3}\)、米制 \(z\)-RMSE **4.9 cm**。Pointmap 来自 DA3 离线标注（预训练不用 AGIBOT 官方残缺深度）。

### 2) 流掩码是注意力，不是损失开关

\(\mathbf{m}^{\mathrm{out}}\) 只改动作读哪些未来、未来流互看规则；四流（含动作）每步都算 FM loss。输入缺流仍生成该流未来 = CMF。去掉 CMF，RoboTwin 消融成功率约 **−21 pt**。

### 3) 部署数字（项目页 / 论文）

- 真机 ID：Put Plate 95.0 / Sort 75.0 / Kitchen 98.8 / Self-Repair 76.0 / Soft-Bag 70.0（full joint）
- Self-Repair 八阶段顺序任务：full joint 11/20 全过，最强基线 1/20；拧螺丝 ±0.25 mm
- 56 种流组合 = 7 非空输入 × 8 输出子集，同一权重运行时切换

### 4) 复现入口

截至入库复核日 **无**。Watch GitHub；论文附录 A–J 含架构、本体 32-D 布局、YAM 部署与完整表。

## 对 wiki 的映射

- 实体页：[wiki/entities/paper-flex-pi.md](../../wiki/entities/paper-flex-pi.md)
- 交叉：[World Action Models](../../wiki/concepts/world-action-models.md)、[DreamWAM](../../wiki/entities/paper-dreamwam.md)、[FACT](../../wiki/entities/paper-fact.md)、[MECo-WAM](../../wiki/entities/paper-meco-wam-4d-geometry-cotraining.md)、[生成式世界模型](../../wiki/methods/generative-world-models.md)、[VLA](../../wiki/methods/vla.md)、[操纵](../../wiki/tasks/manipulation.md)
