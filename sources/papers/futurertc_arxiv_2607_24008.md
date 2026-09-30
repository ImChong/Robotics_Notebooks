# FutureRTC: Real-Time Robot Execution with Anticipatory-Conditioned Action Chunking（arXiv:2607.24008）

> 来源归档（ingest）

- **标题：** FutureRTC: Real-Time Robot Execution with Anticipatory-Conditioned Action Chunking
- **短名：** FutureRTC
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2607.24008>
- **项目页：** <https://jianghaiscu.github.io/FutureRTC_proj/>
- **机构：** 四川大学；电子科技大学（UESTC）；阿尔伯塔大学（University of Alberta）
- **入库日期：** 2026-09-30
- **一句话说明：** 冻结 VLA 前挂 plug-and-play 适配器，预测执行时刻的视觉 latent 与本体状态，缓解异步 chunk 的 prediction–execution misalignment；LIBERO / Kinetix / 双臂真机延迟注入实验。

## 开源状态（步骤 2.5，2026-09-30）

- **待发布**：项目页写明「Under review · code will be released」；截至入库日无 GitHub / 权重链接。
- 实验基于 LeRobot 发布的 π₀.₅ / SmolVLA LIBERO 微调权重；训练期基线（T-RTC、VLASH、REMAC）按论文复现或官方实现对照。

## 核心摘录（面向 wiki 编译）

- 根因：异步下 chunk 在 **stale** \((o,s)\) 上生成、在更晚时刻执行；仅 forward-roll 状态（VLASH）不够，**视觉场景也随 delay 演化**。
- **SCM**：对已提交动作 roll-forward 的状态加 MLP 残差补偿；**OPM**：在 VLA 视觉 latent 空间做 motion-aware warp + synthesis gate，绕过重跑 vision encoder。
- **Policy consistency loss**：预测上下文下单步 flow 近似对齐真执行时刻 chunk。
- LIBERO π₀.₅：\(d=20\) 成功率 68.3%→88.5%（naive async）；参数 +5.19M、延迟 +3.04 ms（SmolVLA 档 +6.45M / +3.64 ms）。
- **对 wiki 的映射：** [paper-futurertc](../../wiki/entities/paper-futurertc.md)
