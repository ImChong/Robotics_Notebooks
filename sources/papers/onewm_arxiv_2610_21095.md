# OneWM: Can World and Action Modeling Benefit Each Other under Unified Causal Training?（arXiv 论文）

> 来源归档（ingest）

- **标题：** Can World and Action Modeling Benefit Each Other under Unified Causal Training?
- **类型：** paper / arXiv preprint
- **作者：** Yue Su、Bokai Lin、Xinkai Wang、Chubin Zhang、Xiangnan Wu、Kuan Zhang、Weiyang Jin、Jiwen Yu、Yiming Li、Cewu Lu、Lixin Yang、Xihui Liu
- **机构：** MMLab@HKU、MVIG@SJTU、清华大学、上海创智学院、中国科学院大学、IAAI Team
- **论文：** <https://arxiv.org/abs/2610.21095>（v1，2026-10-15）
- **项目页：** <https://selen-suyue.github.io/OneWMNet/> — [项目页归档](../sites/onewm-project.md)
- **代码：** <https://github.com/Interactive-Active-AI/OneWM> — [仓库归档](../repos/onewm.md)
- **入库日期：** 2026-10-19

## 核心论文摘录

### 1) OneWM：统一因果视频—动作建模（Yue Su et al., 2026）
- **链接：** <https://arxiv.org/abs/2610.21095>
- **核心贡献：** 在因果 Mixture-of-Transformers 中让动作同时作为未来视频预测的干净条件和策略学习目标；通过 all-clean、partial-clean、all-noisy 三种条件化方式，在同一架构里联合训练前向动态预测和动作生成。
- **主要结果：** 相比同架构 WAM_mask，LIBERO、LIBERO-Plus、RoboCasa365 平均成功率分别提升 2.50、5.90、8.36 个百分点；同架构 WM 对照中，OneWM 的 PSNR、SSIM 和 LPIPS 在三个基准上均改善。真实机器人评估含堆叠、插入与早餐摆放任务。
- **对 wiki 的映射：**
  - [OneWM 论文实体页](../../wiki/entities/paper-onewm-unified-causal-training.md)
  - [World Action Models（WAM）](../../wiki/concepts/world-action-models.md)

### 2) Endless Dream：在想象中闭环（Yue Su et al., 2026）
- **链接：** <https://arxiv.org/abs/2610.21095>
- **核心贡献：** 将同一检查点的动作生成与给定动作的未来视频预测串接：从一次初始观测出发，每轮用预测视频末帧生成下一段动作，形成无需后续真实观测的想象 rollout。
- **主要边界：** LIBERO 演示为 16 轮、512 个预测动作、建模时长 25.6 秒；展示的是想象中的任务推进，不包含这些动作在模拟器中的执行或自动完成判定，长 rollout 可能累积生成误差。
- **对 wiki 的映射：**
  - [OneWM 论文实体页](../../wiki/entities/paper-onewm-unified-causal-training.md)
  - [世界-动作模型（WAM）](../../wiki/concepts/world-action-models.md)

## 阅读提示

论文回答两个不同问题：在固定因果架构下，明确学习“指定动作会让画面如何变化”是否有助于策略学习；联合动作学习是否反过来提升给定动作后的视觉预测。读结果时要分开对应对照：策略任务与 WAM 变体比较，世界预测与仅做动作条件视频预测的 WM 比较。公开仓库提供训练和部署代码；权重与真实 Franka 数据需要按 README 自行准备。
