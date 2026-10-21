# ReEnvision: Future-Guided Selective Action Repair for Long-Horizon Manipulation（arXiv:2610.17977）

> 来源归档（ingest）

- **标题：** ReEnvision: Future-Guided Selective Action Repair for Long-Horizon Manipulation
- **arXiv：** <https://arxiv.org/abs/2610.17977>（PDF：<https://arxiv.org/pdf/2610.17977>；HTML：<https://arxiv.org/html/2610.17977v2>）
- **代码：** 论文称源码与权重 "will be publicly released upon acceptance"；项目页无仓库链接（2026-10-17 核查）
- **项目页：** <https://reenvision.pages.dev/>（归档见 [reenvision-project](../sites/reenvision-project.md)）
- **作者：** Xiyin Zeng、Xuewei Huang、Yi Lu（Information Hub, 香港科技大学（广州））；Weihao Yuan（Daimon Robotics）；Hao Wang（南京大学智能科学与技术学院）（单位来自 arXiv HTML v2）
- **入库日期：** 2026-10-17
- **来源：** 具身智能小站 2026-10-17 公众号摘要 + arXiv HTML 全文
- **一句话说明：** 维护未来与动作的联合信念，据执行反馈定位缓存动作块中受影响的动作，只修复这些动作，严重偏离时才整体重规划。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-17）：** 页面仅有演示视频与图，无 GitHub / Hugging Face 链接，未标注发布状态。
- **论文核查：** 声明录用后公开代码与权重，未给 URL。
- **结论：** **宣称将开源 / 未列链接**（截至 2026-10-17）。

## 摘录 1：方法

- 联合信念 h_t = (z_f, z_a)：未来表示来自预训练 Stable Video Diffusion，动作意图来自 MLP；扩散策略以 (o_t, l, z_f, z_a) 生成 K 步动作块。
- 信念修订：RSSM 的先验 / 后验分支，Δh = h_post − h_prior；TCN 影响模型逐动作预测修正量、有效性与不确定性，经带惩罚的目标选出修复掩码。
- 规则：u_max ≥ η 则整体重规划；无动作被选中则保留后缀；修复比例 r ≤ ρ 且片段数 ≤ s_max 则局部修复，否则重规划。默认 ρ=0.6、s_max=1、η=0.9。

**对 wiki 的映射：** 核心原理与流程总览。

## 摘录 2：结果

- CALVIN ABC→D：5 步成功率 86.6%，平均长度 4.62（VITA 84.5 / 4.57，DeFI 81.2 / 4.51）。
- L-CALVIN：5 步 0.89、10 步 0.63、平均长度 8.36（Long-VLA 0.85 / 0.56 / 8.24；DeFI 0.75 / 0.44 / 7.30）。
- RLBench 18 任务平均 88.8%；整体重规划 88.9%，略高（消融表）。
- 真机 ARX X5，5 任务各 50 次：82.8%（π0.5* 80.8%，Diffusion Policy 68.8%）。
- 开销：端到端约 145 ms，对比骨干约 110 ms（约 32%）。
- 受控扰动（项目页）：小偏离保留缓存后缀 73%，严重偏离整体重规划 52%。

**对 wiki 的映射：** 实验与评测、结论、局限。

## 备注

- 摘要写 10 步成功率「56%→63%」，表 2 中 56% 为 Long-VLA，DeFI 骨干为 44%，基线口径不同。

## 建议 wiki 动作

- 新建 [`wiki/entities/paper-reenvision.md`](../../wiki/entities/paper-reenvision.md)。
