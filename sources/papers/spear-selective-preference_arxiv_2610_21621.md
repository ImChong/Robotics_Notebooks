# SPEAR: Selective Preference Elicitation with Adaptive Rewards for VLM-Guided Robotic Manipulation

> 来源归档（ingest），来自 2026-11-13「具身智能小站」18 篇论文阅读清单；以原始 arXiv 及官方项目和仓库为核查依据。

## 元数据

- **标题：** SPEAR: Selective Preference Elicitation with Adaptive Rewards for VLM-Guided Robotic Manipulation
- **中文：** SPEAR：机器人强化学习中的选择性 VLM 偏好询问
- **arXiv：** 2610.21621
- **版本：** 2026-11-14 检查 arXiv HTML；具体修订号待逐项补齐
- **入库日期：** 2026-11-14
- **核对范围：** 已直接阅读 arXiv HTML 摘要、方法与部分实验/局限章节；尚未逐行检查全文及全部附录，后续继续核对
- **通读：** 未完成
- **论文：** https://arxiv.org/abs/2610.21621
- **HTML：** https://arxiv.org/html/2610.21621
- **PDF：** https://arxiv.org/pdf/2610.21621
- **官方项目页：** https://zevinwu.github.io/spear-project/

## 论文机制

多尺度视觉特征和 distinguishability predictor 筛选轨迹对，减少 VLM 的弃答；在采集的偏好数据上学习奖励，并采用合成奖励 warm-up→偏好奖励的自适应权重过渡、relay exploration 扩大有效覆盖。

## 实验与证据边界

论文在操作任务基准中报告平均成功率 88.9%，相对基线高 23.3 个百分点；使用 Mobile ALOHA 在 3 个真机任务上在线学习。真机运行仍限定工作空间并由人工复位。

## 复现与开放状态

arXiv 给出官方项目页，但本次工具未成功访问，源码和权重状态待核查。

## 局限

自动偏好并非无误的真实奖励，需要诊断 VLM 的弃答及偏差；项目页本次访问失败，暂不确认代码开放。

## 对 wiki 的映射

- [SPEAR：机器人强化学习中的选择性 VLM 偏好询问](../../wiki/entities/paper-spear-selective-preference.md)

## 参考来源

- [原论文 HTML](https://arxiv.org/html/2610.21621)
- [转载阅读线索（仅作索引）](https://mp.weixin.qq.com/s/gatO-d6IxsemBSk62EvS1w)
