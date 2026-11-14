# Learning Low-Latency Action Verification for Frozen Vision-Language-Action Models

> 来源归档（ingest），来自 2026-11-13「具身智能小站」18 篇论文阅读清单；以原始 arXiv 及官方项目和仓库为核查依据。

## 元数据

- **标题：** Learning Low-Latency Action Verification for Frozen Vision-Language-Action Models
- **中文：** HDH：冻结 VLA 的低延迟动作验证
- **arXiv：** 2611.14834
- **版本：** 2026-11-14 检查 arXiv HTML；具体修订号待逐项补齐
- **入库日期：** 2026-11-14
- **核对范围：** 已阅读 arXiv HTML 的摘要、背景、方法、实验、结论、局限及可用附录；公开源码仅核查官方 README 与实际入口，未独立运行。
- **通读：** 已完成
- **论文：** https://arxiv.org/abs/2611.14834
- **HTML：** https://arxiv.org/html/2611.14834
- **PDF：** https://arxiv.org/pdf/2611.14834
- **官方项目页：** https://wen-hanhsieh.github.io/hdh/
- **仓库：** https://github.com/Wen-HanHsieh/hdh_real

## 论文机制

冻结 π0.5 收集自主 rollouts，按回合终止成败标注当前观测与执行动作前缀；训练带历史条件的轻量 verifier；推理时采样两个候选块，按成功分数选择，并执行获选块的前 8 步。

## 实验与证据边界

四个 LIBERO-10 任务平均成功率 82.3%；Franka Panda 三项真机任务的平均成功率由 44.7% 升至 67.3%（每方法每任务 50 次）。5.4% 额外延迟为仿真统计，真机延迟需要独立测量。

## 复现与开放状态

已公开 Franka 真机代码；另外有 hdh_sim 仿真仓库。

## 局限

验证器依赖任务自身的 rollout 成败标签；候选只来自原冻结策略，无法保证发现策略从未采样的动作。

## 对 wiki 的映射

- [HDH：冻结 VLA 的低延迟动作验证](../../wiki/entities/paper-hdh-action-verifier.md)

## 参考来源

- [原论文 HTML](https://arxiv.org/html/2611.14834)
- [转载阅读线索（仅作索引）](https://mp.weixin.qq.com/s/gatO-d6IxsemBSk62EvS1w)
