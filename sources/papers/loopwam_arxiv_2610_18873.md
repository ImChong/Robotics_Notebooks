# LoopWAM: Closed-Loop Generalist Policy Improvement from Deployment Rollouts（arXiv:2610.18873）

> 来源归档（ingest）

- **标题：** LoopWAM: Closed-Loop Generalist Policy Improvement from Deployment Rollouts
- **类型：** paper
- **作者：** Wenxuan Guo、Yingping Liang、Zhaoguo Wang、Ziyuan Li、Angyuan Ma、Zhenyu Wu、Chengkun Wang、Bingyao Yu、Ze Chen、Borui Zhang、Xiuwei Xu、Meng Zhang、Erjin Zhou、Jiwen Lu
- **arXiv：** <https://arxiv.org/abs/2610.18873>（v1，2026-10-13）
- **项目页：** <https://loop-wam.github.io/>
- **代码：** <https://github.com/loop-wam/LoopWAM>（公开仓库含 README、训练与评测入口说明）
- **入库日期：** 2026-10-14
- **一句话说明：** 将部署中成功、次优和失败 rollout 一并用于联合视频–动作世界模型训练，通过质量条件与加权监督闭环提升通用策略。

## 核心摘录（面向 wiki 编译）

1. **混合质量经验：** 使用自身及其他已部署策略的 rollout；对所有质量轨迹保留视频世界建模监督，动作模仿则按质量调节，失败轨迹不直接强化错误动作。
2. **联合去噪：** 视频与动作 token 在并行去噪的每层交互，使预测后果能反过来影响动作生成；源依赖噪声采样统一行为学习与动作条件世界建模。
3. **部署迭代：** 在 RoboChallenge Table30-V2 中，三轮部署与训练把成功率从 23.67% 提升到 43.33%，提升 19.7 个百分点，最终综合分 56.80。
4. **跨基准：** RoboTwin 2.0 平均成功率 94.5%，RoboCasa365 总体成功率 49.5%；这两项不增加具身预训练。
5. **边界：** 项目仓库已公开；不同 benchmark 的任务数、成功率和综合分不可混作同一指标，真机收益来自特定 Table30-V2 设置。

**对 wiki 的映射：** [paper-loopwam](../../wiki/entities/paper-loopwam.md)

## 版本与引用

- arXiv v1，提交于 2026-10-13。
- 来源以 arXiv 论文摘要、项目页和官方仓库 README 为准。
---
