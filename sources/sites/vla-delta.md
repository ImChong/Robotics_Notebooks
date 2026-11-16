# VLA-Δ 官方项目页 / Official Project Page

- **主页：** <https://zju-omniai.github.io/VLA-Delta/>
- **论文：** <https://arxiv.org/abs/2610.21306>（v1）
- **代码：** <https://github.com/ZJU-OmniAI/VLA-Delta>
- **权重：** [base](https://huggingface.co/t0rn0d0/VLA-Delta-base)、[LIBERO](https://huggingface.co/t0rn0d0/VLA-Delta-libero)、[RoboTwin](https://huggingface.co/t0rn0d0/VLA-Delta-robotwin)
- **核验日期：** 2026-11-16
- **核对范围：** 官方页面 Overview、Method、Results、Real robot、Citation 及链接；对照 GitHub README 的权重和训练入口。
- **通读：** 已完成
- **开放程度：** **已开源训练与评测实现及权重**；完整 484h 物理转移标注数据、原始标注全流程的公开可复现程度待核实。
- **一句话说明：** 用任务语义差异、状态变量和运动变化训练 Physical State Difference Queries。

## 页面核查结果

- Qwen3-VL-2B-Instruct + 九层 DiT 动作专家；九类状态槽、光流和差异文本三种训练监督。
- 数据：DROID、MolmoAct、RH20T 约 484 小时/2400 万帧。
- LIBERO 98.4%；RoboTwin Clean / Random 90.06% / 89.74%；真机五任务平均 68.0%。数据均为作者实验。
- “论文使用的示教规模”和“可公开下载的带物理标签数据”不能等同。

## 关联归档

- [论文原文与实验协议](../papers/vla_delta_arxiv_2610_21306.md)
- [官方代码与运行入口](../repos/vla-delta.md)
- [VLA-Δ 实体页](../../wiki/entities/paper-vla-delta.md)
