# InternVLA-A1.5 原文核对 / Unifying Understanding, Latent Foresight, and Action for Compositional Generalization

- **类型：** paper
- **原文：** https://arxiv.org/html/2607.04988v1
- **官方项目：** https://internrobotics.github.io/internvla-a15.github.io
- **代码：** https://github.com/InternRobotics/InternVLA-A-series （默认 `master` 分支，A1.5；A1 另分支）
- **机构：** 上海人工智能实验室 Physical Intelligence Team
- **核对范围：** arXiv HTML §1–6 与附录（理解/未来潜码/动作、两阶段训练、采样、仿真/真机和限制）；官方 GitHub README 及 `launch/internvla_a15_pretrain.sh`、`launch/internvla_a15_finetune.sh`、`tests/openloop_internvla_a1_5.py`。
- **通读：** 已完成

## 方法与论文边界

- **Qwen3.5-2B + 460M expert**：Stage 1 通过 VQA/离散 FAST 等维持语义；Stage 2 用 50 foresight token 在训练期向冻结 Wan2.2 视频生成器蒸馏短期动力学，flow action head 生成连续 chunk；部署时不需要视频生成器。
- 论文报告 LIBERO **98.9**、RoboTwin **93.2**、LIBERO-Plus **84.8**、DOMINO zero-shot **27.7**；属于各自基准，不应合并为一条成功率。
- 论文限制：未来监督仅覆盖一个 chunk 的短时域，不构成高层长程搜索规划；世界模型冻结，效果受原视频预训练覆盖所限。
- **代码开放状态已核对（2026-11-14）：** 官方 README 已标 [x] training/eval code，权重链接发布。旧节点若暗示只有论文则不再准确。
- 真实运行入口：`launch/internvla_a15_pretrain.sh` / `launch/internvla_a15_finetune.sh` 向 `src/lerobot/scripts/lerobot_train.py` 传递配置、数据和 `policy.type=internvla_a1_5`；`tests/openloop_internvla_a1_5.py` 通过 checkpoint/dataset 预测和绘制动作。

## 对 wiki 的映射

- [InternVLA-A1.5 唯一论文详情](../../wiki/entities/paper-internvla-a15-unified-vla.md)
