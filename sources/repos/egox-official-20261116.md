# EgoX: Egocentric Cross-Embodiment Manipulation with Embodiment Dreaming — 公开仓库运行入口核对

- **仓库：** https://github.com/egox-dream/egox
- **日期：** 2026-11-16
- **开放状态：** 已部分开源：训练与 G1 仿真评测可用；数据、VR 管线、示例 checkpoint 及大部分真机部署未发布。
- **对应论文：** [arXiv:2611.16728](https://arxiv.org/abs/2611.16728)
- **项目页：** https://egox-dream.github.io/
- **原文归档：** [论文](../papers/egox-arxiv-2611-16728.md)

## 已核对的实际运行路径

- `scripts/training_scripts/G1-Sim-Dex3/kitchen-Bread/pretrain.sh` 设置跨本体数据路径并调用 `scripts/train.py`。
- `scripts/rollout_scripts/G1-Sim-Dex3/kitchen-Bread/headless.sh` 调用 `scripts/rollout.py sim --mode headless --benchmark kitchen_bread`。
- 预训练与模拟评测运行需要自备匹配数据和产出的 `policy_last.ckpt`；没有官方示例数据/checkpoint。
