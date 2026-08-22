# noitom-robotics/AdaPT

- **标题：** AdaPT 官方实现（Adaptive Motion Planning and Tracking）
- **类型：** repo
- **URL：** <https://github.com/noitom-robotics/AdaPT>
- **许可：** Apache-2.0
- **配套论文：** [arXiv:2608.20087](https://arxiv.org/abs/2608.20087) — [`sources/papers/adapt_arxiv_2608_20087.md`](../papers/adapt_arxiv_2608_20087.md)
- **项目页：** <https://humanoidtennis.github.io/AdaPT/>
- **入库日期：** 2026-08-22

## 一句话说明

基于 **mjlab** 的人形网球 **Stage1 发球速度自适应跟踪** 训练与 play 入口；`uv sync` 建环境，`uv run train/play Mjlab-ServeTracking-Flat-Unitree-G1-Stage1-RandomDt`。

## 仓库状态（2026-08-22 核查）

| 项 | 内容 |
|----|------|
| 环境 | `uv sync`（`pyproject.toml` + `uv.lock`）；Linux 默认 CUDA 12.x PyTorch |
| Stage1 训练 | `uv run train Mjlab-ServeTracking-Flat-Unitree-G1-Stage1-RandomDt --env.commands.motion.motion-file dataset/player1/p1_serve.npz ...` |
| Stage1 推理 | `uv run play ... --checkpoint-file ckpts/player1/model_24000.pt` |
| 配置 | `src/mjlab/tasks/adapt_tennis/stage1_tracking_env_cfg.py`（`HIT_ARM_KEYFRAME_TIMES_S` 等） |
| 预训练 | `ckpts/player1/model_24000.pt` |
| 未开源部分 | 对拉 MVAE 规划、完整 sim2real 部署（感知/定位） |

**结论：** **部分开源、可运行 Stage1 跟踪训练/推理**；非完整 AdaPT 管线。

## 与 wiki 的关系

- 实体页：[paper-adapt](../../wiki/entities/paper-adapt.md) — 含源码运行时序图（Stage1 范围）。
