---
type: entity
tags: [repo, humanoid, motion-tracking, mjlab, unitree-g1, sim2real, vr-teleoperation, compliance]
status: complete
updated: 2026-05-19
related:
  - ../methods/gentlehumanoid-motion-tracking.md
  - ../methods/motion-retargeting-gmr.md
  - ./mjlab.md
  - ./unitree-g1.md
sources:
  - ../../sources/repos/axellwppr_motion_tracking.md
  - ../../sources/sites/motion-tracking-axell-top.md
summary: "Axellwppr/motion_tracking：GentleHumanoid 论文对应的 mjlab 全身跟踪训练/评估/部署仓，含 AMASS/LAFAN 数据管线、WandB 训练、ONNX sim2real 与 VR 遥操作配置。"
institutions: [unitree]

---

# Axellwppr/motion_tracking

**[Axellwppr/motion_tracking](https://github.com/Axellwppr/motion_tracking)** 是 GentleHumanoid 论文作者维护的 **全身运动跟踪** 工程仓库：在 GentleHumanoid 研究代码基上，用 **mjlab** 完成仿真训练，并提供 **评估、ONNX 导出、sim2real 与 VR 遥操作** 文档。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| Sim2Real | Simulation to Real | 把仿真中学到的策略迁移落地真机的工程主线 |
| AMASS | Archive of Motion Capture as Surface Shapes | 大规模统一人体动捕数据集 |
| ONNX | Open Neural Network Exchange | 跨框架神经网络模型交换格式 |
| Isaac Lab | NVIDIA Isaac Lab | 基于 Omniverse 的机器人学习训练框架 |
| AMP | Adversarial Motion Prior | 用对抗判别约束状态转移接近专家运动分布的先验 |
| GMR | General Motion Retargeting | 把人体/视频动作重定向为机器人可执行参考 |
| G1 | Unitree G1 Humanoid | 宇树入门级教育科研人形平台 |

## 为什么单独成页？

- **论文页 ≠ 训练脚本**：[GentleHumanoid 项目页](https://gentle-humanoid.axell.top/) 侧重方法与演示；本仓给出 **可复现的训练命令、数据集目录约定与部署 checklist**。
- **与社区 Isaac 栈分流**：许多 tracking 项目基于 Isaac Lab；本仓明确 **mjlab + uv**，与 [mjlab](./mjlab.md)、[AMP_mjlab](./amp-mjlab.md) 等同生态对照。
- **GMR 分叉**：[Axellwppr/GMR](https://github.com/Axellwppr/GMR) 直接导出本仓所需 npz 字段，是数据准备的关键依赖。

## 仓库能力摘要

| 模块 | 说明 |
|------|------|
| **训练** | `train.sh` + `cfg/task/G1/G1.yaml`；默认约 4×A100、15 h；可调 `NPROC` / `num_envs` |
| **数据** | 预处理后 `dataset/amass_all`、`lafan_all`；或 AMASS/LAFAN + GMR + `generate_dataset.sh` |
| **评估** | `scripts/eval.py`；`-p` 播放，`-p --export` 导出 ONNX |
| **部署** | `sim2real/`：策略拷至 `assets/ckpts/`，改 `tracking.yaml`；README 含 sim2sim/sim2real、UDP selector、VR source |
| **演示** | [motion-tracking.axell.top](https://motion-tracking.axell.top/) 浏览器预览预训练策略 |

## 关联页面

- [GentleHumanoid（方法页）](../methods/gentlehumanoid-motion-tracking.md) — 阻抗柔顺 tracking 论文提炼
- [GMR（运动重定向）](../methods/motion-retargeting-gmr.md) — 数据重定向基线
- [mjlab](./mjlab.md) — 仿真与训练后端
- [Unitree G1](./unitree-g1.md) — 目标硬件平台

## 参考来源

- [sources/repos/axellwppr_motion_tracking.md](../../sources/repos/axellwppr_motion_tracking.md)
- [sources/sites/motion-tracking-axell-top.md](../../sources/sites/motion-tracking-axell-top.md)

## 推荐继续阅读

- [RL Sim2Sim 在线演示：G1 Tracking](https://imchong.github.io/RL_Sim2Sim_Demo_Website/index.html)
- [GentleHumanoid 项目页](https://gentle-humanoid.axell.top/)
- [arXiv:2511.04679](https://arxiv.org/abs/2511.04679)
