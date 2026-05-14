---
type: entity
tags: [repo, diffusion, motion-generation, humanoid, nvidia, unitree-g1, soma, smpl-x]
status: complete
updated: 2026-05-14
related:
  - ../methods/diffusion-motion-generation.md
  - ../methods/motionbricks.md
  - ../entities/protomotions.md
  - ../entities/unitree-g1.md
  - ../concepts/motion-retargeting.md
sources:
  - ../../sources/repos/kimodo.md
summary: "Kimodo 是 NVIDIA nv-tlabs 开源的运动扩散模型与工具链：在大规模动捕上训练，支持文本与多种运动学约束生成 SOMA / G1 / SMPL-X 骨架的高质量轨迹，并提供 CLI、时间线 Demo 与公开评测基准。"
---

# Kimodo（可控人体与人形运动扩散）

**Kimodo**（**Ki**nematic **Mo**tion **D**iffusi**o**n）把 **扩散式生成** 用于 **运动学空间** 的全身轨迹合成：在约 **700 小时** 量级的商业友好型光学动捕上训练，输出可落在 **SOMA、Unitree G1、SMPL-X** 等骨架上；条件除自然语言外，还支持全身姿态关键帧、末端位姿、地面 2D 路径与路点等 **显式运动学约束**。

## 为什么重要？

- **约束 + 文本的可控生成**：把「导演式」关键帧编辑与扩散采样结合，适合作为动画管线或机器人 **参考轨迹生成器** 的上游。
- **与 NVIDIA 人形栈对齐**：官方 README 给出与 [ProtoMotions](./protomotions.md)、MuJoCo、[GEAR-SONIC 在线 Demo](https://nvlabs.github.io/GEAR-SONIC/demo.html) 的衔接示例，和 [MotionBricks](../methods/motionbricks.md) 所代表的「生成式意图层」形成互补阅读。
- **可复现评测**：发布 [Kimodo Motion Generation Benchmark](https://huggingface.co/datasets/nvidia/Kimodo-Motion-Gen-Benchmark) 与基于 BONES-SEED 的测试用例，便于横向对比文本对齐与约束跟随指标。

## 工程要点（速览）

- **推理入口**：CLI（`kimodo_gen`）与本地 Gradio Demo（`kimodo_demo`）；支持 CFG 类型与扩散步数等推理旋钮。
- **显存**：README 提示全 GPU 路径约需 **17GB VRAM**（文本编码占大头），可通过 `TEXT_ENCODER_DEVICE=cpu` 换用 CPU 编码以换延迟换显存。
- **导出格式**：默认 NPZ；G1 分支可写 **MuJoCo qpos CSV**；SMPL-X 分支可写 **AMASS 兼容 npz**，便于 [General Motion Retargeting](https://github.com/YanjieZe/GMR) 等下游重定向。

## 关联页面

- [Diffusion-based Motion Generation](../methods/diffusion-motion-generation.md) — 扩散式全身轨迹生成的范式定位
- [MotionBricks](../methods/motionbricks.md) — 同生态下的实时潜空间生成式运动栈
- [ProtoMotions](./protomotions.md) — 将生成轨迹导入可微仿真与 RL 的典型路径
- [Unitree G1](./unitree-g1.md) — Kimodo-G1 变体与 MuJoCo 可视化脚本所面向的平台

## 参考来源

- [sources/repos/kimodo.md](../../sources/repos/kimodo.md)

## 推荐继续阅读

- [Kimodo 技术报告 PDF](https://research.nvidia.com/labs/sil/projects/kimodo/assets/kimodo_tech_report.pdf)
- [Kimodo 官方文档](https://research.nvidia.com/labs/sil/projects/kimodo/docs)
- [arXiv:2603.15546](https://arxiv.org/abs/2603.15546) — *Kimodo: Scaling Controllable Human Motion Generation*
