# rl-sim2sim-demo-website

> 来源归档（ingest）

- **标题：** Robotics RL Sim2Sim Demo Website
- **类型：** site
- **链接：** https://imchong.github.io/RL_Sim2Sim_Demo_Website/index.html
- **关联仓库：** https://github.com/ImChong/RL_Sim2Sim_Demo_Website
- **上游改造自：** https://github.com/Axellwppr/humanoid-policy-viewer
- **入库日期：** 2026-06-07
- **一句话说明：** Vue 3 + MuJoCo WASM + ONNX 的浏览器 Sim2Sim 演示站，聚合 G1 AMP 走跑起身、PHP 感知跑酷与 Axellwppr 全身 tracking 三条策略。

## 演示策略（Policy 下拉）

| ID | 标题 | 训练/论文入口 |
|----|------|---------------|
| `g1-amp-walk-run-getup` | G1 AMP Walk/Run/Getup | [AMP_mjlab](https://github.com/ccrpRepo/AMP_mjlab) |
| `g1-parkour` | G1 Perceptive Parkour | [PHP 项目页](https://php-parkour.github.io/index.html)（内嵌 iframe） |
| `g1-tracking-latest` | G1 Tracking | [Axellwppr/motion_tracking](https://github.com/Axellwppr/motion_tracking) |

## 对 wiki 的映射

- [sim2real](../../wiki/concepts/sim2real.md) — Sim2Sim 作为迁移前仿真验证入口
- [amp-mjlab](../../wiki/entities/amp-mjlab.md) — AMP 走跑起身策略
- [Perceptive Humanoid Parkour](../../wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md) — 感知跑酷
- [GentleHumanoid](../../wiki/methods/gentlehumanoid-motion-tracking.md) / [Axellwppr/motion_tracking](../../wiki/entities/axellwppr-motion-tracking.md) — 全身 tracking
