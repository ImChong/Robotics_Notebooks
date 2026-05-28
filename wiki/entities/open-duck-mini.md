---
type: entity
tags: [biped, open-source, hardware, entertainment-robotics, sim2real, diy, disney-bdx]
status: complete
updated: 2026-05-28
related:
  - ./open-duck-playground.md
  - ./open-duck-reference-motion-generator.md
  - ./open-duck-mini-runtime.md
  - ../methods/disney-olaf-character-robot.md
  - ../concepts/sim2real.md
  - ../tasks/locomotion.md
  - ./open-source-humanoid-hardware.md
sources:
  - ../../sources/repos/open_duck_mini.md
summary: "Open Duck Mini 是 Disney BDX 双足角色的开源迷你复刻（v2 约 42 cm、BOM 目标 <$400）：Onshape CAD + Feetech 舵机 + 四仓分工（Hub / Playground / 参考运动 / Runtime），完整覆盖 CAD→MJX→RL→Pi Zero 2W 真机。"
---

# Open Duck Mini

**Open Duck Mini** 是社区驱动的 **BDX 风格迷你双足机器人**：在娱乐角色外形下，把 **低成本舵机硬件** 与 **MuJoCo Playground RL + Disney 式模仿奖励** 串成可复现的 sim2real 闭环。v2 为当前主线（`Open_Duck_Mini` 的 `v2` 分支）。

## 为什么重要

- **完整 DIY 栈：** 机械（Onshape/BOM）→ 准确 MJCF + BAM 电机辨识 → JAX 并行训练 → Pi Zero 2W 部署，适合学习「廉价执行器上的 sim2real」。
- **与 Disney 研究线对齐：** 参考运动与 imitation reward 直接承接 [BDX 论文](https://la.disneyresearch.com/publication/design-and-control-of-a-bipedal-robotic-character/) 思路，可与 [Disney Olaf 角色机器人](../methods/disney-olaf-character-robot.md) 对照阅读。
- **生态拆分清晰：** Hub、训练、运动生成、Runtime 分仓，便于单独 fork 或替换某一环。

## 四仓分工

| 仓库 | 职责 |
|------|------|
| [Open_Duck_Mini](https://github.com/apirrone/Open_Duck_Mini) | CAD、BOM、组装文档、社区入口、预训练 ONNX |
| [Open_Duck_Playground](./open-duck-playground.md) | MuJoCo Playground RL 环境与训练 |
| [Open_Duck_reference_motion_generator](./open-duck-reference-motion-generator.md) | Placo 参数化步态 → 模仿奖励系数 |
| [Open_Duck_Mini_Runtime](./open-duck-mini-runtime.md) | Raspberry Pi Zero 2W 机载推理与硬件驱动 |

## 流程总览

```mermaid
flowchart LR
  CAD[Onshape CAD\n+ BOM] --> EXP[onshape-to-robot\nURDF/MJCF]
  BAM[Rhoban BAM\nFeetech 辨识] --> MJCF[Playground MJCF\n执行器参数]
  EXP --> MJCF
  PLACO[Reference Motion Generator\nPlaco 步态 sweep] --> PKL[polynomial_coefficients.pkl]
  PKL --> PG[Open Duck Playground\njoystick + 模仿奖励]
  MJCF --> PG
  PG --> ONNX[ONNX 策略]
  ONNX --> RT[Mini Runtime\nPi Zero 2W]
  RT --> REAL[真机行走]
```

## 硬件要点（v2）

- **尺寸：** 腿伸展约 **42 cm**；BOM 目标 **低于 $400**。
- **执行器：** Feetech 总线舵机（腿部 `xc330-M288-T` 等）；辨识参数见 [BAM STS3215 7.4V](https://github.com/Rhoban/bam/tree/main/params/feetech_sts3215_7_4V)。
- **制造：** 3D 打印 + 公开 BOM；[Tnkr 组装指南](https://tnkr.ai/explore/docs/open-duck-mini/open-duck-mini-v2#home) 与社区 Discord。

## Sim2Real 关键步骤

1. **结构质量：** 切片器估算质量覆盖 Onshape 材料表（见 Hub 仓 `docs/sim2real.md`）。
2. **电机模型：** BAM 导出 `damping` / `kp` / `frictionloss` / `armature` / `forcerange` 写入 MJCF。
3. **策略训练：** Playground 中启用 BDX 风格 imitation reward + 域随机化；详见 [Open Duck Playground](./open-duck-playground.md)。
4. **部署：** 关节偏置标定 + Runtime checklist → ONNX 上机。

## 常见误区或局限

- **不是研究级人形：** 舵机扭矩与背隙限制动态性能；sim2real 高度依赖 BAM 与奖励调参。
- **v1 已过时：** alpha 版机械间隙大；新入门应直接跟 v2 与 Playground 栈。
- **文档仍在完善：** 表情功能（眼 LED、相机、麦克风）在路线图中，不影响核心行走闭环。

## 参考来源

- [sources/repos/open_duck_mini.md](../../sources/repos/open_duck_mini.md)
- [apirrone/Open_Duck_Mini](https://github.com/apirrone/Open_Duck_Mini)（v2 分支）
- Disney Research：[Design and Control of a Bipedal Robotic Character (BDX)](https://la.disneyresearch.com/publication/design-and-control-of-a-bipedal-robotic-character/)

## 关联页面

- [Open Duck Playground](./open-duck-playground.md)
- [Open Duck Reference Motion Generator](./open-duck-reference-motion-generator.md)
- [Open Duck Mini Runtime](./open-duck-mini-runtime.md)
- [Disney Olaf 角色机器人](../methods/disney-olaf-character-robot.md)
- [Sim2Real](../concepts/sim2real.md)
- [Locomotion](../tasks/locomotion.md)

## 推荐继续阅读

- [MuJoCo Playground 官方站点](https://playground.mujoco.org/)
- [Haonan Yu — Sim2Real 实践博客](https://www.haonanyu.blog/post/sim2real/)
- [开源人形机器人硬件方案对比](./open-source-humanoid-hardware.md)（全尺寸人形选型对照）
