---
type: entity
tags: [framework, simulation, humanoid, sim2real, nvidia]
status: complete
updated: 2026-04-28
related:
  - ./kimodo.md
  - ./mimickit.md
  - ../methods/imitation-learning.md
  - ../methods/smp.md
  - ../methods/amp-reward.md
sources:
  - ../../sources/repos/protomotions.md
summary: "ProtoMotions 是 NVIDIA 开发的高性能 GPU 加速的人形机器人仿真与控制框架，是 MimicKit 的姊妹项目。"
---

# ProtoMotions: 大规模人形机器人仿真框架

**ProtoMotions**（现主要为 **ProtoMotions3**）是 NVIDIA Labs 开源的一款专为物理仿真角色和人形机器人设计的高性能学习框架。它是目前实现大规模运动模仿和通用策略训练的最前沿工具之一。

## 核心技术特点

- **GPU 加速仿真**：完全集成 NVIDIA 的物理仿真技术，支持在大规模并行环境（数万个并发代理）中进行强化学习。
- **数据驱动的运动学习**：原生支持 AMASS 等海量人类动作数据集，能够在极短时间内（如 12 小时 A100 训练）让角色学会海量复杂的运动技能。
- **全链路部署**：提供从仿真训练到真机部署（如 Unitree G1/H1）的完整 Pipeline。
- **PyRoki 重定向**：内置高效的运动重定向引擎，解决了人体动捕数据到异构机器人结构的快速映射。

## 与 MimicKit 的关系

**MimicKit** 和 **ProtoMotions** 是 [Xue Bin Peng（彭学斌）](./xue-bin-peng.md) 运动控制研究生态中的两个支柱：
- **MimicKit**：算法导向，更轻量级，专注于运动模仿（Motion Imitation）核心算法的整洁实现。
- **ProtoMotions**：框架导向，更重量级，侧重于高性能仿真、多 GPU 扩展性以及工业级的 Sim2Real 部署。

## 参考来源
- [sources/repos/protomotions.md](../../sources/repos/protomotions.md)
