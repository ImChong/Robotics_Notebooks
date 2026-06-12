# VisualMimic（Stanford 视觉人形 Loco-Manipulation）

> 来源归档

- **标题：** VisualMimic
- **类型：** repo
- **来源：** Stanford University（VisualMimic Team）
- **链接：** <https://github.com/visualmimic/VisualMimic>
- **项目页：** <https://visualmimic.github.io/>
- **论文：** [arXiv:2509.20322](https://arxiv.org/abs/2509.20322) — 归档见 [`sources/papers/visualmimic_arxiv_2509_20322.md`](../papers/visualmimic_arxiv_2509_20322.md)
- **入库日期：** 2026-06-12
- **一句话说明：** VisualMimic 官方仓库：**Sim2Sim 推理管线** 与 **真机任务 checkpoint** 已部分发布；完整训练与 Sim2Real 代码承诺 **论文接收后全面开源**。
- **沉淀到 wiki：** [`wiki/entities/paper-notebook-visualmimic.md`](../../wiki/entities/paper-notebook-visualmimic.md)

---

## 核心定位

**VisualMimic** 实现 [arXiv:2509.20322](https://arxiv.org/abs/2509.20322) 的 **分层视觉 loco-manipulation** 框架：低层 **关键点跟踪器** + 高层 **深度 visuomotor 关键点生成器**，支持 push / lift / kick / dribble 等任务的 **仿真评测与真机零样本部署**（论文设定）。

---

## 发布进度（README Release Progress）

| 组件 | 状态 |
|------|------|
| Sim2Sim pipeline | ✅ |
| Checkpoints for real world tasks | ✅ |
| Low-level tracker training code | ⬜ 待发布 |
| Low-level tracker checkpoint | ⬜ 待发布 |
| High-level policy training code | ⬜ 待发布 |
| Sim2Real pipeline | ⬜ 待发布 |

> **Note（仓库原文）：** *Code will be fully open-sourced after paper acceptance.*

---

## 安装与 Sim2Sim 用法

- **环境：** Ubuntu 20.04；`conda create -n visualmimic python=3.8`；`pip install -r requirements.txt`
- **运行：**
  ```bash
  cd sim2sim
  python sim2sim.py --task kick_ball  # kick_box, push_box, lift_box
  ```

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [TWIST](../papers/humanoid_rl_stack_09_twist_teleoperated_whole_body_imitation_system.md) | 同作者线；低层 motion tracking 奖励同族 |
| [VideoMimic](./videomimic.md) | 视觉人形交互对照；偏环境技能非全身物体操作 |
| [ResMimic](../papers/resmimic_arxiv_2510_05070.md) | loco-manip 残差学习对照；依赖 MoCap 物体参考 |
| [VIRAL](../papers/viral-humanoid-visual-sim2real.md) | 规模化 RGB 视觉 sim2real 对照 |

## 对 wiki 的映射

- 实体页：[VisualMimic（论文）](../../wiki/entities/paper-notebook-visualmimic.md)
- 任务页：[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)
