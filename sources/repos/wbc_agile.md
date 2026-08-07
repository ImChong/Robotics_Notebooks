# WBC-AGILE（AGILE）

> 来源归档

- **标题：** AGILE — A Generic Isaac-Lab based Engine for humanoid loco-manipulation learning
- **类型：** repo
- **来源：** NVIDIA（`nvidia-isaac` 组织）
- **链接：** <https://github.com/nvidia-isaac/WBC-AGILE>
- **文档：** <https://nvidia-isaac.github.io/WBC-AGILE/>
- **论文：** <https://arxiv.org/abs/2603.20147>（Submitted 2026-03-20）
- **许可：** Apache-2.0（`agile/algorithms/rsl_rl/` 为 BSD 3-Clause，源自 ETH RSL-RL）
- **入库日期：** 2026-08-07
- **一句话说明：** NVIDIA 开源的人形全身控制 RL 工作流：在 Isaac Lab 上覆盖 Prepare / Train / Evaluate / Deploy，含确定性评测、MuJoCo Sim2Sim、描述符导出与 G1/T1 任务配置。
- **沉淀到 wiki：** [`wiki/entities/paper-agile-humanoid-loco-manipulation.md`](../../wiki/entities/paper-agile-humanoid-loco-manipulation.md)

---

## 核心定位

相对「只给训练脚本」的人形 RL 仓库，AGILE 强调 **生命周期基础设施**：交互式 MDP 核验、可复现训练与算法工具箱、统一运动质量评测、以及 YAML I/O 描述符驱动的跨仿真/真机推理。

---

## 依赖与入口（README，2026-08-07）

| 项 | 说明 |
|----|------|
| 前置 | [Isaac Lab v2.3.2](https://isaac-sim.github.io/IsaacLab/v2.3.2/) + Isaac Sim 5.1 |
| 安装 | `export ISAACLAB_PATH=...` → `./scripts/setup/install_deps_local.sh` |
| 训练 | `python scripts/train.py --task Velocity-T1-v0 --num_envs 2048 --headless` |
| 评测 | `python scripts/eval.py --task Velocity-T1-v0 --num_envs 32 --checkpoint <path>` |
| 任务示例 | `Velocity-T1-v0` / `Velocity-G1-History-v0` / `Velocity-Height-G1-v0` / `StandUp-T1-v0` / `G1-PickPlace-Tracking-v0` / `Tracking-Flat-G1-v0` / `Debug-*-v0` |
| 远程 | OSMO workflow（集群训练 / 评测 / sweep） |
| 补充 | Office Hour 录像与 `OFFICE_HOUR_FAQ.md` |

---

## 开源边界

- **已发布：** 任务配置、训练/评测脚本、算法增强模块、文档站、演示素材与预训练相关入口（以仓库 README/docs 为准）。
- **论文说明待另发：** 完整真机 sim-to-real 驱动管线「will also be released in the near future separately」——选型时勿默认仓库已含全部硬件驱动。

---

## 关联档案

| 档案 | 关系 |
|------|------|
| [agile_arxiv_2603_20147.md](../papers/agile_arxiv_2603_20147.md) | 论文摘录 |
| [wbc-agile-docs.md](../sites/wbc-agile-docs.md) | 文档站 |
| [isaac_lab.md](./isaac_lab.md) | 仿真/学习底座 |
| [amp_rsl_rl.md](./amp_rsl_rl.md) | RSL-RL 谱系对照 |

## 对 wiki 的映射

- 实体页：[AGILE（论文）](../../wiki/entities/paper-agile-humanoid-loco-manipulation.md)
- 底座：[Isaac Lab](../../wiki/entities/isaac-lab.md)
- 任务：[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)
