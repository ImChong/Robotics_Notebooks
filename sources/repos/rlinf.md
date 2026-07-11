# RLinf（具身智能 RL 训练系统）

> 来源归档

- **标题：** RLinf — Embodied / Agentic RL Infrastructure
- **类型：** repo
- **组织：** 清华大学、北京大学等（见 GitHub / 深蓝 VLA 复现景观）
- **代码：** <https://github.com/RLinf/RLinf>
- **文档：** <https://rlinf.readthedocs.io/>
- **STEAM 示例：** <https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/steam.html>
- **入库日期：** 2026-07-11
- **一句话说明：** 大规模具身/智能体 **RL 训练系统**（弹性流水线、自适应通信、调度）；内置 **RECAP** 与 **STEAM** 等 **离线策略优化** 管线，对接 **OpenPI（π₀/π₀.₅）**、**LeRobot** 数据与 **Maniskill/LIBERO** 环境栈。
- **沉淀到 wiki：** [STEAM（论文实体）](../../wiki/entities/paper-steam-advantage-modeling.md)

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [VLA 开源复现景观](../../wiki/overview/vla-open-source-repro-landscape-2025.md) | **RLinf** 列为「搭集群 RL 基建」入口；**≠** 单一策略 checkpoint 仓库 |
| [STEAM](../../wiki/entities/paper-steam-advantage-modeling.md) | 论文官方实现：**三阶段 offline pipeline**（critic SFT → ensemble advantage → CFG policy） |
| [VLA](../../wiki/methods/vla.md) | π-RL、advantage-conditioned 后训练与 **RECAP/STEAM** 经验学习 |
| [LeRobot](../../wiki/entities/lerobot.md) | STEAM/RECAP 数据格式：**sft** 与 **rollout** 混合 LeRobot 数据集 |
| [强化学习](../../wiki/methods/reinforcement-learning.md) | 系统层支撑具身 RL，与 **SimpleVLA-RL** 等算法仓库互补 |

## STEAM 管线要点（文档摘要）

1. **Value Model SFT：** SigLIP-so400m + Gemma3-270M + 分类头；`ensemble_size` 成员；帧对 stride `data.k`。
2. **Compute Ensemble Advantages：** worst-of-N → `advantage_continuous`；`threshold` 或 `quantile` 二值化；写 `meta/advantages_{tag}.parquet`。
3. **CFG Training：** 与 RECAP 共享 `cfg_rl_openpi`；高 advantage 为 conditional、低为 unconditional。

**依赖：** `requirements/install.sh embodied --model openpi` 或 Docker `rlinf/rlinf:agentic-rlinf0.2-maniskill_libero`；`switch_env openpi`。

## 为何值得保留

- **论文→工程闭环：** STEAM（arXiv:2606.29834）与 RECAP 的 **可复现三阶段脚本** 集中在此仓 `examples/offline_rl/`。
- **系统 vs 算法分工：** 与 OpenPI（策略）、SimpleVLA-RL（veRL 算法扩展）形成 **栈式索引**，避免把 RLinf 误当策略权重源。
