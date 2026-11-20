# Baojch / phygile_tracking（PhyGile GMT 跟踪器仓库）

> 来源归档（2026-11-20 新增核查）

- **仓库：** <https://github.com/Baojch/phygile_tracking>
- **项目页：** <https://baojch.github.io/phygile-page/>
- **论文：** <https://arxiv.org/abs/2603.19305>
- **类型：** repo
- **入库日期：** 2026-11-20
- **通读：** 已完成
- **核对范围：** 2026-11-20 克隆并通读 `Baojch/phygile_tracking` 官方 `README.md`（L1–362 全文）、RSL-RL 训练入口 `scripts/rsl_rl/train.py`（L1–100）与回放入口 `scripts/rsl_rl/play.py`（L1–100），核实其基于 `HybridRobotics/whole_body_tracking` / IsaacLab 2.1.0 实现两阶段 MoE/课程学习跟踪器训练与 ONNX 导出；明确记录 PhyGile 论文中的 `TP-MoE` 262D 生成器与 `physics-prefix` 闭环微调代码未包含在本仓库中。

## 仓库定位与边界（步骤 2.5）

| 组件 | 开源状态 | 说明 |
|---|---|---|
| GMT MoE 课程跟踪器训练/评估 | **已开源** | `scripts/rsl_rl/train.py`、`scripts/rsl_rl/play.py`（含 `--phase_b` 支持）与 `source/whole_body_tracking/` |
| TP-MoE 262D 文本→运动生成器 | **未开源** | 截至 2026-11-20 未包含于 `phygile_tracking`，亦无独立生成器仓库链接 |
| Physics-prefix 闭环微调代码 | **未开源** | 未在公开仓库释放 |

## 关键运行时入口（实测源码）

- 训练入口：`scripts/rsl_rl/train.py`（通过 `--task`、`--motion_file` / `--motion_dir`、`--g1_training_cfg` 与 `AppLauncher` 启动 IsaacLab + RSL-RL PPO/MoE 训练）。
- 回放与 ONNX 导出：`scripts/rsl_rl/play.py`（支持 `--play_full_motion`、`--phase_b` 激活全部 MoE 专家及 `export_motion_policy_as_onnx`）。

## 对 wiki 的映射

- 论文实体页：[paper-phygile](../../wiki/entities/paper-phygile.md)
- 论文来源归档：[phygile_arxiv_2603_19305](../papers/phygile_arxiv_2603_19305.md)
- 项目页归档：[phygile-page](../sites/phygile-page.md)
