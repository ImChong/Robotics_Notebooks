# humanoid-soccer（Agile Striker · Daffan/humanoid-soccer）

> 来源归档

- **标题：** Learning Agile Striker Skills — official code
- **类型：** repo（Isaac Gym / Booster Gym 训练）
- **来源：** UT Austin · Sony AI
- **链接：** <https://github.com/Daffan/humanoid-soccer>
- **项目页：** <https://humanoidsoccer.github.io>
- **论文：** <https://arxiv.org/abs/2512.06571>
- **入库日期：** 2026-07-28
- **一句话说明：** ICRA 2026 Agile Striker 官方实现：`run.py` 单入口串联 PPO → DAgger → P3O 四阶段；基于 Booster Gym，面向 Booster T1。
- **开源状态：** **已开源**（2026-07-28）；根目录有 `LICENSE`（GitHub SPDX：`NOASSERTION`）。
- **沉淀到 wiki：** [`wiki/entities/paper-notebook-learning-agile-striker-skills-for-humanoid-socce.md`](../../wiki/entities/paper-notebook-learning-agile-striker-skills-for-humanoid-socce.md)

## 仓库概况（2026-07-28）

| 字段 | 值 |
|------|-----|
| 托管 | GitHub（`Daffan/humanoid-soccer`） |
| 入口 | `run.py --alg {PPO,DAgger,P3O,Player,Odom} --config configs/kick/*.yaml` |
| 依赖 | Isaac Gym Preview 4 · PyTorch 2.0 · Booster Gym 资产 |

## README 课程映射

| Stage | 算法 | 配置 |
|-------|------|------|
| 1 追球 | PPO | `configs/kick/T1_prekick.yaml` |
| 2 定向踢 | PPO | `configs/kick/T1_kick.yaml` |
| 3 蒸馏 | DAgger | `configs/kick/T1_kick_dagger.yaml` |
| 4 精修 | P3O | `configs/kick/T1_kick_adaptation_p3o.yaml` |
| 消融 | PPO | `T1_kick_adaptation_ppo.yaml` |

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 论文实体 | [`paper-notebook-learning-agile-striker-skills-for-humanoid-socce.md`](../../wiki/entities/paper-notebook-learning-agile-striker-skills-for-humanoid-socce.md) |
| 项目页 | [`humanoidsoccer-agile-striker.md`](../sites/humanoidsoccer-agile-striker.md) |
| 任务 | [`humanoid-soccer.md`](../../wiki/tasks/humanoid-soccer.md) |
