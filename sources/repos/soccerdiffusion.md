# SoccerDiffusion（bit-bots/SoccerDiffusion）

> 来源归档

- **标题：** SoccerDiffusion
- **类型：** repo
- **来源：** University of Hamburg / Hamburg Bit-Bots
- **链接：** <https://github.com/bit-bots/SoccerDiffusion>
- **项目页：** <https://bit-bots.github.io/SoccerDiffusion/>
- **论文：** <https://arxiv.org/abs/2504.20808>
- **许可：** MIT
- **入库日期：** 2026-07-28
- **一句话说明：** 从 RoboCup 真机比赛录像训练 transformer 扩散端到端控制；`poetry` + `cli` 入口；含数据集工具与蒸馏流程。
- **开源状态：** **已开源**（MIT；README 标注 ongoing research）。
- **沉淀到 wiki：** [`wiki/entities/paper-notebook-soccerdiffusion-toward-learning-end-to-end-human.md`](../../wiki/entities/paper-notebook-soccerdiffusion-toward-learning-end-to-end-human.md)

## 仓库概况（2026-07-28）

| 字段 | 值 |
|------|-----|
| 包目录 | `soccer_diffusion/` |
| 安装 | `poetry install --without test,dev` |
| 入口 | `cli --help`（poetry shell 内） |
| 可选 | ROS 2（`recording2mcap`）、B-Human 日志导入 |

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 论文实体 | [`paper-notebook-soccerdiffusion-toward-learning-end-to-end-human.md`](../../wiki/entities/paper-notebook-soccerdiffusion-toward-learning-end-to-end-human.md) |
| 项目页 | [`bit-bots-soccerdiffusion.md`](../sites/bit-bots-soccerdiffusion.md) |
| 任务 | [`humanoid-soccer.md`](../../wiki/tasks/humanoid-soccer.md) |
