# otr-ebla/LegNav-Sim

> 来源归档

- **标题：** LegNav-Sim
- **类型：** repo
- **链接：** <https://github.com/otr-ebla/LegNav-Sim>
- **论文：** <https://arxiv.org/abs/2607.27922>
- **权重目录：** <https://github.com/otr-ebla/LegNav-Sim/tree/eb46ad1b6c3aae126542ad5a6ebc15439ef7aca0/checkpoints>
- **入库日期：** 2026-09-26
- **一句话说明：** JAX 2D LiDAR 社交导航仿真（LegNav）+ CALF/PPO/SAC/TQC 训练与 baseline 动物园 + TurtleBot4 `legnav/deployment/`；checkpoints 随仓发布。
- **沉淀到 wiki：** [`wiki/entities/paper-legnav-calf.md`](../../wiki/entities/paper-legnav-calf.md)

---

## 仓库入口（README，2026-09-26）

| 组件 | 说明 |
|------|------|
| 环境 | Python **3.11–3.13**（推荐 3.12）；`pip install -e .` |
| 快速评测 | `python -m legnav.evaluation.jax_eval_multi --algo sac --headless --steps 10` |
| 包结构 | `legnav/core` 仿真；`legnav/algorithms` 训练；`legnav/evaluation` 基准；`legnav/deployment` 真机 |
| 权重 | `checkpoints/{ppo,sac,tqc,tagd,navrep,vanilla_ppo}/`（`.msgpack`） |
| 许可 | 根目录未声明 SPDX LICENSE（入库日以 README 为准） |

---

## 开源边界

| 已发布 | 备注 |
|--------|------|
| 完整 JAX 仿真与 CALF 训练栈 | IROS 2026 workshop 官方仓 |
| 预训练 checkpoint 子目录 | 用户给定 commit 树可直达 |
| TurtleBot 4 推理脚本 | 零样本部署口径见论文/视频 |
| `legacy/` | 旧 Gymnasium/SB3 代码，仅供参考 |
