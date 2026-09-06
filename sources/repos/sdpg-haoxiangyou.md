# SDPG — Yale 视觉强化学习官方实现

> 来源归档（仓库 README 要点摘录）

- **标题：** SDPG (Stochastic Decoupled Policy Gradient for Visual RL)
- **类型：** repo
- **组织：** HaoxiangYou（Yale）
- **链接：** https://github.com/HaoxiangYou/SDPG
- **项目页：** https://haoxiangyou.github.io/sdpg-website/
- **论文：** arXiv:2605.26478
- **入库日期：** 2026-09-06
- **一句话说明：** Genesis + Hydra 驱动的视觉/状态 on-policy RL；`scripts/run.py` 训练与评估；含 egocentric 任务与 baselines（`externals/`  vendored rl_games、drqv2）。
- **沉淀到 wiki：** [wiki/entities/paper-sdpg-visual-rl-stochastic-decoupled.md](../../wiki/entities/paper-sdpg-visual-rl-stochastic-decoupled.md)

---

## 依赖与运行面（README）

- Python 3.11 conda 环境；`pip install -e ".[dev]"`
- 可选：`pip install -e "externals/Genesis[dev]"`
- 训练：`python scripts/run.py task=genesis/hopper agent=sdpg/genesis_hopper`
- 视觉：`task.config.vis_obs=True` + `agent=sdpg/genesis_hopper_vis`
- 评估：`train=False checkpoint=<path>`；远程无头评估可 `replay.py` 回放 `trajectory.pt`
- 日志：`logs/<backend>/<task>/<agent>/train/<timestamp>/`

---

## 能力边界

| 模块 | 说明 |
|------|------|
| `scripts/run.py` | Hydra 入口：训练 / play / checkpoint |
| `envs/genesis_env/` | 自定义 Genesis 环境指南 |
| `externals/` | rl_games、drqv2 等 baseline 与补丁 |
| SDPG 并行规模 | `num_base_envs * (num_action_perturbations + 1)` 总 env 数 |

---

## 开源状态

**已开源** — 安装、训练、评估、自定义环境文档齐全；论文 Under review。

## 交叉链接

- [sources/papers/sdpg_visual_rl_arxiv_2605_26478.md](../papers/sdpg_visual_rl_arxiv_2605_26478.md)
- [sources/sites/sdpg-haoxiangyou-website.md](../sites/sdpg-haoxiangyou-website.md)
