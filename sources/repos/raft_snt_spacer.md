# snt-spacer/RAFT

> 来源归档

- **标题：** RAFT（Recurrent Asymmetric Fault Tolerant）官方实现
- **类型：** repo
- **代码：** <https://github.com/snt-spacer/RAFT>
- **论文：** [arXiv:2608.22976](https://arxiv.org/abs/2608.22976) — 归档见 [`sources/papers/raft_thruster_fault_arxiv_2608_22976.md`](../papers/raft_thruster_fault_arxiv_2608_22976.md)
- **入库日期：** 2026-08-26
- **一句话说明：** Isaac Lab 浮动平台上的非对称 PPO：critic 训练时吃 \(D_{gt}\)，actor 无故障传感器；Docker + `scripts/rsl_rl/train.py` / eval。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [RAFT 实体页](../../wiki/entities/paper-raft-thruster-fault.md) | 方法与评测 |
| [Privileged Training](../../wiki/concepts/privileged-training.md) | 非对称 critic，无蒸馏 |
| [PPO](../../wiki/methods/ppo.md) | 优化器 |

## 复现入口（README 摘要）

并排克隆 Isaac Lab fork、rsl_rl fork 与本仓后：

```bash
docker/container.py start
docker/container.py enter
# 容器内
python scripts/rsl_rl/train.py
python scripts/rsl_rl/eval_gt_failures.py
python scripts/rsl_rl/eval_mid_episode_failures.py
```

README 还列出 `paper_checkpoints/RAFT/seed_{42,7,1337}.pt` 与全方法档案路径。

## 开源状态

**已开源** — 训练、评测与实验脚本齐全；依赖自有 Isaac Lab / rsl_rl fork 与 Docker GPU 镜像。
