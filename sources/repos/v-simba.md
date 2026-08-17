# DAVIAN-Robotics/V-Simba

- **标题：** V-Simba 官方实现
- **类型：** repo
- **URL：** <https://github.com/DAVIAN-Robotics/V-Simba>
- **许可：** Apache-2.0
- **配套论文：** [arXiv:2608.07870](https://arxiv.org/abs/2608.07870) — [`sources/papers/v_simba_arxiv_2608_07870.md`](../papers/v_simba_arxiv_2608_07870.md)
- **入库日期：** 2026-08-17

## 一句话说明

视觉连续控制的 Simba 风格 SAC：`scale_rl/agents/vsimba/`，对照 DrQ-v2；DMC / Adroit / Meta-World 脚本。

## 仓库状态（2026-08-17 核查）

| 项 | 内容 |
|----|------|
| 单实验 | `uv run python run_online.py --overrides env.env_name=cheetah-run` |
| 并行 | `run_parallel.py --env_type vsimba_1M` |
| 复现 | `scripts/vsimba_dmc.sh` / `vsimba_adroit.sh` / `vsimba_metaworld.sh` |
| 配置 | `configs/agent/vsimba.yaml`、`configs/agent/drqV2.yaml` |

最短复现：`uv python pin`（按 GPU 选 3.10/3.11）→ `uv sync` → `uv run python run_online.py --overrides env.env_name=cheetah-run`。

## 与 wiki 的关系

- 实体页：[paper-v-simba](../../wiki/entities/paper-v-simba.md) — 含源码运行时序图。
