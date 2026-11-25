# CommNav（S4b3/CommNav）

> 来源归档（repo）

- **标题：** CommNav — Communication-Enabled Social Navigation
- **类型：** repo / social-navigation / embodied-ai
- **来源：** S4b3（GitHub）
- **链接：** <https://github.com/S4b3/CommNav>
- **论文：** [arXiv:2607.01044](https://arxiv.org/abs/2607.01044) · IROS 2026（仓内 bibtex）
- **Stars：** 见 GitHub 实时状态（`pushed_at: 2026-09-26`，758 files，含 `v1.0.0` release）
- **入库日期：** 2026-08-09
- **复核日期：** 2026-11-21
- **一句话说明：** CommNav 官方开源仓（MIT，758 files）：包含 Habitat 3.0c 多智能体扩展（`multi_agent_sensors.py` 等）、COMM（numeric/text）训练代码、全套 `habitat-baselines/scripts/release/` 训练评测脚本及 `v1.0.0` 预训练权重。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-commnav.md`](../../wiki/entities/paper-commnav.md)
- **论文归档：** [`sources/papers/commnav_arxiv_2607_01044.md`](../papers/commnav_arxiv_2607_01044.md)

---

## 开源状态（步骤 2.5，2026-11-21 复核）

| 项 | 状态 |
|----|------|
| 训练 / 评测代码 | **已开源**（`habitat-baselines/scripts/release/{train_baseline,train_commnav,eval_commnav}.sh` 等） |
| Habitat 3.0c 任务与传感器 | **已开源**（`habitat-lab/habitat/tasks/rearrange/multi_agent_sensors.py`、`config/benchmark/multi_agent/*.yaml`） |
| COMM 实现与权重 | **已开源**（`scripts/release/train_comm_module_{numeric,text}.sh` + `v1.0.0` release 权重） |
| 人类口语研究原始包 | human-study 原始数据未随仓分发 |
| 许可证 | **MIT License** |

**结论：** **已完整开源（MIT + `v1.0.0` checkpoint + 官方发布启动脚本面）**；wiki 已补基于源码和 `EXPERIMENTS.md` 的「源码运行时序图」。

---

## 官方入口与启动脚本（`habitat-baselines/scripts/release/`）

| 阶段 | 官方脚本 / 入口 | 产出 |
|------|------------------|------|
| Baseline 导航策略 | `scripts/release/train_baseline.sh` | `checkpoints/baseline/` |
| 结构化 COMM 模块 | `scripts/release/train_comm_module_numeric.sh`（或 `communication_module_train.py`） | `comm_checkpoints/numeric/` |
| 文本 COMM 模块 | `scripts/release/train_comm_module_text.sh`（或 `communication_module_text_train.py`） | `comm_checkpoints/text/` |
| CommNav 联合微调 | `scripts/release/train_commnav.sh` | `checkpoints/training/comm_nav/` |
| 评测 | `scripts/release/eval_commnav.sh` | `checkpoints/evaluation/`（指标与视频） |

---

## 目录快照（2026-11-21 复核，758 files）

```
CommNav/
  README.md / LICENSE
  habitat-lab/habitat/
    tasks/rearrange/multi_agent_sensors.py
    config/benchmark/multi_agent/hssd_spot_human*.yaml
  habitat-baselines/
    EXPERIMENTS.md
    scripts/release/{common,train_baseline,train_comm_module_numeric,train_comm_module_text,train_commnav,eval_commnav}.sh
```
