# Anytime GTMP

> 来源归档

- **标题：** Anytime Global Tensor Motion Planning
- **类型：** repo
- **链接：** https://github.com/CoMMALab/anytime_gtmp
- **论文：** https://arxiv.org/abs/2608.25830
- **License：** MIT
- **入库日期：** 2026-08-28
- **一句话说明：** GTMP 的 anytime / 渐近最优实现：层状张量图 + VAMP 局部连接器；`examples/` 含 MotionBenchMaker 与 2D occupancy 评测入口。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-anytime-gtmp.md`](../../wiki/entities/paper-anytime-gtmp.md)

## 开源核查（2026-08-28）

**已开源** — 可辨识的规划 / 评测入口。

| 项 | 内容 |
|----|------|
| 安装 | `uv venv --python 3.11` → `uv pip install -e ./pyroffi ./vamp -r requirements.txt`；需 `git submodule update --init --recursive`，vamp 切到 `benchmark_aorrtc_backend` |
| 规划核 | `src/planners/` |
| 评测 | `examples/benchmark_mbm_anytime_*.py`、`benchmark_mbm_ao_time_budget_*.py`、occupancy/street 脚本 |
| 局部器 | VAMP 直线 / RRTC 连接器 |

## 对 wiki 的映射

- 论文摘录：[anytime_gtmp_arxiv_2608_25830.md](../papers/anytime_gtmp_arxiv_2608_25830.md)
- 实体：[paper-anytime-gtmp.md](../../wiki/entities/paper-anytime-gtmp.md)
