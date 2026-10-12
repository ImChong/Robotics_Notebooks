# easagru（e-puck 通信价值演化代码）

> 来源归档

- **标题：** ferdiex/easagru
- **类型：** repo
- **链接：** <https://github.com/ferdiex/easagru>
- **代码：** <https://github.com/ferdiex/easagru>
- **Zenodo：** <https://doi.org/10.5281/zenodo.23051976>（v1.0.0，2026-09-30）
- **论文：** <https://arxiv.org/abs/2610.13118>
- **许可：** MIT
- **核查日期：** 2026-10-12（clone 后浏览；最近提交 2026-10-11）
- **沉淀到 wiki：** [`wiki/entities/paper-epuck-value-of-information-communication.md`](../../wiki/entities/paper-epuck-value-of-information-communication.md)

## 仓库入口

| 文件 / 目录 | 作用 |
|-------------|------|
| `easagru_world.py` / `easagru_sensors.py` / `easagru_signal.py` / `easagru_env.py` | PyBullet 机器人、传感器、1 比特信号、双机觅食任务 |
| `easagru_bg.py` | EASA 基底神经节核心（对照 `easa_ref/bg_core.py`） |
| `easagru_brain.py` | Res-GRU、先天通道、EASA 变体与条件 |
| `easagru_evolve.py` | 遗传算法（默认 pop 48、150 代） |
| `easagru_evaluate.py` / `easagru_diagnose.py` / `easagru_robustness.py` / `easagru_signal_value.py` | 评测与信号价值分析 |
| `easagru_stats.py` | 秒级重建论文 Table 3–5 与统计检验 |
| `easagru_standalone.py` | PyBullet 窗口回放冠军个体 |
| `scripts/` | `run_tests.sh`、`run_evolution.sh`、`evaluate_run.sh`、`diagnose_run.sh`、`robustness.sh`、`signal_value_big.sh` |
| `runs/` | 12 个运行的配置、演化日志、冠军基因组、100 episode 评测 |
| `results_tables/` | 论文表格 CSV / MD 与 `statistical_tests.txt` |

依赖：numpy、scipy、pybullet；Python 3.9+。
