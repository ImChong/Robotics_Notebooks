# YilunKuang/lpworldmodel

> 来源归档

- **标题：** LpWorldModel 官方实现
- **类型：** repo
- **代码：** <https://github.com/YilunKuang/lpworldmodel>
- **License：** MIT
- **论文：** <https://arxiv.org/abs/2608.22764>
- **入库日期：** 2026-09-05
- **一句话说明：** LpWM 训练/规划仓：Hydra `train.py`/`plan.py` + `lpwm_swm/`（stable-worldmodel 另环境）。

## 开源核查（2026-09-05）

| 项 | 状态 |
|----|------|
| 代码 | **已开源** · MIT · 28★ |
| 权重 | 未随仓发布；需自训 |
| 数据 | DINO-WM OSF：`pusht_noise` / `wall_single` |

## 入口速查

| 路径 | 作用 |
|------|------|
| `conda env create -f environment.yaml` | §3 Wall/PushT 环境 |
| `train.py` / `plan.py` | Hydra 训练 / CEM 规划 |
| `models/infojepa_modules.py` | RDMReg + Link + 预测器梯子 |
| `lpwm_swm/` | Piecewise / OGBench-Cube |
