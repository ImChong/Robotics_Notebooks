# KaRMA（mfpeticco/karma-hand-metric）

> 来源归档

- **标题：** KaRMA: Kinematic Rolling Manipulation Ability
- **类型：** repo + metric CLI + visualization
- **组织：** MIT Improbable AI Lab（Martin Peticco, Pulkit Agrawal）
- **代码：** <https://github.com/mfpeticco/karma-hand-metric>
- **项目页：** <https://martinpeticco.com/karma/>
- **论文：** [arXiv:2605.15548](https://arxiv.org/abs/2605.15548)（IROS 2026）
- **License：** 见仓库（论文实现）
- **入库日期：** 2026-09-17
- **一句话说明：** CPU-only URDF 灵巧手运动学指标：`run_metric.py` 输出 KaRMA-T/R/S；bundled 16 手 + `run_viser_app.py` 体素可视化；bit-reproducible。

## 仓库入口

| 资源 | 路径 / 命令 | 说明 |
|------|-------------|------|
| 主入口 | `run_metric.py --config robots/robot_*.yaml` | 单 hand 评分；写 `workspace/current.yaml` + `current.pkl` |
| 批量 | `run_all_hands.py` | 16 手 ~18 min（32-thread desktop） |
| 可视化 | `run_viser_app.py` | localhost:8080 voxel 云 |
| 新 hand | `robots/robot_template.yaml` + URDF in `robots/urdfs/` | LLM 辅助见 `robots/prompt.txt` |
| 论文结果 | `results/16_hand_batch/summary.yaml` | Table I |
| 基线 | `baselines/run_baselines.py` | opposability / Yoshikawa / GCI 相关 |

## 典型复现

```bash
git clone https://github.com/mfpeticco/karma-hand-metric
cd karma-hand-metric
conda env create -f environment.yml && conda activate karma-hand-metric
python run_metric.py --config robots/robot_leap.yaml
python run_viser_app.py --result workspace/current.pkl
```

## 开源状态（2026-09-17）

**已开源、可运行。** 依赖 Pinocchio（PyPI 包名 `pin`）、hpp-fcl；无 GPU/torch。

## 对 wiki 的映射

- 实体页：[KaRMA](../../wiki/entities/paper-karma-hand-metric.md)
- 站点：[karma-hand-metric-martinpeticco.md](../sites/karma-hand-metric-martinpeticco.md)
