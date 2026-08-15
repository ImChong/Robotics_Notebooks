# ricardoGrando/go2_rescue_eval

- **标题：** go2_rescue_eval（四足搜救 frontier 评测包）
- **类型：** repo
- **URL：** <https://github.com/ricardoGrando/go2_rescue_eval>
- **配套论文：** [arXiv:2608.02571](https://arxiv.org/abs/2608.02571) — [`sources/papers/situation_aware_frontier_arxiv_2608_02571.md`](../papers/situation_aware_frontier_arxiv_2608_02571.md)
- **入库日期：** 2026-08-15

## 一句话说明

ROS 2 Jazzy 外包：不改 `unitree_go2_ros2`，在单机 Go2 仿真上跑 nearest / info_gain / risk_aware / full_sa，并做多种子批量评测。

## 仓库状态（2026-08-15 核查）

| 项 | 内容 |
|----|------|
| 入口 | `ros2 launch go2_rescue_eval trial.launch.py`；`ros2 run go2_rescue_eval run_batch_eval` |
| 世界 | `worlds/s1_structured_indoor.sdf`、`s2_cluttered_disaster.sdf`、`s3_collapsed_response.sdf` |
| 控制器 | `go2_rescue_eval/mission_controller.py` |
| 依赖 | 需本机已有 `unitree_go2_sim` / `unitree_go2_description` / `champ_base` |
| 受害者 | 红色静态圆柱体视觉代理 |

## 与 wiki 的关系

- 实体页：[paper-situation-aware-frontier-quadruped-sar](../../wiki/entities/paper-situation-aware-frontier-quadruped-sar.md) — 含源码运行时序图。
