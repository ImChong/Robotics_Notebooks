# Situation Aware Frontier Prioritization for Quadruped Search and Rescue（arXiv:2608.02571）

> 来源归档（ingest）

- **标题：** Situation Aware Frontier Prioritization for Quadruped Search and Rescue
- **缩写 / 框架：** **SA Frontier** / `full_sa`；评测包 **go2_rescue_eval**
- **类型：** paper / quadruped / search-and-rescue / exploration
- **arXiv：** <https://arxiv.org/abs/2608.02571>
- **代码：** <https://github.com/ricardoGrando/go2_rescue_eval>（归档见 [`sources/repos/go2-rescue-eval.md`](../repos/go2-rescue-eval.md)）
- **视频：** <https://www.youtube.com/watch?v=BbtPfF-NLac>
- **作者：** Kevin Farias、Santiago Martin、Bárbara Flores、Vinicio Melgar、Igor Nunes、Hiago Sodre、Pablo Moraes、Ricardo B. Grando∗
- **机构：** 乌拉圭技术大学（UTEC）Robotics and AI Lab
- **入库日期：** 2026-08-15
- **一句话说明：** 在经典 frontier 探索上加入信息增益、观测赤字、救援相关性、地形惩罚与行程代价，用 Unitree Go2 在 Gazebo 室内搜救场景评测；复杂 clutter 下完成率与受害者回收最高。

## 开源状态（步骤 2.5）

- **仓库核查（2026-08-15）：** [ricardoGrando/go2_rescue_eval](https://github.com/ricardoGrando/go2_rescue_eval) 为 ROS 2 Jazzy 外包，含 `trial.launch.py`、`run_batch_eval`、三套 SDF 世界与 `mission_controller.py`。**已开源、可运行**（依赖本机已有 `unitree_go2_ros2` 仿真栈）。受害者用红色静态圆柱体作视觉代理。
- **结论：** **已开源。** 实体页须含源码运行时序图。

## 摘录 1：打分

\[
J(f)=w_I I(f)+w_O O(f)+w_R R(f)-w_T T(f)-w_D D(f)
\]

救援项 \(R(f)\) 对暂定受害者线索做高斯空间加权。对照：nearest frontier / information gain / risk-aware。局部控制器与恢复行为四方法共用。

## 摘录 2：实验（Table 1，各 20 run）

| 场景 | 方法 | 完成率 | 受害者回收 | 任务时间 (s) |
|------|------|--------|------------|--------------|
| S1 简单（1 人） | Info Gain | **19/20** | **0.95** | 424.2 |
| S1 | SA Frontier | 15/20 | 0.75 | 548.7 |
| S2 复杂（2 人） | **SA Frontier** | **20/20** | **2.00** | **373.5** |
| S2 | Risk-aware | 19/20 | 1.95 | 391.8 |
| S2 | Nearest | 18/20 | 1.80 | 414.9 |
| S2 | Info Gain | 14/20 | 1.55 | 612.9 |

读法：简单场景不必上局势感知；frontier 歧义变大时救援项才拉开差距。仓库另有 s3 世界，论文主表写 S1/S2。

**对 wiki 的映射：** [`wiki/entities/paper-situation-aware-frontier-quadruped-sar.md`](../../wiki/entities/paper-situation-aware-frontier-quadruped-sar.md)；交叉 [autonomy_stack_go2](../../wiki/entities/autonomy-stack-go2.md)、[Unitree](../../wiki/entities/unitree.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（已开源）
