# RoboGauge（MoE 四足 Sim-to-Real 可预测性评估）

- **标题：** Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion
- **类型：** site
- **项目页：** <https://robogauge.github.io/complete/>
- **论文：** RSS 2026；arXiv <https://arxiv.org/abs/2602.00678>
- **机构：** 西安交通大学（XJTU）
- **收录日期：** 2026-09-15

## 开源状态（2026-09-15 项目页核查）

| 组件 | URL | 状态 |
|------|-----|------|
| 训练 `go2_rl_gym` | <https://github.com/wty-yy/go2_rl_gym> | **已开源** |
| 评估 `RoboGauge` | <https://github.com/wty-yy/RoboGauge> | **已开源** |
| 部署 `unitree_cpp_deploy` | <https://github.com/wty-yy/unitree_cpp_deploy> | **已开源** |

项目页声明：文内 demo 可用所提供模型在 `unitree_cpp_deploy` 复现。

## 一句话说明

RoboGauge 是 **Isaac Gym → MuJoCo** 跨引擎分层压力测试套件，用 8 维本体感受指标（含 ZMP / friction margin）在真机前筛选 MoE locomotion checkpoint。

## 对 wiki 的映射

- [paper-robogauge-moe-quadruped-locomotion](../../wiki/entities/paper-robogauge-moe-quadruped-locomotion.md)
