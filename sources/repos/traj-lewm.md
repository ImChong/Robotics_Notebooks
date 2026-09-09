# XiaodiHuang-code/Traj_LeWM

> 来源归档

- **标题：** Traj-LeWM 官方实现（源码发布）
- **类型：** repo
- **代码：** <https://github.com/XiaodiHuang-code/Traj_LeWM>
- **License：** MIT
- **论文：** <https://arxiv.org/abs/2608.14125>
- **入库日期：** 2026-09-09
- **一句话说明：** 在 LeWM 式 JEPA 上增加 LTC 模块、轨迹偏好训练与联合 endpoint+LTC 的 CEM 规划；四仿真环境配置 + 分析脚本。

## 开源核查（2026-09-09）

| 项 | 状态 |
|----|------|
| 代码 | **已开源** · MIT · source-only |
| Checkpoints | **未发布**（README「Code only」） |
| 依赖 | Python 3.10 · PyTorch CUDA |

## 入口速查

| 命令 | 作用 |
|------|------|
| `python train.py` | 训练（含 LTC 偏好与 LeWM 预测损失） |
| `python eval.py` | 闭环 CEM 规划评测 |
| `config/` | Push-T / Cube / Reacher / Two-Room 配置 |
