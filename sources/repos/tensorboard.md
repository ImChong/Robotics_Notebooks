# TensorBoard

> 来源归档

- **标题：** TensorBoard — TensorFlow's Visualization Toolkit
- **类型：** repo / 桌面 Web 工具
- **来源：** Google / TensorFlow 社区
- **链接：** https://github.com/tensorflow/tensorboard
- **官方文档：** https://www.tensorflow.org/tensorboard
- **Stars：** ~7.2k（2026-06）
- **入库日期：** 2026-06-25
- **许可证：** Apache License 2.0
- **一句话说明：** 离线优先的 ML 实验 **可视化套件**：从本地 event 日志读取标量、直方图、计算图、embedding、图像/音频与 profiler；机器人 RL 栈（RSL-RL、mjlab、MimicKit、robot_lab）默认用其监控 `mean_reward`、分项 reward 与 PPO/AMP loss。

## 为什么值得保留

- **RL 训练默认仪表盘**：本库 [AMP_mjlab](../../wiki/entities/amp-mjlab.md)、[BeyondMimic](../../wiki/methods/beyondmimic.md) 等页均以 TensorBoard tag 命名空间作为「训练是否收敛」的判据。
- **完全离线**：README 强调可在本地、防火墙内或数据中心运行，**无需互联网**（与 W&B 云托管形成对照）。
- **框架无关**：虽源自 TensorFlow，现已被 **PyTorch**（`torch.utils.tensorboard.SummaryWriter`）、**rsl_rl** 等广泛采用，仅需写入 `events.out.tfevents.*` 文件。

## 核心能力（官网 + README 摘要）

| 能力 | 说明 |
|------|------|
| **标量曲线** | loss、accuracy、`Train/mean_reward` 等随 step/time 变化 |
| **计算图** | 可视化 ops 与层连接（TF 图或 trace） |
| **直方图** | 权重、偏置或其它 tensor 分布随训练演化 |
| **Embedding Projector** | 高维向量降维可视化 |
| **多媒体** | 图像、文本、音频样本 |
| **Profiler** | TensorFlow 程序性能剖析 |
| **HPARAMS** | 超参对比视图 |

## 使用方式（README）

```bash
# 训练侧写入 summary（PyTorch 示例）
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="/path/to/logs")

# 启动本地 UI（默认 http://localhost:6006）
tensorboard --logdir path/to/logs
```

- 浏览器推荐 **Chrome / Firefox**。
- 历史曾提供 [TensorBoard.dev](https://tensorboard.dev) 云端托管（公开实验链接）；**本地 `--logdir` 仍是机器人研发最常见路径**。

## 机器人 RL 中的典型 tag（本库实践）

| 前缀 | 来源示例 | 用途 |
|------|----------|------|
| `Train/` | rsl_rl `amp_on_policy_runner` | `mean_reward`、`mean_episode_length` |
| `Episode_Reward/` | mjlab `RewardManager` | 各 reward term 分项曲线 |
| `Loss/` | PPO / AMP `loss_dict` | value、surrogate、entropy、`amp_*` |

详见 [AMP_mjlab 实体页](../../wiki/entities/amp-mjlab.md) §训练监控。

## 对 wiki 的映射

- 升格实体：[tensorboard.md](../../wiki/entities/tensorboard.md)
- 选型对比：[wandb-vs-tensorboard.md](../../wiki/comparisons/wandb-vs-tensorboard.md)
- 调试 playbook：[robot-policy-debug-playbook.md](../../wiki/queries/robot-policy-debug-playbook.md)

## 与本库其他条目的关系

| 资料 | 关系 |
|------|------|
| [amp_mjlab.md](amp_mjlab.md) | README 以 TensorBoard 曲线作为训练成功基准 |
| [robot_lab.md](robot_lab.md) | checkpoint + TensorBoard 工作流 |
| [mimickit.md](mimickit.md) | `logger_type="tb"` 默认路径 |
| [plotjuggler.md](plotjuggler.md) | 训练期标量 vs 真机 rosbag 时序：TB 看训练，PJ 看部署 |
