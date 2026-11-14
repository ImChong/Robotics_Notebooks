# verl-vla 官方源码导航（2026-11-14）

- **资料类型：** repo / 云边协同 VLA 后训练框架
- **核对范围：** 2026-11-14 阅读 verl-project/verl-vla 官方 README 全文（框架分层、TrainCluster、遥操作与介入、支持矩阵、维护配方、Quick Start、FAQ/许可），并核对 pyproject.toml 中的 `vvla-teleop`、`vvla-train-sft`、`vvla-train-sac` CLI 入口。
- **通读：** 已完成
- **名称：** verl-vla: a unified VLA post-training framework
- **代码：** https://github.com/verl-project/verl-vla
- **文档：** https://verl-vla.readthedocs.io/en/latest/
- **代码维护组织：** verl-project（不要仅依据转载将组织归属写成单一高校）
- **核对范围：** 官方 README 的 Framework、Distributed execution、Human intervention、Training/Evaluation Matrix、Quick Start。
- **通读：** 未完成（已核对 README 与操作入口，未逐一读完训练/RL 算法源码）。

## 栈分层与操作入口

官方明确区分三层：`workflow` 控制训练/采集/评估阶段；`trainer` 实现训练算法；`TrainCluster` 协调 actor、rollout、环境、录制和权重同步 worker。人类可以接入示范采集与失败干预，而不是只有离线 SFT。

**开源状态：** 云边训练调度、SFT、RL、LIBERO/Isaac Lab/Piper 等示例已公开。官方 README 的最小操作验证入口（先按文档安装环境）为：

```bash
source .venv/bin/activate
vvla-teleop cluster.env.env_worker.simulator.libero.task_suite_name=libero_spatial cluster.env.env_worker.simulator.libero.task_ids='[0]' cluster.env.env_worker.teleop.devices='[keyboard]'
```

随后可访问 `http://localhost:18000` 的 teleoperation 界面。此入口是**人介入采集的快速演示**，不能据此声称已复现分布式 RL 全链路；未进行实际部署测试。

## 对 wiki 的映射

- [独立项目详情：verl-vla](../../wiki/entities/verl-vla.md)
- [2026 VLA 开源生态对比](../../wiki/comparisons/vla-open-source-ecosystem-2026-three-shifts.md)
- [VLA 方法](../../wiki/methods/vla.md)

## 参考来源

- https://github.com/verl-project/verl-vla
- https://verl-vla.readthedocs.io/en/latest/
