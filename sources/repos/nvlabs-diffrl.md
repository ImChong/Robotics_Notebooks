# NVlabs/DiffRL（SHAC 官方实现）

- **类型：** repo
- **链接：** <https://github.com/NVlabs/DiffRL>
- **关联论文：** [Accelerated Policy Learning with Parallel Differentiable Simulation](https://openreview.net/forum?id=ZSKRQMvttc)（ICLR 2022）
- **收录日期：** 2026-10-23
- **一句话说明：** 论文配套的 GPU 可微仿真、SHAC 训练与基线复现代码。

## README 运行入口

| 用途 | 入口 |
|------|------|
| 环境测试 | examples/test_env.py |
| SHAC 训练 | examples/train_shac.py 与 examples/cfg/shac |
| 多种子复现 | examples/train_script.py |
| 策略评测 | 将 checkpoint 传给训练脚本并启用 play |
| 仿真和基线 | dflex 与 externals/rl_games |

README 给出 Ant 与 SNUHumanoid 示例；结果可能因 CUDA、操作系统、GPU 与 Python 版本而略有变化。

## 对 Wiki 的映射

- [SHAC 论文实体](../../wiki/entities/paper-shac.md)
- [可微仿真](../../wiki/concepts/differentiable-simulation.md)
