# foothold-prediction-model（Watch Your Step 官方代码）

> 来源归档

- **标题：** Watch Your Step: Anticipated Perception for Humanoid Locomotion
- **类型：** repo
- **来源：** ETH Zürich, Robotic Systems Lab
- **链接：** <https://github.com/leggedrobotics/foothold-prediction-model>
- **项目页：** <https://leggedrobotics.github.io/watch-your-step/>
- **论文：** arXiv:[2610.29612](https://arxiv.org/abs/2610.29612) — 归档见 [论文摘录](../papers/watch_your_step_arxiv_2610_29612.md)
- **入库日期：** 2026-10-23
- **一句话说明：** Unitree G1（29 DoF）在 Isaac Lab 中进行感知式行走的训练与评估代码；包含足点预测模型、策略训练和预训练 demo。
- **沉淀到 wiki：** [Watch Your Step](../../wiki/entities/paper-watch-your-step.md)

## 核心定位

仓库提供论文方法的训练和评估实现：Foothold Prediction Model（FPM）根据本体感觉与速度指令预测即将到来的两个足点，策略随后只读取这些足点附近的高程图 patch。README 指出 FPM 与策略联合训练，训练过程中每 500 次迭代使用当前策略 rollout 重新训练 FPM。

## 已公开内容（README / 仓库结构）

| 组件 | 说明 |
|------|------|
| `fpm/foot_pred/` | 足点模型、足点标签、重拟合及训练调度 |
| `fpm/mdp/` | 足点区域高度扫描、奖励、事件、命令与对称增强 |
| `fpm/terrains/` | 训练、行走验证与走廊验证地形 |
| `fpm/env_cfg.py` | 训练、播放及验证环境配置 |
| `fpm/agents/` | PPO 配置 |
| `fpm/assets/` | Unitree G1（29 DoF）USD |
| `scripts/train.py` | 训练入口 |
| `scripts/play.py` | 策略回放入口 |
| `scripts/validate_*.py` | locomotion、走廊和足点预测验证入口 |
| `demo/` | 预训练策略与 FPM |

## 最小运行信息（README）

依赖 Isaac Sim 5.1、Isaac Lab 2.3.0 和 `rsl-rl-lib==3.3.0)。在 Isaac Lab Python 环境中安装仓库后，可运行：

```bash
pip install -e .
python scripts/train.py --headless --seed 1
```

训练初始 500 次迭代将足点固定在机器人基座下方；此后每 500 次迭代，FPM 使用当前策略的 5×10^5 条 transition 从头训练。README 还给出播放、验证与 checkpoint 恢复说明。

## 开源边界

- **已开源：** 仓库含训练、播放、验证代码、环境配置和 demo 权重；可复现仿真侧主要流程。
- **真机复现实验：** 论文摘要报告 Unitree G1 零样本迁移；仓库 README 描述训练与评估实现。具体真机运行依赖及实验流程以项目页和论文为准。
- **项目页：** 仓库 README 链接至项目页，项目页与论文及本归档相互关联。

## 对 wiki 的映射

- 实体页：[Watch Your Step](../../wiki/entities/paper-watch-your-step.md)
- 论文摘录：[arXiv:2610.29612](../papers/watch_your_step_arxiv_2610_29612.md)
