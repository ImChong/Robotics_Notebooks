# humanoid-kick.github.io（Vision-Driven Reactive Soccer）

> 来源归档（ingest）

- **标题：** Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots
- **类型：** site / project-page
- **官方入口：** <https://humanoid-kick.github.io>
- **入库日期：** 2026-07-28
- **再核日期：** 2026-08-22
- **一句话说明：** 清华 / 字节跳动 Seed / 中国农大视觉驱动反应式足球技能项目页：AMP + 虚拟感知 + encoder-decoder；室外与 RoboCup 演示；**Science Robotics 2026** 正式发表。
- **开源状态（2026-08-22 项目页核查）：** **部分开源** — 页头 **Code** 链至 Zenodo [21620490](https://zenodo.org/records/21620490)（`code.zip`：Isaac Gym 训练 + MuJoCo/Isaac 推理 + `model.pth`）；**无 GitHub 训练仓**。真机部署栈未随包发布。

## 页面公开信息

| 资源 | URL |
|------|-----|
| 项目首页 | <https://humanoid-kick.github.io> |
| arXiv | <https://arxiv.org/abs/2511.03996> |
| PDF | <https://arxiv.org/pdf/2511.03996> |
| 正式发表 | [Science Robotics 11, eaed1152 (2026)](https://doi.org/10.1126/scirobotics.aed1152) |
| 代码（Zenodo） | <https://zenodo.org/records/21620490> |

## 方法摘要（项目页）

- **训练：** 部分观测 + encoder-decoder 恢复状态；PPO + AMP 判别器；多 critic。
- **部署：** 机载相机球检测进策略；里程计估门位。
- **摘要量化（arXiv v2 / 项目页）：** 相对规则基线，球位估计误差 **−46%**、踢球准备时间 **−64%**；前场约 **90%** 踢球成功率。

## 源码开放核查（步骤 2.5）

| 类别 | 状态 | 说明 |
|------|------|------|
| 训练 / 仿真推理 | **部分开源** | Zenodo `code.zip`：`train.py`、`play.py`、`play_mujoco.py`、`envs/T1.yaml`、`logs/model.pth` |
| GitHub 持续维护仓 | **未列** | 页头 Code 图标指向 Zenodo，非 GitHub |
| 真机部署 | **未开源** | 项目页仅描述 onboard camera + odometer，无对应脚本 |

## 对 wiki 的映射

- [`wiki/entities/paper-hrl-stack-26-learning_vision_driven_reactive_socc.md`](../../wiki/entities/paper-hrl-stack-26-learning_vision_driven_reactive_socc.md)
- [`sources/papers/humanoid_rl_stack_26_learning_vision_driven_reactive_soccer_skills_fo.md`](../papers/humanoid_rl_stack_26_learning_vision_driven_reactive_soccer_skills_fo.md)
- [`sources/repos/humanoid-kick-vision-driven-soccer.md`](../repos/humanoid-kick-vision-driven-soccer.md)
