# Humanoid Kick（Vision-Driven Reactive Soccer）

> 来源归档

- **标题：** Humanoid Kick
- **类型：** repo / Zenodo artifact
- **链接：** <https://zenodo.org/records/21620490>（`code.zip`）
- **DOI：** <https://doi.org/10.5281/zenodo.21620490>
- **项目页：** <https://humanoid-kick.github.io>
- **论文：** [arXiv:2511.03996](https://arxiv.org/abs/2511.03996) · [Science Robotics 11, eaed1152 (2026)](https://doi.org/10.1126/scirobotics.aed1152) — 归档见 [`sources/papers/humanoid_rl_stack_26_learning_vision_driven_reactive_soccer_skills_fo.md`](../papers/humanoid_rl_stack_26_learning_vision_driven_reactive_soccer_skills_fo.md)
- **入库日期：** 2026-08-22
- **一句话说明：** 清华 / 字节 Seed / 中国农大视觉驱动反应式足球技能的 **Zenodo 代码包**：Isaac Gym 训练 + Isaac/MuJoCo 推理回放，附预训练 `model.pth`；**无 GitHub 持续维护仓**。
- **开源状态：** **部分开源**（Zenodo `code.zip`，BSD-3-Clause）；含训练/推理脚本与 checkpoint，**不含**真机部署栈与完整训练日志。

---

## 可复现边界

| 可做 | 不可做 / 未包含 |
|------|----------------|
| `python train.py --task=T1 --headless` 在 Isaac Gym 复训 | 一键真机相机 + 里程计部署管线 |
| `python play.py` / `play_mujoco.py` 加载 `logs/model.pth` 评测 | GitHub Issues / 持续更新 |
| 阅读 `envs/T1.yaml` 与 AMP 判别器实现 | 论文全部室外 / RoboCup 真机配置 |

---

## 安装与入口（README 摘要）

- **环境：** Python 3.8 · PyTorch (CUDA 11.8) · **Isaac Gym Preview 4** · Pinocchio · `requirements.txt`
- **训练：** `python train.py --task=T1 --headless` → 日志与模型写入 `logs/<date-time>/`
- **推理：** `python play.py --task=T1 --checkpoint=-1`（Isaac Gym）或 `python play_mujoco.py`（MuJoCo 交叉仿真）
- **配置：** `envs/T1.yaml`；任务注册于 `envs/__init__.py` / `utils/task_registry.py`
- **代码谱系：** 部分源自 [legged_gym](https://github.com/leggedrobotics/legged_gym) 与 [rsl_rl](https://github.com/leggedrobotics/rsl_rl)

---

## 核心模块

| 路径 | 作用 |
|------|------|
| `train.py` / `play.py` | 训练与 Isaac Gym 回放入口 |
| `play_mujoco.py` | MuJoCo 交叉仿真评测 |
| `utils/runner.py` | PPO + AMP 判别器、ExperienceBuffer、checkpoint |
| `utils/model.py` | Actor-Critic（含 encoder-decoder 状态恢复） |
| `envs/t1.py` | T1 人形足球任务环境与虚拟感知 |
| `data/*.csv` | AMP 运动先验参考轨迹 |
| `logs/model.pth` | 预训练权重（约 15 MB） |

---

## 对 wiki 的映射

- 实体页：[paper-hrl-stack-26-learning_vision_driven_reactive_socc.md](../../wiki/entities/paper-hrl-stack-26-learning_vision_driven_reactive_socc.md)
- 项目页：[humanoid-kick-vision-driven-soccer.md](../sites/humanoid-kick-vision-driven-soccer.md)
