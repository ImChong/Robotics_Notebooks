# htwk-gym

> 来源归档

- **标题：** htwk-gym
- **类型：** repo
- **来源：** NaoHTWK（RoboCup 参赛队 HTWK Leipzig）
- **链接：** https://github.com/NaoHTWK/htwk-gym
- **入库日期：** 2026-04-27
- **一句话说明：** 针对人形机器人（Booster T1/K1）足球任务的强化学习框架，支持参数化步行与踢球动作训练。
- **沉淀到 wiki：** [htwk-gym](../../wiki/methods/htwk-gym.md), [humanoid-soccer](../../wiki/tasks/humanoid-soccer.md)

---

## 核心定位

htwk-gym 是一个专注于人形机器人足球竞技（Humanoid Soccer）的强化学习训练框架，由 RoboCup 强队 HTWK Leipzig 开源。它提供了一个完整的从训练到真机部署的流水线。

---

## 关键技术栈

- **仿真引擎：** NVIDIA **Isaac Gym** (主要训练), **MuJoCo** (跨仿真验证)
- **深度学习：** **PyTorch** (训练与 JIT 导出), **TensorFlow Lite** (嵌入式端量化部署)
- **可视化/调试：** **Streamlit** (实时观测编辑器，用于在线调整步态参数)
- **支持硬件：** **Booster T1**, **Booster K1**

---

## 核心任务与功能

1. **参数化步行 (Parameterized Walking)：**
   - 支持频率 (Frequency)、足部偏航 (Foot Yaw)、身体俯仰/翻滚 (Body Pitch/Roll) 的实时参数化控制。
   - 目标是实现稳定且快速的足球场移动。

2. **足球专项动作：**
   - `T1/Kicking`: 包含球交互、目标速度奖励等逻辑，支持踢球动作的闭环训练。

3. **Sim-to-Real (从仿真到现实)：**
   - **领域随机化 (Domain Randomization)：** 对地形摩擦力、重力、机器人质量分布等进行大规模随机化。
   - **多格式导出：** 支持 PyTorch JIT、ONNX 和 TFLite 格式，方便部署到不同的嵌入式算力平台。

---

## 仓库结构

- `envs/`: 环境定义，包含 `T1_walking`, `T1_kicking` 等。
- `learning/`: RL 算法逻辑（基于 PPO）。
- `scripts/`: 训练、评估与导出脚本。
- `tools/`: 包含 `obs_editor.py` (Streamlit 界面) 等调试工具。

---

## 与本项目其他资料的关系

| 资料 | 关系 |
|------|------|
| [booster-robocup-demo.md](../../wiki/entities/booster-robocup-demo.md) | htwk-gym 训练出的策略可集成到该决策框架中 |
| [isaac_gym_isaac_lab.md](../isaac_gym_isaac_lab.md) | htwk-gym 使用 Isaac Gym 作为底层引擎 |
| [robot_lab.md](robot_lab.md) | robot_lab 也支持 Booster T1，但 htwk-gym 更偏重足球垂直任务 |
