# f1tenth_gym

> 来源归档

- **标题：** F1TENTH Gym
- **类型：** repo
- **来源：** F1TENTH 社区 / UPenn 起源
- **链接：** https://github.com/f1tenth/f1tenth_gym
- **文档：** https://f1tenth-gym.readthedocs.io
- **Stars：** ~246（2026-08-23）
- **许可证：** MIT
- **入库日期：** 2026-08-23
- **一句话说明：** F1TENTH 1/10 竞速官方 Python Gym 环境：轻量单车动力学仿真、waypoint follow 示例与 Docker GUI；大量 RL/MPC 赛车工作的默认仿真后端。
- **代码：** https://github.com/f1tenth/f1tenth_gym（**已开源**）
- **沉淀到 wiki：** 是 → [`wiki/entities/f1tenth-gym.md`](../../wiki/entities/f1tenth-gym.md)

---

## 典型入口

```bash
virtualenv gym_env && source gym_env/bin/activate
git clone https://github.com/f1tenth/f1tenth_gym.git && cd f1tenth_gym
pip install -e .
cd examples && python3 waypoint_follow.py
```

---

## 下游生态

- [Gym-Khana](./gym_khana.md) — SB3 漂移/竞速封装
- [LearningMPC](./learning_mpc.md) — ROS 侧 F1/10 仿真（不同后端）
- [autonomous_f1tenth.md](./autonomous_f1tenth.md) — Gazebo + ROS 2 RL

---

## 已知局限（README）

- Windows 需 Python 3.8（截至 README 快照）
- macOS Big Sur+ 渲染可能缺 OpenGL framework
