# LearningMPC

> 来源归档

- **标题：** LearningMPC — Learning-based MPC for F1/10 Autonomous Racing
- **类型：** repo
- **来源：** mlab-upenn（宾大 F1/10 实验室）
- **链接：** https://github.com/mlab-upenn/LearningMPC
- **Stars：** ~170（2026-08-23）
- **入库日期：** 2026-08-23
- **一句话说明：** F1/10 尺度在线迭代 LMPC：用路径跟踪收集初始 safe set，逐圈 OSQP 求解 20Hz MPC 缩短圈速，终端约束落在历史样本凸包内保证递归可行。
- **代码：** https://github.com/mlab-upenn/LearningMPC（**已开源**；仓内无 SPDX license 文件，以 README 为准）
- **沉淀到 wiki：** 景观页 → [`wiki/overview/racing-drift-rl-open-source-landscape.md`](../../wiki/overview/racing-drift-rl-open-source-landscape.md)

---

## 依赖栈

1. [UPenn racecar_simulator](https://github.com/mlab-upenn/f110-fall2019-skeletons/tree/master/racecar_simulator)（ROS）
2. [OSQP](https://osqp.org/) + [OsqpEigen](https://robotology.github.io/osqp-eigen/)

---

## 典型入口

```bash
roslaunch racecar_simulator simulator.launch
roslaunch LearningMPC lmpc.launch
```

---

## 关联

- F1TENTH 生态：[`f1tenth_gym.md`](./f1tenth_gym.md)
- MPC 方法：[`wiki/methods/model-predictive-control.md`](../../wiki/methods/model-predictive-control.md)
