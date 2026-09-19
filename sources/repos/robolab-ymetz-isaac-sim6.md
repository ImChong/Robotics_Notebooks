# RoboLab — ymetz Isaac Sim 6 port

- **URL：** <https://github.com/ymetz/RoboLab>
- **上游：** <https://github.com/NVLabs/RoboLab>
- **文档：** <https://github.com/ymetz/RoboLab/blob/main/docs/isaac_sim_6.md>
- **维护：** Manda Robotics 评测使用（博客致谢 ymetz fork）
- **入库日期：** 2026-09-19

## 一句话说明

NVLabs/RoboLab 的 **Isaac Sim 6 + Isaac Lab 3** 移植 fork：保留观测/动作/录制约定，供 [Manda 五策略横评](https://mandarobotics.com/blog/state-of-robot-policies/index.html) 与官方 leaderboard 做 aggregate sanity check。

## 与上游差异（文内）

- 上游文档：Isaac Sim 5.0/5.1；本 fork：**Isaac Sim 6**
- 回归：初始状态、动作、轨迹、结果 matched-run；1,200 non-camera physical states 跨五策略一致
- **不保证** 跨 Sim 版本逐 episode 轨迹等价（渲染/接触动力学差异）

## 交叉链接

- [RoboLab 官方仓库](./robolab.md)
- [Manda 评测 blog](../blogs/mandarobotics_state_of_robot_policies_2026-09-17.md)
- [wiki/entities/robolab.md](../../wiki/entities/robolab.md)
- [wiki/entities/manda-robotics-open-policy-evaluation.md](../../wiki/entities/manda-robotics-open-policy-evaluation.md)
