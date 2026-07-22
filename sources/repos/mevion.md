# haraduka/mevion

- **仓库：** <https://github.com/haraduka/mevion>
- **论文：** <https://arxiv.org/abs/2607.17970>
- **项目页：** <https://haraduka.github.io/mevion-hardware/>
- **定位：** MEVION 双臂遥操作采集、MuJoCo 仿真与实机控制的官方实现。
- **主要接口：** Python；MuJoCo；ROS / RViz；CAN；Scikit-Robot。
- **开放状态（2026-07-22）：** **已开源、可用但仍属研究原型**；复现需自备四臂硬件、驱动器与 24/48V 电气系统。
- **运行主链：** 操作者 leader arm → Python 控制层 → follower 目标 → MuJoCo 或 ROS/CAN → 轨迹记录 → ACT 训练/回放。
- **对应 wiki：** [`wiki/entities/paper-mevion.md`](../../wiki/entities/paper-mevion.md)

