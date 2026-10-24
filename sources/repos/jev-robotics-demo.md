# Jev vs LLM: Robot Arm Control（FazalAAli/jev-robotics-demo）

> 来源归档

- **标题：** Jev vs LLM: robot arm control
- **类型：** repo / MuJoCo 机械臂操作对比演示
- **来源：** Fazal Ali
- **链接：** <https://github.com/FazalAAli/jev-robotics-demo>
- **许可：** MIT
- **入库日期：** 2026-10-24
- **核查日期：** 2026-10-24
- **一句话说明：** 在 MuJoCo 中对比 Jev 与 Claude Opus 5 操作 Franka + Allegro 手堆叠方块的两种控制编排。
- **关联 wiki：** [Jev（TypeSafe）](../../wiki/entities/typesafe-jev.md)

## 仓库内容与边界

- **任务与资产：** Franka 机械臂配四指 Allegro 手，在 MuJoCo 中把蓝色方块堆到红色方块上。
- **Jev 路径：** jev_agent.py 由代码提出小幅动作，并在物理状态副本中前向预演；Jev 选择动作、抓取/释放时机和任务完成状态。
- **LLM 路径：** agent.py 通过 move_hand_to(x, y, z)、set_hand(closed)、get_state() 等工具逐步推理并操控。
- **README 记录：** 各一条演示运行中两者都堆叠成功；Jev 记录 19.1 秒、约 $0.0006，Claude Opus 5 记录 158.8 秒、约 $0.75。样本各一条且实验条件有限，不能据此推断一般性能优势。
- **复现边界：** 这是 MuJoCo 仿真，不是真实机械臂部署或标准化基准。

## 推荐入口

- 仓库说明：[README.md](https://github.com/FazalAAli/jev-robotics-demo/blob/main/README.md)
- 代码许可：[MIT License](https://github.com/FazalAAli/jev-robotics-demo/blob/main/LICENSE)
