# jev-drone（RomanSlack/jev-drone）

> 来源归档

- **标题：** Camera-only autonomous drone in MuJoCo with Jev in the loop
- **类型：** repo / 无人机仿真与自主导航演示
- **来源：** RomanSlack
- **链接：** <https://github.com/RomanSlack/jev-drone>
- **许可：** MIT
- **项目页：** <https://jev-drone.vercel.app/>
- **入库日期：** 2026-10-24
- **核查日期：** 2026-10-24
- **一句话说明：** 使用 MuJoCo Menagerie 的 Skydio X2 模型，在相机感知和常规几何飞控之上，以约 2.5 Hz 调用 Jev 做战术判断。
- **关联 wiki：** [Jev（TypeSafe）](../../wiki/entities/typesafe-jev.md)

## 仓库内容与边界

- **任务：** 四旋翼在 MuJoCo 中仅依据机载相机完成五站障碍路线；README 将其描述为非脚本化飞行。
- **分层：** 深度与分割提取场景约 15 Hz；50 Hz guidance / safety reflex 始终拥有安全控制；500 Hz 几何控制器负责姿态与推力；Jev 约 2.5 Hz 给出 advisory tactical judgment。Jev 本身不是视觉模型。
- **作者报告：** README 记录一次 65 秒运行中 80 次 Jev 调用，延迟中位数约 0.11 秒；同时明确该结果是单次运行。更早的简化场景三种子对照未显示 Jev 优于 baseline，作者也指出额外机动会增加目标可见性代价、方差较大。
- **复现边界：** 这是 MuJoCo 仿真项目，不是实机飞行结果；单次展示不构成统计性结论。

## 推荐入口

- 仓库说明：[README.md](https://github.com/RomanSlack/jev-drone/blob/main/README.md)
- 在线项目页与演示：<https://jev-drone.vercel.app/>
- 代码许可：[MIT License](https://github.com/RomanSlack/jev-drone/blob/main/LICENSE)
- [项目页归档](../sites/jev-drone.md)
