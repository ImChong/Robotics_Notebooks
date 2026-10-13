# sim2real Documentation

> 来源归档

- **类型：** project site
- **链接：** https://egalahad.github.io/sim2real/
- **访问日期：** 2026-10-13
- **一句话说明：** 官方使用文档：拆分策略推理/仿真/机器人 I/O 与 Pico/XR 遥操作环境，并覆盖 G1 PC/Orin 部署。

## 页面结构

文档包含 Getting Started、Root Project、Pico Teleoperation、Motion Recording、Run External Policies、Download Artifacts、Tracking Framework 与 Motion Tracking Leaderboard 等入口。

- Root project 负责 inference、tracking policy、MuJoCo simulation 与 robot I/O。
- Pico/XR 相关 retargeting、viewer 和动作记录放在独立环境中。
- 文档提供 PC 通过 Ethernet 控制 Unitree G1 与机载 Orin 两种部署布局，并说明部署前的网络、SDK、依赖及 I/O mode 设置。
- README 的 quick start 先启动 MuJoCo base simulation 和 tracking policy；迁移外部策略还要适配 observation/history/action contract，并进行 sim2sim 验证。
