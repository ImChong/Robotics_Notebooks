# booster_mjlab 项目页（intelligentroboticslab.github.io）

- **标题：** booster_mjlab — mjlab integration for Booster K1
- **类型：** site / project-page
- **URL：** <https://intelligentroboticslab.github.io/booster_mjlab/>
- **代码：** <https://github.com/IntelligentRoboticsLab/booster_mjlab> — 归档见 [`sources/repos/booster_mjlab.md`](../repos/booster_mjlab.md)
- **数据集：** <https://huggingface.co/datasets/whirlwind-ams/lafan_locomotion_k1> — LAFAN1  locomotion 片段重定向至 parallel-ankle K1
- **入库日期：** 2026-09-21

## 一句话摘要

whIRLwind / Intelligent Robotics Lab 的 **Booster K1 × mjlab** 官方展示页：仿真 AMP 速度跟踪与特技 motion tracking、真机 30k 步 AMP 速度策略、浏览器内 MuJoCo WebAssembly 实时交互 demo，以及 LAFAN1 重定向 motion 数据集预览。

## 公开信息要点（截至入库日）

- **机构 / 团队：** [whIRLwind Amsterdam](https://whirlwind.team/)（机器人足球；阿姆斯特丹 Science Park；UvA 关联实验室 Intelligent Robotics Lab）。
- **仿真板块：** locomotion 来自 AMP velocity 任务；acrobatics 来自 motion tracking。
- **真机板块：** Booster K1 上 AMP velocity-tracking 策略（约 30k training steps 后录制）。
- **浏览器 demo：** 导出策略以 **50 Hz** 在 MuJoCo WebAssembly 中运行；支持摇杆 / 键盘 WASD+QE / 手柄；策略包约 14 MB。
- **Motion references：** LAFAN1 重定向至 **parallel-ankle K1**；驱动 AMP motion prior；内置 motion viewer 可浏览 / 回放 / 编辑 clip。
- **招募：** 页脚链向 whirlwind.team，面向学生与合作方。

## 开源核查（步骤 2.5）

| 维度 | 结论 |
|------|------|
| **训练 / 仿真代码** | **已开源** — GitHub `IntelligentRoboticsLab/booster_mjlab`，`uv run train` / `play` / `list_envs` |
| **参考 motion 数据** | **已发布** — Hugging Face `whirlwind-ams/lafan_locomotion_k1` |
| **浏览器 demo 权重** | **可在线体验** — 项目页加载导出网络（非独立 HF 模型页） |
| **真机部署示例** | **待发布** — README 写 "Deployment … coming soon"（截至 2026-09-21） |

## 为何值得保留

- **非 README 证据：** 真机行走、浏览器 WASM 交互、motion 可视化比命令行更能支撑 Sim2Real / 部署选型判断。
- **与 Booster 官方栈对照：** 加速进化 [`booster_gym`](../../sources/repos/booster_gym.md) 走 Isaac Gym；本页代表 **mjlab + AMP** 的 K1 社区/学术路线。
- **足球主线锚点：** whIRLwind RoboCup Humanoid Soccer 2026 第 4 名，与本库 [humanoid-soccer](../../wiki/tasks/humanoid-soccer.md) 任务页互证。

## 关联资料

- 代码归档：[`sources/repos/booster_mjlab.md`](../repos/booster_mjlab.md)
- 底层框架：[`sources/repos/mjlab.md`](../repos/mjlab.md)
- AMP 对照：[`sources/repos/amp_mjlab.md`](../repos/amp_mjlab.md)（Unitree G1 社区实现）
