# RoboClaw（代码仓库）

> 来源归档

- **标题：** RoboClaw — Embodied AI Assistant
- **类型：** repo
- **组织：** MINT-SJTU（上海交通大学 MINT 实验室）
- **机构：** 上海交通大学（SJTU）
- **链接：** <https://github.com/MINT-SJTU/RoboClaw>
- **入库日期：** 2026-08-22
- **一句话说明：** 开源具身智能助手：面向任意本体/环境/任务的助手 + 具身层（本体建模、熟悉校准、能力抽象、训练辅助）+ ROS2 执行层 + 仿真/真机载体层；早期阶段，社区共建。
- **沉淀到 wiki：** [`wiki/entities/roboclaw.md`](../../wiki/entities/roboclaw.md)

## 开源状态

**已开源**（early stage）。README 标明 **status: early_stage**；提供非 Docker / Docker 安装文档；Discord 社区；继承部分 [nanobot](https://github.com/HKUDS/nanobot) 与 [OpenClaw](https://github.com/openclaw/openclaw) 思路。

**近期里程碑（README News）：** 2026-04-11 Web dashboard；2026-03-24 对话式臂 setup/标定/遥操/采数/训练/推理；2026-03-17 框架骨架与 domain contracts。

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [roboclaw](../../wiki/entities/roboclaw.md) | 实体页 |
| [rosclaw](../../wiki/entities/rosclaw.md) | 古月居文对比的 IM→ROS2 交互层 |
| [openclaw](../../wiki/entities/openclaw.md) | 助手运行时参考线 |
| [ros2-control](../../wiki/entities/ros2-control.md) | ROS2 执行与控制器生态 |

## 来源博文

- [古月居：用自然语言控制 ROS2 机器人的完整技术方案](../blogs/wechat_guyue_rosclaw_ros2_natural_language.md)
