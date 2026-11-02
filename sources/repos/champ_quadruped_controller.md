# CHAMP 仓库

> 来源归档（repo）

- **标题：** CHAMP
- **类型：** repo
- **链接：** <https://github.com/chvmp/champ>
- **入库日期：** 2026-11-01
- **线索来源：** [wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md](../blogs/wechat_guyueju_quadruped_opensource_deep_dive_2026-11-01.md)（古月居转载，原作者敢敢AUTOHUB）
- **一句话说明：** CHAMP 是用于搭建新四足机器人与开发控制算法的 ROS 开源框架，控制框架基于 MIT Cheetah 的「模式调制 + 阻抗控制」分层控制器论文。
- **最近提交：** 2024-07-04
- **核查：** 仓库可达（`git ls-remote`，浅克隆读 README，不运行代码）
- **文章星标（文章表述）：** 2196；分支 428
- **沉淀到 wiki：** [`wiki/entities/champ-quadruped-controller.md`](../../wiki/entities/champ-quadruped-controller.md)

## README 摘录（入库日人工阅读）

README 核心特性：基于 ROS 导航栈的自主能力、setup-assistant 配置新机器人、预配置 URDF（Anymal、MIT Mini Cheetah、Spot、LittleDog）、Gazebo 仿真，并兼容 SpotMicroAI 与 OpenQuadruped。README 称控制框架依据的是 MIT「Hierarchical controller for highly dynamic locomotion utilizing pattern modulation and impedance control」论文。
