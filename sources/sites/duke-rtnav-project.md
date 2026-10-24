# RTNav 项目页（Duke General Robotics Lab）

> 来源归档（site）

- **标题：** RTNav: Towards Real-Time Zero-Shot Object Navigation
- **类型：** site
- **项目页：** <https://generalroboticslab.com/RTNav>
- **论文：** <https://arxiv.org/abs/2608.26496>
- **仿真代码：** <https://github.com/generalroboticslab/RTNav>
- **真实机器人代码：** <https://github.com/generalroboticslab/RTNav-RealWorld>
- **机构：** 杜克大学（Duke University），General Robotics Lab
- **入库日期：** 2026-10-24
- **一句话说明：** 项目主页汇总论文、代码与演示；研究重点是将推理延迟、异步环境推进和有限算力作为零样本物体导航系统的设计约束。
- **沉淀到 wiki：** [`wiki/entities/paper-rtnav.md`](../../wiki/entities/paper-rtnav.md)

## 开源状态

- **已开源：** 项目页链接到官方仿真评测仓库与真实机器人部署仓库；两者均可公开访问。
- **仿真仓库：** Habitat 环境以 30 Hz 运行，环境与 agent 通过 ROS 2 交换状态和动作；提供多种同步/异步基线评测。
- **真实机器人仓库：** 公开 Stretch 3 + Jetson AGX Thor 部署代码与启动说明。