# Keeping the Franka Emika Panda alive: a ROS 2 stack with a reliable position interface

> 来源归档（ingest）

- **标题：** Keeping the Franka Emika Panda alive: a ROS 2 stack with a reliable position interface
- **短名：** FER ROS 2 Panda 栈
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2608.19740>
- **PDF：** <https://arxiv.org/pdf/2608.19740>
- **项目页：** <https://sites.google.com/view/fer-ros2/>
- **代码：** <https://anonymous.4open.science/r/libfranka-636F>
- **入库日期：** 2026-08-22
- **索引来源：** [具身智能小站 10 篇盘点](../../sources/blogs/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md)（<https://mp.weixin.qq.com/s/EmC4gNgcQdPX34vxy-qSVQ>）
- **一句话说明：** 见下方摘录与 wiki 映射。

## 开源状态（步骤 2.5）

- **待发布**：论文称开源；站点有应用视频；代码链为 **匿名 open-science**（2026-08-22 双盲期），公开 GitHub 待核实。

## 核心摘录（面向 wiki 编译）

### 摘录 1：根因

- 位置控制振动/保护停机来自外部控制回路时序与采样抖动，而非机器人本体。

**对 wiki 的映射：** ros2、franka-panda

### 摘录 2：方案

- 异步硬件接口解耦实时通信与 ros2_control；rate-matching；位置域参考生成。

**对 wiki 的映射：** 真机基础栈

## 对 wiki 的映射

- 升格 [`wiki/entities/paper-fer-ros2-panda-stack.md`](../../wiki/entities/paper-fer-ros2-panda-stack.md)

## 当前提炼状态

- [x] 方法要点与开源核查
- [x] wiki 实体与技术地图回链
