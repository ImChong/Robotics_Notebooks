# ControlIt!（Whole-Body Operational Space Control 中间件）

- **标题：** ControlIt! — A Whole-Body Operational Space Control Middleware
- **类型：** repo / software framework
- **仓库：** <https://github.com/liangfok/controlit>
- **论文：** arXiv:[1506.01075](https://arxiv.org/abs/1506.01075)
- **机构：** 德州大学奥斯汀分校（UT Austin）HCRL / Luis Sentis 组（论文作者）
- **许可：** LGPL-2.1
- **收录日期：** 2026-09-15
- **开源结论：** **已开源**（截至入库日 README 与源码可获取；依赖 **ROS Indigo**、Gazebo、RBDL、yaml-cpp 0.3.0 等，属 legacy 栈）

## 一句话摘要

面向 **WBOSC（Whole Body Operational Space Control）** 的 **ROS Catkin 插件化中间件**：Task/Constraint 插件扩展 WBC 原语，两插件 + URDF 适配新机器人；多线程与参数绑定降低伺服延迟（论文报告 ~0.5 ms vs UTA-WBC ~5 ms）。

## 为何值得保留

- **Sentis–Khatib WBC 理论线的可运行软件后继**（相对 ICRA 2006 论文无代码）。
- **插件架构范本：** 与 [Stack of Tasks](../../wiki/entities/paper-hmi-stack-of-tasks.md)、[TSID](https://github.com/stack-of-tasks/tsid) 同属「任务栈 + 求解器」工程谱系，但专注 **操作空间力/运动统一 + 浮基**。
- **历史基准：** Dreamer 上半身力控平台与拆解任务演示，适合理解 WBC 软件集成问题（线程、绑定、插件边界）。

## 环境与依赖（编译自 README）

| 组件 | 说明 |
|------|------|
| ROS | Indigo + Catkin workspace |
| 仿真 | Gazebo 插件与模型包（同生态多仓） |
| 动力学 | RBDL |
| 配置 | yaml-cpp 0.3.0 |
| 系统 | 需配置 shared memory |

## 对 Wiki 的映射

- [ControlIt! 实体页](../../wiki/entities/controlit.md)
- [controlit_arxiv_1506_01075.md](../papers/controlit_arxiv_1506_01075.md)
- [whole-body-control](../../wiki/concepts/whole-body-control.md)、[hub-wbc](../../wiki/overview/hub-wbc.md)

## 参考来源（原始）

- 代码：<https://github.com/liangfok/controlit>
- 论文：<https://arxiv.org/abs/1506.01075>
