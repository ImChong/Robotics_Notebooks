# Nate711/pupperv3-monorepo

> 来源归档

- **标题：** pupperv3-monorepo
- **类型：** repo
- **链接：** https://github.com/Nate711/pupperv3-monorepo/tree/main
- **入库日期：** 2026-06-04
- **一句话说明：** Pupper v3 机载软件 monorepo：ROS 2 工作区与运行时代码，与 [官方文档](https://pupper-v3-documentation.readthedocs.io/en/latest/development/modifying_code.html) 中的 `~/pupperv3-monorepo` 路径一致。

## 为什么值得保留

- 文档站「Modifying Pupper code」指向的 **唯一权威代码入口**，区别于早期 [StanfordQuadruped](https://github.com/stanfordroboticsclub/StanfordQuadruped) / [easy_quadruped](easy_quadruped.md) 控制栈。
- 与 **Foxglove** 可视化、**Pupper AI**（进行中）及实机 `ssh pi@pupper.local` 工作流绑定。

## 工程要点（文档摘录）

| 项 | 说明 |
|----|------|
| 机内路径 | `~/pupperv3-monorepo` |
| ROS 2 | `~/pupperv3-monorepo/ros2_ws` |
| 远程 | `ssh pi@pupper.local` |
| 参考消息 | `geometry_msgs/Twist` 等（见文档 Colab / ROS 链接） |

## 对 wiki 的映射

- [Stanford Doggo / Pupper](../../wiki/entities/stanford-doggo-and-pupper.md)
- [Pupper v3 文档站](../sites/pupper-v3-documentation-readthedocs.md)
