# AprilTag（视觉基准标记与检测库）

> 来源归档

- **标题：** AprilTag / AprilTag 3（C 参考实现）
- **类型：** repo + project-site
- **机构：** APRIL Lab（University of Michigan）发起；当前 C 库由 [AprilRobotics](https://github.com/AprilRobotics) 组织维护
- **链接（本次 ingest）：**
  - 代码仓库：<https://github.com/AprilRobotics/apriltag>
  - 官方项目介绍页：<https://april.eecs.umich.edu/software/apriltag>
- **入库日期：** 2026-05-12
- **一句话说明：** 面向机器人与标定的小型视觉 fiducial 系统：打印即可用，检测库给出 ID 与相对相机的位姿；本仓 C 实现为 AprilTag 3，依赖极少、易嵌入与实时运行。
- **为什么值得保留：** 手眼标定、工作台基准、AR/VR 对齐、PBVS 位姿输入、多机/场地全局参考等场景的默认工程选项之一；与 ROS / OpenCV 生态对接成熟。
- **沉淀到 wiki：** 是 → [`wiki/entities/april-tag.md`](../../wiki/entities/april-tag.md)

## 对 wiki 的映射（编译要点）

- **定位：** 2D 黑白 fiducial；**数据载荷小**（官方介绍称约 4–12 bit 量级思路），换取更远距离与更高检出/位姿精度；与 QR 码目标不同。
- **实现：** README 标明本仓库为 **AprilTag 3**：更快检测、小标签改善、**flexible layouts**、**pose estimation**；小型 **C** 库、**无强制外部依赖**（应用侧自行解决采图）；官方推荐多数场景使用 **tagStandard41h12**，图案见 [apriltag-imgs](https://github.com/AprilRobotics/apriltag-imgs)；自定义族见 [apriltag-generation](https://github.com/AprilRobotics/apriltag-generation)。
- **理论来源：** Olson 2011（ICRA）、Wang 2016（IROS）、Krogius 2019（IROS）等论文链，集中索引于 UMich 软件页与仓库 README「Papers」节。
- **平台说明：** README 写官方仅支持 **Linux**，Windows 有社区成功案例；UMich 页另列邮件列表、历史 Java/iOS 叙述（工程上优先以当前 C 库与绑定为准）。
