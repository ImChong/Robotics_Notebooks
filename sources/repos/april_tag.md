# AprilTag（视觉基准标记与检测库）

> 来源归档

- **标题：** AprilTag / AprilTag 3（C 参考实现）
- **类型：** repo + project-site
- **机构：** APRIL Lab（University of Michigan）发起；当前 C 库由 [AprilRobotics](https://github.com/AprilRobotics) 组织维护
- **版本 / 提交：** `AprilRobotics/apriltag@b7c0ebe`（BSD-2-Clause）
- **链接（本次 ingest）：**
  - 代码仓库：<https://github.com/AprilRobotics/apriltag>
  - 官方项目介绍页：<https://april.eecs.umich.edu/software/apriltag>
- **入库日期：** 2026-05-12
- **一句话说明：** 面向机器人与标定的小型视觉 fiducial 系统：打印即可用，检测库给出 ID 与相对相机的位姿；本仓 C 实现为 AprilTag 3，依赖极少、易嵌入与实时运行。
- **为什么值得保留：** 手眼标定、工作台基准、AR/VR 对齐、PBVS 位姿输入、多机/场地全局参考等场景的默认工程选项之一；与 ROS / OpenCV 生态对接成熟。
- **核对范围：** 核查 `AprilRobotics/apriltag@b7c0ebe` 的 `README.md`、`apriltag.h`、`apriltag.c`（`apriltag_detector_detect`、`quad_decode_task`、`quad_decode`、`reconcile`）、`apriltag_pose.c`（`estimate_tag_pose`、`estimate_pose_for_tag_homography`、`orthogonal_iteration`、`fix_pose_ambiguities`）以及 `example/opencv_demo.cc`。
- **通读：** 已完成
- **沉淀到 wiki：** 是 → [`wiki/entities/april-tag.md`](../../wiki/entities/april-tag.md)

## 步骤 2.5：源码开放核查

| 入口 | 结论 |
|------|------|
| GitHub 仓库 | **已开源**（BSD-2-Clause），核心纯 C99（POSIX 线程），无强制第三方视觉库依赖 |
| 构建与运行入口 | **可运行** — 支持 CMake / Makefile 构建共享库、`apriltag_demo` / `opencv_demo` 示例与 Python `apriltag` 绑定 |
| 关联资源 | 图案集 `AprilRobotics/apriltag-imgs`；自定义族生成器 `AprilRobotics/apriltag-generation` |

## 源码与 API 核心核查（`b7c0ebe`）

1. **零外部依赖检测流水线（`apriltag.c` / `apriltag.h`）：** `apriltag_detector_create()` 创建检测器，`apriltag_detector_add_family()`（默认允许 `bits_corrected = 2`）挂载编码族；`apriltag_detector_detect()` 先按 `quad_decimate` / `quad_sigma` 预处理，经 `apriltag_quad_thresh()` 提取候选四边形，再用 `workerpool` 并行执行 `quad_decode_task`（单应矩阵拟合、边界黑白灰度模型回归、双线性采样 + `sharpen()` 与快速哈希表解码），最后 `reconcile` 阶段对重叠的同 ID 检测按更低 Hamming 误差与更高 `decision_margin` 去重。
2. **平面对称位姿估计（`apriltag_pose.c`）：** `estimate_tag_pose()` 调用 50 次正交迭代（`estimate_pose_for_tag_homography` 初始化 + Lu 2000 `orthogonal_iteration`），并经 `fix_pose_ambiguities` 求解四次多项式以寻找第二局部最优解，最终比较目标空间重投影残差择优返回 `R, t`。
3. **官方示例与工程约定（`README.md` / `example/opencv_demo.cc`）：** 默认推荐通用族 `tagStandard41h12`，同时原生内置 `tag36h11`、`tag52h13`、`tagCircle21h7`/`49h12`、`tagCustom48h12` 与原生 `tagAruco*` 族；`tagsize` 定义为标签内部黑白交界边框的物理边长；OpenCV 接入仅需转 `CV_8UC1` 并零拷贝封装 `image_u8_t`。

## 对 wiki 的映射（编译要点）

- **定位：** 2D 黑白 fiducial；**数据载荷小**（官方介绍称约 4–12 bit 量级思路），换取更远距离与更高检出/位姿精度；与 QR 码目标不同。
- **实现：** README 标明本仓库为 **AprilTag 3**：更快检测、小标签改善、**flexible layouts**、**pose estimation**；小型 **C** 库、**无强制外部依赖**（应用侧自行解决采图）；官方推荐多数场景使用 **tagStandard41h12**，图案见 [apriltag-imgs](https://github.com/AprilRobotics/apriltag-imgs)；自定义族见 [apriltag-generation](https://github.com/AprilRobotics/apriltag-generation)。
- **理论来源：** Olson 2011（ICRA）、Wang 2016（IROS）、Krogius 2019（IROS）等论文链，集中索引于 UMich 软件页与仓库 README「Papers」节。
- **平台说明：** README 写官方仅支持 **Linux**，Windows 有社区成功案例；UMich 页另列邮件列表、历史 Java/iOS 叙述（工程上优先以当前 C 库与绑定为准）。
