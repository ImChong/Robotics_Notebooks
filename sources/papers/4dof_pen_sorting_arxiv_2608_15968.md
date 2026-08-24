# Tabletop Pen Manipulation With a Vision-Guided 4-DoF Arm（arXiv:2608.15968）

> 来源归档（ingest）

- **标题：** Tabletop Pen Manipulation With a Vision-Guided 4-DoF Arm
- **类型：** paper / low-cost-robotics / computer-vision / manipulation
- **arXiv abs：** <https://arxiv.org/abs/2608.15968>
- **PDF：** <https://arxiv.org/pdf/2608.15968>
- **代码：** <https://github.com/Anirudhpro/4DoF_vision_robotic_pen_sorting>（归档见 [`sources/repos/4dof-vision-robotic-pen-sorting.md`](../repos/4dof-vision-robotic-pen-sorting.md)）
- **机构：** Dougherty Valley High School；University of Pennsylvania
- **作者：** Anirudh Rangarajan、Bibit Bianchini
- **入库日期：** 2026-08-24
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_9_papers_vla_predict_grasp_2026-08-24.md)

## 开源状态（步骤 2.5，2026-08-24）

- GitHub 仓公开；论文正文链到同一仓库。
- **结论：** **已开源**（感知 + 规划 + 分拣全栈脚本）。

## 摘录 1：硬件与任务

- Waveshare RoArm-M2-S（约 **200 USD**）四自由度臂 + 固定俯视相机。
- 7 种笔具按颜色分拣至四色收纳盒。

## 摘录 2：管线

- **YOLO11n-OBB** 定位与朝向；相机内参 + ArUco 外参映射到机器人坐标；HSV 颜色分类。
- 接近进给方向的笔直接抓取；大角度笔通过 **纠偏扫动** 逐步转到可抓姿态。

## 摘录 3：日志统计

- 326 次动作：196 次直接抓取 + 130 次纠偏扫动；可修正最高 **90°** 错位。

**对 wiki 的映射：** [`wiki/entities/paper-4dof-pen-sorting.md`](../../wiki/entities/paper-4dof-pen-sorting.md)。

## 当前提炼状态

- [x] GitHub 开源核查（已开源）
- [x] 升格 `wiki/entities/paper-4dof-pen-sorting.md`
