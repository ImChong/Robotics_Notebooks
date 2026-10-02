---
type: entity
tags: [paper, human-motion, video, slam, motion-retargeting]
status: complete
updated: 2026-10-02
arxiv: "2403.17346"
code: https://github.com/yufu-wang/tram
related:
  - ./wham-world-human-motion.md
  - ./gvhmr.md
sources:
  - ../../sources/blogs/humanoid_motion_intelligence_day1_data_retargeting_2026_10_02.md
  - ../../sources/repos/tram.md
  - ../../sources/sites/tram4d-project.md
summary: "TRAM 先在动态人体视频中恢复公制尺度相机轨迹，再用 VIMO 重建人体局部运动，组合得到世界坐标人体轨迹。"
---

# TRAM：Global Trajectory and Motion of 3D Humans from in-the-wild Videos

## 一句话定义

**TRAM** 把相机运动与人体局部运动分别恢复，再将两者组合，得到野外单目视频中的**世界坐标人体轨迹**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| TRAM | Trajectory and Motion of 3D Humans | 全局人体运动恢复方法 |
| SLAM | Simultaneous Localization and Mapping | 估计移动相机轨迹与场景深度 |
| VIMO | Video-based Human Motion Regressor | 回归相机坐标人体运动的时序模型 |

## 为什么重要

移动相机视频中，画面里的人体位移可能来自人或相机。没有世界尺度和相机轨迹，下游机器人重定向会误用根部速度和方向。

## 方法

第一阶段使用双重掩码增强 DROID-SLAM，抑制动态人体对背景几何估计的污染，并将相对场景深度与公制深度预测对齐；第二阶段由 VIMO 估计相机系人体姿态，组合公制相机位姿得到世界系运动。

## 实验与评测

官方项目页报告相较先前工作，全局运动误差下降约 60%；这是**人体重建**的比较，不是机器人动作跟踪成功率。ECCV 2024 工作。

## 与其他工作对比

[WHAM](./wham-world-human-motion.md) 结合图像、相机角速度与足部接触恢复人体；TRAM 显式拆分相机轨迹、公制尺度和 VIMO 姿态分支。[GVHMR](./gvhmr.md) 则强调重力对齐的坐标系。

## 结论

**视频转机器人动作前，先分离移动相机与人体根运动；公制尺度和相机轨迹质量会成为下游重定向的误差上限。**

1. 动态人体应从 SLAM 背景估计中排除。
2. 单目尺度必须借助额外深度约束恢复。
3. 人体重建结果仍需机器人形态与动力学适配。

## 工程实践

先核对相机公制轨迹和人体根速度，再交给机器人骨架映射；[官方仓库](https://github.com/yufu-wang/tram) 的 README 明确了三步运行脚本及模型准备。修改版 DROID-SLAM 需单独编译，SMPL 模型需按许可取得。

## 局限与风险

低纹理或动态背景会损害 SLAM，深度尺度估计也会传播到人体位移；没有直接解决物体接触、机器人关节限制与执行器动力学。

## 源码运行时序图

```mermaid
sequenceDiagram
  participant Video as 视频
  participant Camera as scripts/estimate_camera.py
  participant Human as scripts/estimate_humans.py
  participant Viz as scripts/visualize_tram.py
  Video->>Camera: 帧与人体掩码
  Camera-->>Human: 相机轨迹与场景深度
  Video->>Human: 视频人体帧
  Human-->>Viz: 人体局部运动
  Camera-->>Viz: 公制相机位姿
  Viz-->>Video: 世界系人体轨迹可视化
```

脚本顺序与[官方 README 归档](../../sources/repos/tram.md)一致。

## 关联页面

- [WHAM](./wham-world-human-motion.md)
- [GVHMR](./gvhmr.md)

## 参考来源

- [Day 1 文章逐篇索引](../../sources/blogs/humanoid_motion_intelligence_day1_data_retargeting_2026_10_02.md)
- [官方运行入口](../../sources/repos/tram.md)
- [项目页开放状态](../../sources/sites/tram4d-project.md)
- [官方项目页](https://yufu-wang.github.io/tram4d/)
- [论文](https://arxiv.org/abs/2403.17346)

## 推荐继续阅读

- [TRAM 项目页](https://yufu-wang.github.io/tram4d/)
