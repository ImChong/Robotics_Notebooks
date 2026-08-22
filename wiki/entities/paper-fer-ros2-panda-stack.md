---
type: entity
tags: [paper, ros2, franka-panda, position-control, hardware-interface]
status: complete
updated: 2026-08-22
arxiv: "2608.19740"
related:
  - ./franka-research-3.md
  - ../tasks/manipulation.md
  - ./paper-cartesian-impedance-controller.md
  - ../overview/video-contact-control-10-papers-technology-map.md
sources:
  - ../../sources/papers/fer_ros2_arxiv_2608_19740.md
  - ../../sources/sites/fer-ros2-google-sites.md
  - ../../sources/blogs/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md
summary: "FER ROS 2 栈（arXiv:2608.19740）：异步硬件接口 + rate-matching + 位置域参考，恢复 Panda 可靠外部位置控制；站点演示齐全，公开代码链待发布。"
---

# FER ROS 2 Panda 栈

**Keeping the Franka Emika Panda alive: a ROS 2 stack with a reliable position interface**（[arXiv:2608.19740](https://arxiv.org/abs/2608.19740)，[项目页](../../sources/sites/fer-ros2-google-sites.md)）——（见论文；双平台 Panda 验证）。

## 一句话定义

**机器人基础栈的可靠性本身就是研究加速器——先治好位置接口时序抖动。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| FER | Franka Emika Robot | 社区所称 Panda 研究臂 |
| FCI | Franka Control Interface | 底层实时控制接口 |
| ROS 2 | Robot Operating System 2 | 机器人中间件第二代 |

## 为什么重要

- 纳入 [具身智能小站 2026-08-22 十篇盘点](../../sources/blogs/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md) 的「视频→接触→控制→VLA 持续学习」主线。
- 开源状态（入库日）：**待发布**。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | （见论文；双平台 Panda 验证） |
| **出处** | arXiv:2608.19740（2026-08） |
| **开源** | **待发布** |

### 流程总览

```mermaid
flowchart LR
  cmd[外部位置/速度指令] --> async[异步 FCI 通信线程]
  async --> match[Rate-matching]
  match --> ref[位置域平滑参考]
  ref --> hw[ros2_control 硬件接口]
```

## 结论

**Panda 位置控制不稳的根因是控制回路时序，而非机械本体极限。**

- 解耦实时通信与 ros2_control 主循环
- 两独立实验室 Panda 验证 MoveIt/柔顺/遥操作
- 依赖社区 libfranka 分支与现代 FCI
- 公开仓库仍为匿名链（2026-08-22）

## 源码运行时序图

**不适用**（截至 **2026-08-22**）：官方训练/推理入口尚未公开发布。

## 与其他页面的关系

- [franka-research-3](./franka-research-3.md)
- [manipulation](../tasks/manipulation.md)
- [paper-cartesian-impedance-controller](./paper-cartesian-impedance-controller.md)
- [视频–接触–控制 10 篇技术地图](../overview/video-contact-control-10-papers-technology-map.md)

## 参考来源

- [fer_ros2_arxiv_2608_19740](../../sources/papers/fer_ros2_arxiv_2608_19740.md)
- [fer-ros2-google-sites](../../sources/sites/fer-ros2-google-sites.md)
- [wechat_embodied_station_video_contact_control_10_papers_2026-08-22](../../sources/blogs/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md)

## 推荐继续阅读

- [arXiv:2608.19740](https://arxiv.org/abs/2608.19740)
