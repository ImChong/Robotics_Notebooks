---
title: "没有遥操作也能训人形VLA？loco-manip时代来了！斯坦福 LEGS 用 3DGS 把 loco-manip 数据成本打下来"
author: 具身智能研究室
date: "2026-06-04 11:55:00"
source: "https://mp.weixin.qq.com/s/B1sYOPKg6TQwnNGs-_8NDw"
fetch_tool: "Agent-Reach v1.4.0 + wechat-article-for-ai (Camoufox)"
---

# 没有遥操作也能训人形VLA？loco-manip时代来了！斯坦福 LEGS 用 3DGS 把 loco-manip 数据成本打下来

（正文由 `wechat-article-for-ai` 抓取；推广二维码与社群图片已省略，完整策展见 [`sources/blogs/wechat_embodied_ai_lab_legs_vla_3dgs_loco_manip.md`](../blogs/wechat_embodied_ai_lab_legs_vla_3dgs_loco_manip.md)。）

📄 论文标题：LEGS: Fine-Tuning Teleop-Free VLAs for Humanoid Loco-manipulation in an Embodied Gaussian Splatting World

🏛 机构：Stanford University

🔗 项目链接：https://legsvla.github.io/

📅 发表时间：2026 年 5 月 31 日，arXiv:2606.01458

## 01 这篇仿真论文，重点落在“数据能不能真机可用”

人形 VLA 的 loco-manip 数据，能不能少一点依赖真人遥操作。

LEGS 把物理仿真和视觉真实感放到同一条数据链路里看：

- **物理上，机器人和物体要真的能交互；**
- **视觉上，VLA 看到的画面要尽量接近真机头部相机。**

## 02 它为什么要用 3DGS：核心是降低重采成本

Teleop 做 50 条 Task 3 数据，大约要 **1.5 小时**；换一个新条件，还要 **超过 1.5 小时**。LEGS 初始生成大约 **0.5 小时**，换新条件大约 **0.1 小时**。

同一段 motion 可以在新背景、新物体上重新渲染。

## 03 LEGS 的管线：视觉前端和物理后端分开干活

输入：场景视频、物体照片、机器人 URDF。3DGS 重建静态背景，SAM3D 处理物体 mesh。视觉前端与 MuJoCo 物理后端解耦；程序化生成 Walk/Pick/Place episode；微调 ψ0、π0.5、GR00T N1.6。

## 04 视角对比：视觉仿真不能太糊弄

mesh-only 与 LEGS（含颜色校准）在真机头部相机视角下差距明显；去掉 3DGS 背景与颜色校准后成功率明显下降。

## 05 三个任务

Task 1 桌面操作；Task 2 走到桌边再 pick-place；Task 3 走、拿、转身、到低桌蹲下放置。

## 06 结果

LEGS(200) 在 9 个 backbone×task 组合里最好或并列最好。Task 3：Teleop(50) 三个 backbone 均为 **0/10**；LEGS(200) 为 5/10、2/10、6/10。SAM3D mesh-only 平均约 **33%**，LEGS(200) 约 **67%**。

## 07 边界

依赖 SONIC 低层全身控制；程序化 motion primitive；场景需 1–2 分钟手持视频建 3DGS；主要评测 Unitree G1 + RealSense D435；动态场景与透明/反光物体未充分展开。

## 08 收束

人形 VLA 数据工厂将混合遥操作、真人视频、3DGS 场景、物理仿真与程序化任务；LEGS 给出「采一次、重渲染、多场景复用」的具体路径。
