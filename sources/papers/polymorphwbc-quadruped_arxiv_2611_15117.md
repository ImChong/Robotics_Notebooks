# 四足跨机械臂形态末端轨迹全身控制 PolyMorphWBC（2611.15117）

> 论文一手原文归档

- **英文题目：** PolyMorphWBC: Whole-Body Control from End-Effector Trajectories for Quadrupedal Mobile Manipulation across Arm Morphologies with Extended Reach
- **arXiv：** https://arxiv.org/abs/2611.15117
- **HTML 全文：** https://arxiv.org/html/2611.15117v1
- **PDF：** https://arxiv.org/pdf/2611.15117v1
- **机构：** 中国科学技术大学
- **入库日期：** 2026-11-16
- **核对范围：** 通读并核对 arXiv v1 III 轨迹预览、形态条件、底座/臂协调，IV 四臂仿真、Go2–ARX5 实机、V 结论；与用户提供的 29 篇 PDF 周报核对研究问题、实验口径与局限。
- **通读：** 已完成
- **一句话说明：** 任务层只给末端短时预览轨迹；协调器将 EE 目标转换到基座坐标，结合 URDF 提取的形态条件及残余外力反馈，输出底座运动；各机械臂单独 IK 跟踪当前 EE 目标，冻结 locomotion 后端执行底座命令。

## 原文摘录（提炼）
1. **问题与动机：** 末端轨迹本身不告诉四足底座何时移动；不同臂长会改变可达范围与底座调整需求。
2. **方法：** 任务层只给末端短时预览轨迹；协调器将 EE 目标转换到基座坐标，结合 URDF 提取的形态条件及残余外力反馈，输出底座运动；各机械臂单独 IK 跟踪当前 EE 目标，冻结 locomotion 后端执行底座命令。
3. **实验：** 四个 6DoF 臂家族下 held-out anchor arms：ARX X5 99.30%、Compact xArm6 99.27%、PiPER 93.16%、Compact MICO 86.62% 轨迹完成率；MuJoCo 末端位置误差 1.38–2.04 cm；真机 Go2+ARX5 实现超初始臂展的抓放。
4. **边界与局限：** 依赖准确 URDF、单 Go2 底座及低速平滑 EE 轨迹；跨机型的数值多数来自仿真，未见全套官方可运行代码。

## 开源状态
未从本次原文核实到作者发布的可运行算法代码入口；有第三方框架引用不表示本论文完整开源。

## 对 wiki 的映射
- [PolyMorphWBC](../../wiki/entities/paper-polymorphwbc-quadruped.md)
