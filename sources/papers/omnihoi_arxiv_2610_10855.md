# OmniHOI 论文来源归档

## 元数据

- **题目：** OmniHOI: Dexterous Hand-Object Interaction from Monocular Human Video
- **作者：** Ting Mao, Yanming Shao, Ziheng Wang, Haoyu Liu, Yiqun Wang, Xuanye Wu, Yao Mu
- **单位（据 HTML 版作者脚注）：** 浙江大学；香港大学；上海交通大学；上海人工智能实验室
- **预印本：** https://arxiv.org/abs/2610.10855
- **HTML：** https://arxiv.org/html/2610.10855v1
- **版本：** v1，2026-10-07，cs.RO
- **项目页：** https://hydrofoooil.github.io/OmniHOI/（归档见 [omnihoi-project.md](../sites/omnihoi-project.md)）
- **代码：** 截至 2026-10-12 未发布（项目页标注 "Code soon"，无仓库链接）
- **核查日期：** 2026-10-12

## 论文要解决的问题

单目 RGB 视频里有大量人手与物体交互，但缺少度量尺度与完整 3D，手物之间还有遮挡；直接把人手轨迹映射到灵巧手，会出现手物穿插、手指交叉、接触位置错误，轨迹在物理仿真中无法驱动物体。OmniHOI 把问题拆成三段，并在每段加入不同类型的物理一致性约束：重建阶段用图像证据，重定向阶段用接触几何，精修阶段用带物理的仿真回路。

## 方法与训练信息

1. **关键帧重建**：SAM 3 分割手、物体、桌面；MoGe-3 给出内参与深度点图；WiLoR 恢复 MANO 手网格；SAM 3D 生成物体网格；FoundationPose 逐帧跟踪物体位姿。自适应关键帧采样只精修约 20% 的帧（r=0.2），按运动累积量等间隔放置关键帧。精修交替做两件事：图像对齐（有序遮挡损失 + 可微光栅化）与接触穿插优化（grasp loss），再加手指交叉修复；优化器为 Adam。
2. **接触感知重定向**：以 WujiHand 的关键点求解器初始化；接触指被拉向演示中的接触区域，同时消除穿插与手指交叉，交叉先修（字典序目标）。
3. **物理回路精修**：关键帧用 PCHIP 插值到 60 Hz，在 MuJoCo 回放；CMA-ES 在滚动时域上优化每个关键帧的偏移量；代价项包括物体位姿误差、接触、对向接触力、穿插、偏移先验与终端惩罚。
4. 该流水线是优化型，没有端到端训练的神经策略；论文称真机执行不需要任务特定的 RL 训练。

## 论文报告的结果

- **数据**：150 条动捕轨迹（TACO 58、OakInk2 62、ARCTIC 30；单手 60、双手 90）；60 段单目视频（TACO 51、OakInk2 9，从 150 条中按遮挡最少、手物像素占比最大挑选）；重建评测在 ARCTIC、DexYCB、OakInk2 各采样 500 张图。
- **五种灵巧手**：Inspire（6 DoF）、XHand（12）、Wuji（20）、SharpaWave（22）、Shadow（20）。
- **成功判据**：物体轨迹 ATE 低于 10 cm 且低于 0.5 rad；双手任务要求两个物体都通过。
- **运动迁移成功率**（Inspire/XHand/Wuji/SharpaWave/Shadow）：OmniHOI 39/89/86/89/85%；SPIDER 26/26/20/19/15%；OmniRetarget 24/31/27/20/16%；ManipTrans 24/22/–/–/26%。单条轨迹耗时：OmniHOI 19 min，SPIDER 37 min，ManipTrans 385 min，OmniRetarget 0.3 min。
- **端到端视频（60 段，XHand）**：OmniHOI 成功率 53%、进度 71%、位置误差 5.7 cm、角度误差 26°；VideoManip 28%/49%/8.6 cm/31°；Do as I Do 25%/48%/9.1 cm/31°。OmniHOI 的 28 次失败中 14 次为感知、13 次为执行、1 次为运动规划。
- **重建**：F5 0.3878、F10 0.5831、CD 0.9574 cm²、MPVPE 6.05 mm、MPJPE 5.71 mm、穿插体积 IV 2.75 cm³；最佳基线 EasyHOI 的 F5 为 0.2187、CD 1.7110 cm²、MPVPE 14.63 mm。
- **消融**：穿插体积 IV 从仅放置的 4.27 降到加图像对齐 3.61、加接触穿插 3.03、加交叉修复 2.75 cm³（PA-MPVPE 由 5.37 升至 6.05 mm）；物理回路把四个高自由度手的成功率从 9–29% 提到 85–89%，Inspire 从 21–25% 提到 36–39%；最大穿插中位数由 9.17 降到 4.73 mm；自适应关键帧 83% 对均匀采样 81%。
- **真机**：两只 XHand 装在 UR5e 上，七项任务（插入、堆叠、清扫、扫描、擦洗、倒水杯/倒水马克杯），在仿真中对齐轨迹并解 IK，开环执行；论文未报告真机成功率表（原文未报告）。

## 复现与解释边界

- 精度受单目度量尺度限制；视频推得的物体几何与物理属性误差会传导到物理精修。
- 真机为开环执行，闭环反馈列为未来工作。
- 39%–89% 的区间主要由手的自由度决定：低自由度的 Inspire 仅 39%。
- 代码未发布，无法核对 MuJoCo 代价项权重与 CMA-ES 超参。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-omnihoi.md
- 汇总页：wiki/overview/frontier-manipulation-2026-10-05-10-09.md
- 项目页归档：sources/sites/omnihoi-project.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.10855>
- arXiv HTML：<https://arxiv.org/html/2610.10855v1>
- 项目页：<https://hydrofoooil.github.io/OmniHOI/>
