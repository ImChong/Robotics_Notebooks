# 单图度量场景对齐与接触引导动作重定向（2611.14981）

> 论文一手原文归档

- **英文题目：** From a Single RGB Image to Robot Execution: Metric Scene Alignment and Contact-Guided Retargeting
- **arXiv：** https://arxiv.org/abs/2611.14981
- **HTML 全文：** https://arxiv.org/html/2611.14981v1
- **PDF：** https://arxiv.org/pdf/2611.14981v1
- **机构：** 北京化工大学、国防科技大学、松延动力
- **入库日期：** 2026-11-16
- **核对范围：** 通读并核对 arXiv v1 III-A/B/C 单图到动作、JMIR/GMR/CGRR、IV 表 I/II/III 与 G1 试验、V 结论；与用户提供的 29 篇 PDF 周报核对研究问题、实验口径与局限。
- **通读：** 已完成
- **一句话说明：** SAM 3D Objects 重建单图场景并生成人体交互参考；JMIR 先优化人–景尺度、原交互相对位置与平滑；GMR 适配 G1 结构；CGRR 以足底接触几何做有限残差修正，最终交给可互换的预训练动作跟踪器。

## 原文摘录（提炼）
1. **问题与动机：** 单目重建的场景尺度误差使人体交互错位，而人体到 G1 的关节映射又会额外破坏脚底支持关系。
2. **方法：** SAM 3D Objects 重建单图场景并生成人体交互参考；JMIR 先优化人–景尺度、原交互相对位置与平滑；GMR 适配 G1 结构；CGRR 以足底接触几何做有限残差修正，最终交给可互换的预训练动作跟踪器。
3. **实验：** 各方法 100 次仿真时完整 JMIR+GMR+CGRR 的支撑误差 0.21 cm、MPJPE-L 38.2 mm、成功率 89%；单 GMR 为 0.57 cm、50.7 mm、75%。G1 真机 50 次成功率 70%；源端精修 87 s vs 对照 GRAIL 480 s。
4. **边界与局限：** 支持主导的交互为主；未覆盖任意复杂柔性物体，单图遮挡可能造成几何不确定性。多个 tracker 的实机对照样本量较少。

## 开源状态
未从本次原文核实到作者发布的可运行算法代码入口；有第三方框架引用不表示本论文完整开源。

## 对 wiki 的映射
- [Single RGB to Robot](../../wiki/entities/paper-rgb-scene-contact-retargeting.md)
