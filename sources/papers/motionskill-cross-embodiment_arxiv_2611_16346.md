# 从双足视频到跨形态运动控制 MotionSkill（2611.16346）

> 论文一手原文归档

- **英文题目：** MotionSkill: From Bipedal Motion Video to Control across Embodiments
- **arXiv：** https://arxiv.org/abs/2611.16346
- **HTML 全文：** https://arxiv.org/html/2611.16346v1
- **PDF：** https://arxiv.org/pdf/2611.16346v1
- **机构：** Fortune AI
- **入库日期：** 2026-11-16
- **核对范围：** 通读并核对 arXiv v1 III source alignment/embodiment mapping/physical tracking、IV 十机体和遮挡实验、V 局限；与用户提供的 29 篇 PDF 周报核对研究问题、实验口径与局限。
- **通读：** 已完成
- **一句话说明：** 恢复带时间戳的根状态、肢体关键点和步态相位；对遮挡关键点保留 observed/imputed/invalid 标志并适当补全。针对十种机器人用段长、关节约束和闭链做逐帧 IK 映射；每机型在 MuJoCo 中独立训练参考条件的接触残差策略。

## 原文摘录（提炼）
1. **问题与动机：** 同一段双足步态视频有时间节奏但无法直接匹配不同机器人腿长、关节限制与驱动特性。
2. **方法：** 恢复带时间戳的根状态、肢体关键点和步态相位；对遮挡关键点保留 observed/imputed/invalid 标志并适当补全。针对十种机器人用段长、关节约束和闭链做逐帧 IK 映射；每机型在 MuJoCo 中独立训练参考条件的接触残差策略。
3. **实验：** 基于一段 4 秒步态视频在十种双足机器人上仿真训练；未见初始状态的原视频动作完成率约 98%，上身遮挡 1 秒后约 26%。G1 二次视频试验原始/遮挡的关节 RMS 为 0.123/0.133 rad，水平位置误差 0.207/0.352 m。
4. **边界与局限：** 每机体仍分别训练策略，不是一个策略零样本泛化；只覆盖短视频步态和 MuJoCo 模拟。

## 开源状态
未从本次原文核实到作者发布的可运行算法代码入口；有第三方框架引用不表示本论文完整开源。

## 对 wiki 的映射
- [MotionSkill](../../wiki/entities/paper-motionskill-cross-embodiment.md)
