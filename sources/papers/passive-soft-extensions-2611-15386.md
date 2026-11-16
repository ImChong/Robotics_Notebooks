# Actuation and Control of Passive Tendon-Driven Soft Extensions Through Rigid Manipulator Motion｜用刚性机械臂驱动无源腱索软体延伸段

- **类型：** 论文
- **原文：** https://arxiv.org/abs/2611.15386（v1）
- **PDF：** https://arxiv.org/pdf/2611.15386
- **HTML：** https://arxiv.org/html/2611.15386v1
- **项目：** https://ignaciocuiral.com/passive_soft_extension_control/
- **代码：** https://github.com/nachocz/passive_soft_extension_control
- **核对范围：** arXiv 原文方法、系统实现边界、主要实验与作者局限；ARTS 使用 7 页原 PDF 并核对系统结构及主结果。未做模型训练或实际硬件实验。
- **通读：** 已完成
- **入库日期：** 2026-11-16
- **wiki：** [论文实体](../../wiki/entities/paper-passive-soft-extensions-2611-15386.md)

## 原文摘录与 Wiki 映射

- **机制：** 软体段的腱索锚定在机械臂上游关节，外部腱索长度变化通过辨识的 span-to-deformation 映射到末端；对末端执行器位置/孔口 RCM 约束，利用零空间控制软体尖端，并以 CBF/QP 确保避障及辨识模型有效域。
- **实验：** UR10 仿真与真实系统中的局部尖端调节；有辨识误差、不可达目标与约束满足对照，不能推广至未经辨识的大形变。
- **局限：** 辨识映射只在探测过的腱索跨度范围有效，冗余度与支点约束会限制可驱动的形变方向。
- **对 wiki 的映射：** 独立论文详情的核心原理、评测及工程实践。
