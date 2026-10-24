# WholeBodyMem：人形移动操作的目标前历史记忆（arXiv:2610.31166）

> 来源归档（ingest）

- **标题：** Memory before Goal: Role of Pre-History Memory for Humanoid Loco-Manipulation
- **简称：** WholeBodyMem
- **类型：** paper / humanoid / VLA / object-memory / loco-manipulation
- **arXiv：** <https://arxiv.org/abs/2610.31166>（v1，2026-10-22）
- **项目页：** <https://memory-before-goal.github.io/>
- **作者：** Chenyuan Chen、Ran Ding、Lechen Zhang、Zewen He、Yoshihiko Nakamura
- **投稿信息：** arXiv 页面注明已投稿 IEEE Robotics and Automation Letters（RA-L）
- **入库日期：** 2026-10-24
- **一句话说明：** WholeBodyMem 为人形 VLA 保留机器人执行目标前观察到的物体事件，并通过目标物体引导的关键帧记忆支持后续全身移动操作。

## 摘要级要点

- **问题：** 动态场景中的物体线索会消失、被遮挡，或随机器人转身和移动离开视野；只看当前帧的策略可能丢失目标相关历史。
- **事件引导记忆：** SAM3 跟踪任务物体，物体事件从历史观察中自适应选择关键帧；动作专家通过交叉注意力读取这些帧及编码外观、几何和时间的物体 token。
- **HuMeGo 基准：** 包含六项仿真和六项 Unitree G1 真机任务。
- **摘要所报结果：** 仿真成功率从基线 29.1% 提至 72.6%，真机从 36.7% 提至 88.3%；在 SIMPLE 任务上为 70.2%→86.9%。以上按论文摘要报告，需结合论文评测定义解读。

## 项目与代码

- [WholeBodyMem 官方项目页](https://memory-before-goal.github.io/)
- [arXiv 摘要](https://arxiv.org/abs/2610.31166)
- [arXiv PDF](https://arxiv.org/pdf/2610.31166)

## 对 wiki 的映射

- 实体页：[paper-wholebodymem.md](../../wiki/entities/paper-wholebodymem.md)
- 项目页归档：[wholebodymem.md](../sites/wholebodymem.md)
- 相关主题：人形 VLA、物体记忆、关键帧检索、loco-manipulation
