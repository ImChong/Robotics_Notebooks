# PushMimic: Learning Adaptive Humanoid Box Pushing via Dynamic Motion Augmentation and Reference Distillation

- **作者：** Yulun Zhuang、Hanyu Zhang、Jingyu Liu、Yanran Ding
- **机构：** Department of Robotics, University of Michigan
- **版本：** arXiv:2611.30077v1，2026-11-23；预印本
- **原文：** [摘要](https://arxiv.org/abs/2611.30077) · [HTML](https://arxiv.org/html/2611.30077v1) · [9 页 PDF](https://arxiv.org/pdf/2611.30077v1)
- **作者项目：** https://silvery107.github.io/pushmimic/
- **核对范围：** PDF §I–V、图 1–10 与附录成本和奖励表；对照官网交互演示。未运行训练、仿真或实机。
- **通读：** 已完成

## 方法与证据

- §IV-A 明确名义人体动作由 Kimodo 生成并由 GMR 重定向到 G1；不是测得的单条真人动作捕捉记录。0°、10 kg 为名义条件；离线优化网格是 5–15 kg / 0–15°，间距 1 kg / 1°，共 176 条参考。
- §II 用平面准静态力矩模型解释前倾与足部后撤趋势，不用该模型代替仿真接触。基于采样的轨迹优化对关节目标结点采样，以环境物理 rollout 评分，使用躯干前倾先验、足位死区及箱体局部手位跟踪。
- §III 的观测与奖励用两套同步参考：actor 观测统一平地参考、机器人本体状态和箱体相对姿态/速度；训练按条件优化后的参考重置并评分，critic 还见任务参数和增强参考。并非先分别训练 176 个教师，也不在部署时显式估计坡度与质量。
- 表 I：三种训练种子的 MuJoCo sim2sim 成功率从 57.8 ± 11.6% 到 88.8 ± 1.0%，阈值为终点误差 0.1 m。仿真在训练任务参数网格评估，不是保留未见条件的独立泛化实验。手臂成本是成功条件上的关节力矩平方归一化指标，不是电耗实测。
- §IV-D：23 自由度 G1 改装被动阻尼铰链平掌。坡面跑台分开测试 10° 下 10→15→19 kg，及 10 kg 下 0→12.6°；没有同时测试 19 kg 与 12.6° 的包络顶点。箱体相对状态需要外部动捕。
- [ARCAD Lab](https://sites.google.com/umich.edu/arcad-lab/research) 项目入口与论文吻合。可读的 JS 只证明在线演示开放；未核实完整训练、真机部署源码或可再分发许可证。

- **Wiki：** [PushMimic](../../wiki/entities/paper-pushmimic.md)
