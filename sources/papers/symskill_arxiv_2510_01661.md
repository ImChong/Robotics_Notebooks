# SymSkill: Symbol and Skill Co-Invention for Data-Efficient and Reactive Long-Horizon Manipulation（一手论文阅读归档）

- **类型：** paper / 机器人研究获奖论文
- **标题：** SymSkill: Symbol and Skill Co-Invention for Data-Efficient and Reactive Long-Horizon Manipulation
- **中文题名：** SymSkill：低数据量下符号与技能共发明的可反应长程操作
- **arXiv：** <https://arxiv.org/abs/2510.01661>
- **原文 HTML：** <https://arxiv.org/html/2510.01661>
- **PDF：** <https://arxiv.org/pdf/2510.01661>
- **版本：** v3（2026-06-08）
- **会议奖项线索：** ICRA 2026 Best Conference Paper / Best Paper on Planning and Control（奖项由二手综述提出，与一手项目资源交叉核对）
- **项目页：** <https://symskill.github.io/>
- **代码：** <https://github.com/shaoyifei96/Symskill>
- **核对范围：** arXiv HTML v3：摘要、§I–VII、方法中参考帧/谓词/算子/DS、RoboCasa/Franka 实验表 I–III、Appendix A 在线重规划伪代码及附录对比；参考文献未逐条阅读。
- **通读：** 已完成
- **开源核验：** 已开源：Predicators 实现、RoboCasa/robosuite 依赖和调试配置（依赖需逐一配置）
- **入库日期：** 2026-11-17

## 要点摘录与 wiki 映射

1. 状态不是关键帧序列：谓词由对象参考系中的相对位姿簇构造。
2. 学习算子包含 pre、eff、maintain、skill；执行中检查维持条件、失败重规划。
3. 实验对比需同时报告去监控消融和小样本 DP，而非只列平均成功率。

- **技术机制：** 离线先按末端运动与物体运动分割演示；通过运动物体或 VLM 选择参考帧，在相对 SE(3) 坐标中聚类动作起止位姿，形成姿态谓词。比较演示抽象状态变化提炼算子的 pre/effect/maintain 条件，使用 LPV-DS 拟合吸引子技能。在线根据目标谓词规划算子序列、执行低层控制，监视 maintain/effect，并在失败后重采样或全局重规划。
- **评测证据：** RoboCasa 12 项单步任务各 10 次试验，平均成功率 **85.0%**；去除在线监控为 **65.0%**，少量演示下 Diffusion Policy 对照为 **3.3%**。作者展示在 Franka 上从约 5 分钟 play 数据组合 12 步任务。跨任务组合不需额外示范。
- **局限：** 依赖识别出操作物体和稳定的相对坐标系；高壁容器等几何约束造成碰撞失败。实时重规划不能等同于任意开放场景通用；MOSEK 和多仓依赖提高复现实操门槛。
- **对 wiki 的映射：** [SymSkill：低数据量下符号与技能共发明的可反应长程操作](../../wiki/entities/paper-symskill.md)（每篇论文只创建一个独立详情节点）
- **二手索引：** [2026 四大顶会 12 篇获奖论文](../blogs/wechat_shenlan_four_robotics_conferences_12_awards_2026_11_17.md)（仅奖项线索，非技术证据）
