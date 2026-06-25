# Embrace Collisions: Humanoid Shadowing for Deployable Contact-Agnostics Motions（arXiv:2502.01465）

> 来源归档（ingest）

- **标题：** Embrace Collisions: Humanoid Shadowing for Deployable Contact-Agnostics Motions
- **类型：** paper / humanoid / whole-body-control / contact-rich / shadowing / sim2real
- **venue：** Conference on Robot Learning（CoRL）2025
- **arXiv abs：** <https://arxiv.org/abs/2502.01465>
- **arXiv HTML：** <https://arxiv.org/html/2502.01465v1>
- **PDF：** <https://arxiv.org/pdf/2502.01465>
- **项目页：** <https://project-instinct.github.io/embrace-collisions>
- **机构：** 清华大学交叉信息研究院（IIIS, Tsinghua）、上海期智研究院（Shanghai PIL）
- **硬件：** Unitree G1；机载 Jetson Orin NX 运行策略；高层运动命令由外置笔记本 ROS2 回放（无动捕）
- **入库日期：** 2026-06-25
- **一句话说明：** **关键帧式基座系运动命令** + **Transformer 编码器** + **多 critic advantage mixing** + **目标偏差终止条件**，在简化碰撞体与域随机下零样本部署 **坐、爬、地面翻滚、起身** 等全身多接触 shadowing（非经典 AMP，但属 AMP 专题 #19 收束篇）。

## 摘要级要点

- **问题：** 前人形 RL 假设 **仅足/手接触**；膝、髋、肘、躯干触地使 **终止条件、命令接口、奖励稀疏性** 全部失效；MPC 难实时规划随机接触序列。
- **极端动作数据集：** AMASS 多为站立 → 自建 **extreme-action**：AMASS 起身片段 + **4D-Human** 互联网视频跟踪 retarget；覆盖大 roll/pitch、低基座高度。
- **运动命令（基座系）：** 每帧拼接 $[\tilde{p}_b,\tilde{\alpha}_b,\hat\theta^i,\hat{l}_b^j,(\hat\theta^i-\theta^i),(\hat{l}_b^j-l_b^j),t_{\text{passed}},t_{\text{left}}]$；**稀疏关键帧奖励**（$t_{\text{left}}=0$ 时计任务奖）+ 稠密正则 → **advantage mixing**（3 组 reward / 3 critic，加权归一化 advantage）。
- **Transformer 命令编码：** 可变长度目标序列 token 化；取 **最小正 $t_{\text{left}}$** 对应 embedding 与本体历史 MLP 出动作；训练加 state-target 防序列耗尽。
- **终止重定义：** 仅在 **应到达目标帧** 检查：基座位置误差 $>0.5$ m、四元数虚部阈值、单关节误差 $>1.0$ rad——跪/躺不再视为 fall。
- **部署：** 无全局里程计假设；回放仿真录制的 **基座系目标位姿** + IMU 实时对齐 yaw 修正 $q_{\text{yaw,correct}}$；FK 目标连杆误差 ONNX 加速。
- **仿真成功率（Table I）：** Multi-critic + 精选动作：起身 **94.3%**、地面交互 **98.3%**；单 critic 或全 AMASS 训练显著下降。

## 核心摘录（面向 wiki 编译）

### 与 AMP 专题 #19 的关系

- **非对抗运动先验**，而是 **可部署 contact-agnostic shadowing**；专题收束强调：AMP 线之外，**全身多接触 + 命令接口 + sim2real** 同样是「像身体」的边界问题。
- 与 [Deep Parkour #18](../../wiki/entities/paper-deep-whole-body-parkour.md) 互补：本文 **地面极端姿态**，彼 **感知障碍全身穿越**。

### 与跟踪 / locomotion 对照

| 维度 | Embrace Collisions | 速度命令 locomotion | CLOT #16 |
|------|-------------------|---------------------|----------|
| 接触 | **全身随机接触** | 足–地 | 足为主 + 跟踪 |
| 命令 | 关键帧基座系序列 | $v_x,v_y,\omega$ | 全局参考窗口 |
| 奖励 | 稀疏目标 + mixing | 稠密跟踪 | AMP + 任务 |
| 终止 | **相对目标偏差** | 倾倒/高度 | 跟踪误差 |

## 对 wiki 的映射

- 沉淀实体页：[Embrace Collisions（AMP #19）](../../wiki/entities/paper-amp-survey-19-embrace_collisions.md)
- 交叉：[project-instinct](../../wiki/entities/project-instinct.md)、[whole-body-control](../../wiki/concepts/whole-body-control.md)、[sim2real](../../wiki/concepts/sim2real.md)、[unitree-g1](../../wiki/entities/unitree-g1.md)、[paper-deep-whole-body-parkour](../../wiki/entities/paper-deep-whole-body-parkour.md)

## 参考来源（原始）

- arXiv:2502.01465
- [humanoid_amp_survey_19_embrace_collisions_humanoid_shadowing_for_deploy.md](humanoid_amp_survey_19_embrace_collisions_humanoid_shadowing_for_deploy.md)
- [wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)
