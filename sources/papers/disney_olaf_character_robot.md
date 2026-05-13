# disney_olaf_character_robot

> 来源归档（ingest）

- **标题：** Olaf: Bringing an Animated Character to Life in the Physical World
- **类型：** paper
- **来源：** arXiv（Disney Research Imagineering）
- **入库日期：** 2026-05-13
- **最后更新：** 2026-05-13
- **一句话说明：** 将动画角色 Olaf 做成可行走、可表演的实机角色：非对称 6-DoF 腿藏于泡沫裙下、远端连杆驱动表情与手臂，RL（PPO）在 Isaac Sim 中跟踪动画参考并叠加热安全与降噪奖励。

## 官方链接（同一稿件多形态）

- **摘要页：** <https://arxiv.org/abs/2512.16705>
- **HTML 全文（v2）：** <https://arxiv.org/html/2512.16705v2>
- **PDF：** <https://arxiv.org/pdf/2512.16705>

## 核心论文摘录（MVP）

### 1) Olaf: Bringing an Animated Character to Life in the Physical World（Müller, Knoop et al., arXiv:2512.16705）
- **链接：** 见上文 abs / html v2 / pdf
- **核心贡献：**
  - **机电一体化：** 约 88.7 cm、14.9 kg、25 DoF；双腿非对称 6-DoF 布局以在狭小躯干内容纳工作空间并减少自碰；下肢藏于聚氨酯泡沫「裙」与雪球足中制造「脚在身体下方自由移动」的视觉效果；肩、眼、嘴等用球面/平面连杆把执行器放到有空间处。
  - **控制分层：** 「骨架」躯干+颈+腿用 RL；眼、眉、下颌、手臂等 show functions 用经典控制（多项式映射 + PD，下颌加布料张力前馈）。
  - **RL 设定：** 行走/站立分立策略；路径坐标系（path frame）对齐动画引擎高层指令 $g_t$；状态含 IMU/关节历史与**执行器温度**；动作为关节 PD 目标；PPO 训练，8192 并行环境（单卡 RTX 4090），约 100k iter。
  - **奖励：** 动画运动学模仿 + 正则（力矩、加速度、动作平滑）+ **基于 CBF 的关节限位与颈部热约束**（温度一阶模型 $\dot T=-\alpha(T-T_\mathrm{amb})+\beta\tau^2$）+ 足–足碰撞惩罚 + **垂向脚速变化饱和惩罚**以显著降低落地噪声。
  - **部署：** 50 Hz 策略 + 一阶保持上采样到 600 Hz + 37.5 Hz 低通；机载状态估计融合 IMU 与执行器测量；木偶式动画引擎切换策略与触发内容（延续作者先前角色管线工作）。
- **对 wiki 的映射：**
  - [Disney Olaf 角色机器人（方法页）](../../wiki/methods/disney-olaf-character-robot.md)
  - [Reward Design](../../wiki/concepts/reward-design.md)
  - [Locomotion](../../wiki/tasks/locomotion.md)
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)
  - [Control Barrier Function](../../wiki/concepts/control-barrier-function.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 关联 wiki 页面的参考来源段落已添加 ingest 链接
