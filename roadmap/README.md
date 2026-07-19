# 成长路线总览

本目录用于承载 `Robotics_Notebooks` 的成长路线设计。

## 路线设计原则

- 终极目标：全栈机器人工程师
- 当前切入口：机器人运动控制算法工程师
- 当前重点：运动控制、强化学习、模仿学习、人形机器人

## 主路线（核心）

- **[主路线：运动控制算法工程师成长路线](motion-control.md)**  
  含 L−1 序言 → L0–L6 主干 → L7 出口的完整阶段。覆盖人形运动控制的传统控制主干（LIP/ZMP → Centroidal → MPC → TSID/WBC），以及 RL/IL 扩展与 sim2real 实战。

## 纵深路线（按目标选其一深入）

主路线偏"先打通主干"，下列十七条纵深路线**各自独立**，从主路线某个阶段衔接出去，可单独阅读。

排序说明：按各方向**起点里程碑的历史顺序**排列（与首页按钮、README 一致）——越靠前的方向理论积淀越深，越靠后的方向越依赖学习方法与算力：

- [如果目标是力矩控制电机设计（指标 → 电磁热 → FOC 力矩闭环 → 关节模组）](depth-torque-motor-design.md) —— 想把关节电机从任务指标一路做到可验收的力矩闭环模组（起点：磁场定向控制 FOC，1971）
- [如果目标是传统模型控制（LIP/ZMP → MPC → WBC）](depth-classical-control.md) —— 想系统掌握 model-based 人形控制主干（起点：ZMP 判据，1972）
- [如果目标是安全控制（CLF / CBF / Safe RL）](depth-safe-control.md) —— 想加可证明的安全约束（起点：CLF，1983）
- [如果目标是接触丰富的操作任务](depth-contact-manipulation.md) —— 想做装配、插拔、双臂协同等精细接触（起点：阻抗控制，1985）
- [如果目标是导航（SLAM → Nav2 → VLN → 导航 VLA）](depth-navigation.md) —— 想让机器人知道自己在哪、该往哪走（起点：概率 SLAM，1986）
- [如果目标是模仿学习与技能迁移](depth-imitation-learning.md) —— 想从人类演示数据学习技能（起点：行为克隆，1988）
- [如果目标是 RL 运动控制](depth-rl-locomotion.md) —— 想用强化学习驱动人形 locomotion（起点：Q-learning，1989）
- [如果目标是 Loco-Manipulation（移动操作）](depth-loco-manipulation.md) —— 想让机器人边走边动手（起点：移动操作臂协调控制，1994）
- [如果目标是人形足球（全向行走 → 感知踢球 → 多机战术）](depth-humanoid-soccer.md) —— 想让机器人追球、射门、打整场比赛（起点：首届 RoboCup，1997）
- [如果目标是动作重定向（人体动作 → 机器人参考轨迹）](depth-motion-retargeting.md) —— 想把人体动捕/视频动作变成机器人可执行参考（起点：Gleicher 动作重定向，1998）
- [如果目标是 Sim2Real（域差画像 → 执行器对齐 → 鲁棒训练 → 真机部署）](depth-sim2real.md) —— 想把仿真里训好的策略稳定搬上真机（起点：域随机化 DR，2017）
- [如果目标是人形拳击（动作跟踪 → 潜空间技能 → 对抗自博弈）](depth-humanoid-boxing.md) —— 想让两台人形在擂台上像人一样对打（起点：MuJoCo 人形对抗自博弈，2017）
- [如果目标是 BFM（人形行为基础模型）](depth-bfm.md) —— 想用一个 checkpoint 控住人形全身（起点：DeepMimic 动作跟踪谱系，2018）
- [如果目标是感知越障（Perceptive Locomotion）](depth-perceptive-locomotion.md) —— 想让机器人看着地形上楼梯、跨障碍、跑酷（起点：2020s 感知策略浪潮）
- [如果目标是动作生成（文本/多模态 → 人形动作）](depth-motion-generation.md) —— 想用生成模型造出人体/人形动作（起点：MDM 扩散动作生成，2022）
- [如果目标是 VLA（视觉-语言-动作模型）](depth-vla.md) —— 想让机器人听懂指令干活（起点：RT-2 确立 VLA，2023）
- [如果目标是 WAM（世界–动作模型）](depth-wam.md) —— 想让策略在出动作前显式预知世界会怎么变（起点：World Action Models 综述形式化，2026）

说明：感知、规划、系统、部署等更广的全栈知识在主路线的 **L−1 全景层** 和 **L7 出口层** 集中扫盲，提供进入对应子专题的入口，而不在本目录维护并列的独立路线图文件。
