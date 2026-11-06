# Learning to Walk in Minutes 论文通读（足式运动任务视角）

> 来源归档（原文通读 · 支撑 `wiki/tasks/locomotion.md`）

- **标题：** Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning
- **作者：** Nikita Rudin, David Hoeller, Philipp Reist, Marco Hutter（ETH Zurich / NVIDIA）
- **类型：** paper / legged-locomotion / rl / sim2real
- **原文：** <https://arxiv.org/abs/2109.11978>（CoRL 2021，2026-11 通过 arXiv PDF v3 获取全文）
- **一句话说明：** 用 Isaac Gym 端到端 GPU 管线并行数千台机器人 + PPO + 自动地形课程，在单卡上约 20 分钟训出可上真机的 ANYmal 崎岖地形行走策略（legged_gym 的论文来源）。
- **核对范围：** 通读正文 §1–§5（含 PPO 超参 / 重置处理、课程、观测动作奖励、sim2real、并行规模实验、多机器人泛化）与附录 A.1（吞吐分析）、A.2（超时 bootstrapping）；附录 A.3 奖励项表只浏览了符号表，未逐项核对权重数值；未运行其开源代码。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **训练时间（摘要 / §1）：** 平地策略 4 分钟内、不平地形约 20 分钟；对比此前报告的 12 / 82 / 88 / 120 小时训练（§1 引 [16][17][18][19]）。
- **并行与 batch（§2.2.1）：** 批大小 B = n_robots × n_steps；机器人数量增大时 n_steps 必须变小，但 GAE 需要时间连贯的轨迹，少于约 25 步（0.5 s 仿真时间）算法难以收敛；mini-batch 取数万样本反而更稳。
- **超时重置（§2.2.2）：** 超时（time-out）与失败终止要区分，超时时用 critic 自身预测 bootstrap；附录 A.2 报告总奖励约提升 10%–20%。
- **并行规模权衡（§4.1）：** 机器人过多时每台的时间视野过短导致性能骤降；过少则样本相关性高。4096 台、batch 约 98304 为文中选用点；训练时间在约 4000 台前近似线性缩短。
- **地形课程（§3.1）：** 五类程序化地形（平地 / 斜坡 / 随机粗糙 / 离散障碍 / 台阶），走出地形边界升级、移动不足目标速度所需距离的一半则降级，通关后随机回环以防灾难性遗忘；台阶高度 5 cm 增至 20 cm，坡度 0° 增至 25°。
- **观测 / 动作 / 奖励（§3.2）：** 观测含基座线 / 角速度、重力向量、关节位置速度、上一动作、108 个地形高度采样点；动作是期望关节位置，由 PD 产生力矩；奖励 9 项加权和（速度跟踪为主，惩罚力矩、关节加速度、动作变化、碰撞，另加迈大步项）；奖励与动作空间无步态相关设计。
- **sim2real（§3.3、§4.3）：** 摩擦系数 [0.5, 1.25] 随机化、观测噪声、每 10 s 随机推（基座最大 ±1 m/s）；ANYmal 用 LSTM 执行器网络；真机高度图来自 LiDAR 高程图、质量不完美，真机速度指令上限降到 0.6 m/s。
- **仿真结果（§4.2）：** 台阶 0.2 m 内成功率近 100%；斜坡超过 25° 无法爬升；策略总收敛到 trot，但常有拖腿、基座过高 / 过低等伪影，需调奖励权重后才可上真机。
- **跨机型（§4.2）：** ANYmal B、带臂 ANYmal C 无需改动即可重训；Unitree A1 需去掉执行器网络、降低 PD 增益；Cassie 需额外「单脚站立」奖励才能学出行走步态。
- **局限（§5、§4.3）：** 目标是「快速可用」而非最强鲁棒性；高度图不完美使真机鲁棒性低于仿真；作者计划与 teacher-student 方案（Lee 等）融合。

## 对 wiki 的映射

- [`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md)
- [`wiki/entities/paper-anymal-walk-minutes-parallel-drl.md`](../../wiki/entities/paper-anymal-walk-minutes-parallel-drl.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2109.11978>
