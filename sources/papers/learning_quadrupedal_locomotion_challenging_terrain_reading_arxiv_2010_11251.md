# Learning Quadrupedal Locomotion over Challenging Terrain 论文通读（足式运动任务视角）

> 来源归档（原文通读 · 支撑 `wiki/tasks/locomotion.md`）

- **标题：** Learning Quadrupedal Locomotion over Challenging Terrain
- **作者：** Joonho Lee, Jemin Hwangbo, Lorenz Wellhausen, Vladlen Koltun, Marco Hutter（ETH Zurich / Intel）
- **类型：** paper / legged-locomotion / rl / privileged-learning / sim2real
- **原文：** <https://arxiv.org/abs/2010.11251>（Science Robotics 5(47), eabc5986, 2020；2026-11 通过 arXiv PDF 获取全文）
- **一句话说明：** 在仿真中先训带特权信息的 teacher，再蒸馏为只用本体感知历史的 TCN student，配合自适应地形课程，零样本部署到泥、雪、植被、碎石等自然环境。
- **核对范围：** 通读摘要、§1 引言、§2 结果（自然环境、DARPA SubT、室内对比实验、Table 1）、§3 讨论与局限、§4 方法中的概述、运动生成（FTG / PMTG）、teacher 策略、student 策略与地形可通过性课程（至公式 5）；§4 后半段（课程采样细节、仿真中的执行器 / 随机化设置）与全部补充材料（Table S4–S7、奖励定义 S4）只浏览，未逐项核对；未运行代码。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题设定（§1）：** 传统方案靠状态机编排运动基元与反射，并依赖接触 / 打滑显式估计（经验阈值），在泥、雪、植被中会失效；外感知（相机 / LiDAR）在雪、水、植被下不可靠，因此以本体感知（关节编码器 + IMU）为核心。
- **三个关键要素（§1）：** (1) 用 TCN 处理本体感知历史，隐式推断接触与打滑，不设显式估计模块；(2) 特权学习：直接 RL 训练监督信号稀疏、学不出来，改为先训特权 teacher 再蒸馏 student；(3) 按策略当前水平自适应合成地形的课程（粒子滤波维持中等难度地形参数分布）。
- **控制结构（§4）：** 策略 50 Hz、关节 PD 400 Hz；用 PMTG：策略输出各腿频率偏移与足端位置残差（16 维动作），经足端轨迹发生器 + 解析 IK 变成关节位置目标；基础频率 f0 = 1.25 Hz（trot）。仿真使用学习得到的关节 PD 执行器模型以利迁移。
- **训练（§4）：** teacher 用 TRPO，特权信息含地形轮廓、接触状态与力、摩擦系数、外部扰动；student 用 TCN 模仿 teacher 的潜变量与动作；指令仅给方向与转向（{-1,0,1}），不给目标速度，因为崎岖地形上可行速度不明确。
- **课程（§4）：** 以可通过性 Tr 在 [0.5, 0.9] 内的地形为「合适难度」。仅用刚性程序化地形（hills / steps / stairs）训练。
- **零样本泛化（§2）：** 同一控制器在泥、苔藓、植被、雪、流水、碎石上无需调参；森林中对比基线（ANYmal 既有控制器）：速度更高（Table 1：苔藓 0.452 vs 0.199 m/s）、机械 CoT 更低；基线遇树枝 / 植被 / 泥频繁失败，该控制器未失败。
- **DARPA SubT（§2）：** 驱动两台 ANYmal-B 完成 4 场各 60 分钟任务，零失败；文中称这是无模型 RL 腿足控制器首次用于此类竞赛部署。
- **对比实验（§2）：** 单台阶成功率高于基线；10 kg 负载（约 22.7% 体重，训练中未仿真）下仍能过 13.4 cm 台阶，基线任何台阶都过不了；负载下航向误差保持约 10° 内，基线侧向可达约 30°；自发出现「脚被绊住即抬脚」反射，训练中未显式设计。
- **局限（§3）：** 只出现 trot 步态；纯本体感知（盲走）步态保守，被指令走向悬崖也会走下去，未来需融合外感知；作者认为不必在仿真里还原泥、雪等复杂物理。

## 对 wiki 的映射

- [`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md)
- [`wiki/concepts/sim2real.md`](../../wiki/concepts/sim2real.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2010.11251>
