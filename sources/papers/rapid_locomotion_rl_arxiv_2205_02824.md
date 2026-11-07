# Rapid Locomotion via Reinforcement Learning（Agile Locomotion via Model-free Learning）

> 来源归档

- **标题：** Rapid Locomotion via Reinforcement Learning
- **博文别名：** Agile Locomotion via Model-free Learning（Robot Daycare 清单用名）
- **类型：** paper
- **作者：** Gabriel B. Margolis, Ge Yang, Kartik Paigwar, Tao Chen, Pulkit Agrawal
- **机构：** MIT CSAIL（Improbable AI）
- **链接：** https://arxiv.org/abs/2205.02824
- **PDF：** https://arxiv.org/pdf/2205.02824
- **项目页：** https://agility.csail.mit.edu/ ；https://sites.google.com/view/model-free-speed/
- **代码：** https://github.com/Improbable-AI/rapid-locomotion-rl
- **入库日期：** 2026-07-25（2026-11-07 按 arXiv v1 全文通读复核）
- **版本：** arXiv:2205.02824v1（2022-05-05），pdftotext 提取全文
- **核对范围：** 通读摘要、§I 引言、§II 硬件与仿真设置、§III 方法（控制架构：观测 / 动作 / 奖励概览，教师–学生训练，策略优化，速度指令课程 Box 与 Grid 的思路）、§IV 结果（课程对比、真机室内外测试、表 IV sim-to-real 差距、系统辨识与粗糙地形训练消融）、§V 相关工作；附录里 Box / Grid 课程更新规则的逐式推导与奖励表 VI 逐项数值只浏览，未逐项核对；项目页视频与代码未读、未运行。
- **通读：** 已完成
- **一句话说明：** 端到端 RL 控制器使 Mini Cheetah 野外高速奔跑/转向，持续速度至 **3.9 m/s**；关键为速度命令自适应课程 + 在线系统辨识式 Sim2Real。
- **开源状态：** **已开源**（Improbable-AI/rapid-locomotion-rl）
- **沉淀到 wiki：** [paper-rapid-locomotion-rl](../../wiki/entities/paper-rapid-locomotion-rl.md)

---

## 核心贡献（摘录）

1. Mini Cheetah 记录级敏捷：3.9 m/s，草地/冰/砾石等。
2. adaptive curriculum on velocity commands。
3. online system identification for sim-to-real（承接先验工作）。

## 对 wiki 的映射

- [paper-rapid-locomotion-rl](../../wiki/entities/paper-rapid-locomotion-rl.md)
- [sim2real](../../wiki/concepts/sim2real.md)
- [curriculum-learning](../../wiki/concepts/curriculum-learning.md)
- [mit-mini-cheetah](../../wiki/entities/mit-mini-cheetah.md)

## 2026-11-07 通读复核补充（出自论文原文）

- **平台与频率：** MIT Mini Cheetah（30 cm 高、9 kg、12 个准直驱执行器、峰值力矩 17 N·m），网络在机载 Jetson TX2 NX 上 50 Hz 运行；Isaac Gym，4000 个并行体、4 亿仿真步（约 92 个实时日，3090 上不到 3 小时）。
- **接口：** 观测 $o_t=[q_t,\dot q_t,g^{ori}_t,a_{t-1}]$ 的历史（IMU 重力方向），拼速度指令得 42 维；动作为 12 维关节位置目标，PD 增益固定 $K_p=20,\ K_d=0.5$（作者选低增益求平滑，实验中未再调）。
- **随机化（表 I）：** 地面摩擦 0.05–4.00、恢复系数 0–1、负载 −1.0 到 3.0 kg、质心 ±0.10 m、电机强度 90%–110%；师生网络：教师编码器 $d_t$（12 维）→ $z_t$（8 维），学生适应模块读 15 步 42 维历史。
- **sim-to-real 差距（表 IV）：** 指令 6.0 m/s 时，仿真中有系统辨识 5.46 m/s、无 5.07 m/s；真机有 3.81 ± 0.09 m/s、无 2.49 ± 0.07 m/s；作者承认剩余差距可能来自仿真评估参数选得不准（如电机强度、摩擦被高估），也可能有任何仿真配置都覆盖不到的真实动力学，各自贡献不确定。
- **粗糙地形消融（图 5）：** 对盲策略，训练时地形越粗糙（4 / 6 / 8 cm 噪声），平地可跟踪的指令面积越小——鲁棒性与平地极速之间存在取舍；仅在平地训练的策略仍可部署于多种户外地形。
- **其他：** 无课程时策略学不会（随机探索难以产生快速运动）；Grid 课程优于 Box 课程；户外 10 m 冲刺 2.94 s（均速 3.4 m/s）；最高持续速度 3.9 m/s，Froude 数 ≥ 1 的首批 RL 结果之一。
