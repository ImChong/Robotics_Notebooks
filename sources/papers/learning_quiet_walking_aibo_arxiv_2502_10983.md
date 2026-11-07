# Learning Quiet Walking for a Small Home Robot（arXiv:2502.10983）

> 来源归档（ingest）

- **标题：** Learning Quiet Walking for a Small Home Robot
- **类型：** paper / quadruped locomotion / quiet walking / sim2real RL / variable PD / home robot
- **arXiv abs：** <https://arxiv.org/abs/2502.10983>
- **arXiv HTML：** <https://ar5iv.labs.arxiv.org/html/2502.10983>
- **PDF：** <https://arxiv.org/pdf/2502.10983>
- **会议：** ICRA 2025（accepted）
- **作者：** Ryo Watanabe（ETH / Sony）、Takahiro Miki、Fan Shi（ETH / NUS）、Yuki Kadokawa、Filip Bjelonic、Kento Kawaharazuka、Andrei Cramariuc、Marco Hutter
- **机构：** 苏黎世联邦理工（ETH Zürich RSL）；索尼集团（Sony Group Corporation）；新加坡国立大学（NUS）；奈良先端科学技术大学院大学（NAIST）；东京大学（The University of Tokyo）
- **项目页：** <https://sony.github.io/QuietWalk/>（归档见 [sony-quietwalk-github-io.md](../sites/sony-quietwalk-github-io.md)）
- **硬件：** Sony **aibo**（小型家用四足；12 腿关节；足底 **开关式接触传感**；无 F/T 力传感）
- **仿真：** Isaac Gym；物理 400 Hz / 策略 100 Hz；PPO + MLP（3×128, ELU）
- **入库日期：** 2026-08-02（2026-11-07 按 arXiv v2 全文通读复核）
- **版本：** arXiv:2502.10983v2（2025-02-28），pdftotext 提取全文
- **核对范围：** 通读摘要、§I 引言、§II 相关工作、§III 方法（动作与可变 PD 公式 (1)、观测表 I、奖励表 II、课程学习、域随机化表 III）、§IV 实验（声学评估、噪声惩罚代理表 IV、鲁棒–安静权衡与四项消融、PD 增益缩放分析图 5）、§V 讨论；参考文献仅浏览；表 II / III 的部分数值列在 pdftotext 中错位，未引用错位数值；未运行代码（官方未开源）。
- **通读：** 已完成
- **一句话说明：** 用仿真中可算的 **足端接触速度** 等「噪声行走惩罚」作声学代理，配合 **策略输出可变 PD gain**、**足底开关接触** 与 **noisy→quiet 两阶段课程**，在 aibo 真机上得到比 RL 基线与索尼商用 normal/quiet 控制器更安静的步态，并显式展示 **安静度–鲁棒性** 权衡。

## 开源状态（核查，2026-08-02）

- **项目页已上线**，GitHub 组织仓 `sony/QuietWalk` **仅为项目页源码**（Academic project page template）。
- **确认未开源：** 无训练代码、checkpoint、aibo SDK 集成或部署脚本；wiki **源码运行时序图：不适用**。
- 勿与后人形论文 [QuietWalk（arXiv:2604.23702）](./quietwalk_arxiv_2604_23702.md) 混淆：后者是 G1 + PINN-GRF，项目名亦称 QuietWalk。

## 摘要级要点

- **问题：** 家用四足陪伴机器人（aibo）用户反馈 **脚步声过大**；既有腿足 RL 多优化鲁棒/能效，少以声学舒适为一等目标。
- **代理指标：** 生物力学：脚步声与 **足端接触动能** $\propto\|\mathbf{v}_{f,xyz}\|^2$ 相关；仿真难直接建模声场，故在 Isaac Gym 中最小化接触速度（辅以关节加速度、基座角加速度惩罚）。
- **三要素：**
  1. **可变 PD：** 策略对每关节输出目标位置 + **单一 gain scale** $x_i$，经 sigmoid 调制 $P_i=P^*+\alpha\sigma(x_i)$、$D_i=D^*+\beta\sigma(x_i)$（$\alpha=4,\beta=0.02,P^*=3,D^*=0.03$）；触地前阻尼、支撑相加硬。
  2. **足底开关接触：** 消费级 aibo 无力传感；二进制接触进观测，供策略选择何时加硬。
  3. **课程：** 先 **noisy walking**（接触速度惩罚 −5），速度跟踪回报和 >1.5 后切 **quiet walking**（接触速度惩罚 ×5→−25；关节/基座角加速度惩罚 ×2）；无课程则易收敛失败或学停走。
- **观测：** 关节位置/速度、上一动作（目标位置 + gain）、4 足接触、重力方向（含观测噪声，见表 I）。
- **DR：** 基座质量、速度扰动、外力/力矩、地形高度、摩擦等（表 III）；**加大 DR 可抬鲁棒、但常损安静度**。
- **真机声学：** aibo 头后麦克风 48 kHz；分析 20 Hz–20 kHz；Welch/FFT；麦克风距足约 10 cm。提出方法在多速度下 **平均声级低于** RL baseline 与 Sony normal / quiet 商用控制器。
- **仿真代理验证（表 IV，10 s 均值）：** 接触速度 0.417→0.123 m/s；关节加速度 114.3→76.7 rad/s²；基座角加速度 57.2→23.7 rad/s²。
- **鲁棒–安静权衡：** 0.5 m 斜坡 20 s 内；**最响 baseline 可上约 7°**，提出方法最安静但坡度能力最弱；消融：无课程、无接触传感、固定 PD、改摩擦/地形高度 DR 均可调权衡。
- **PD 时序：** 触地前 foreleg 各关节 gain 下降（阻尼），触地后肩 pitch 等回升以支撑；足端速度在触地前降低。

## 核心摘录（面向 wiki 编译）

### 1) 足端接触速度作为声学代理

- **链接：** §I；§III-C；§IV-B；Table IV
- **摘录要点：** 不直接最小化真机声压，而在仿真惩罚 $\|\boldsymbol{v}_{f,xyz}\|^2$ 等；真机更安静策略对应仿真更低惩罚项，支持 sim-to-real 代理有效。
- **对 wiki 的映射：**
  - [Learning Quiet Walking（aibo）实体页](../../wiki/entities/paper-learning-quiet-walking-aibo.md)
  - [Locomotion 奖励设计指南](../../wiki/queries/locomotion-reward-design-guide.md)

### 2) 可变 PD + 接触开关 + 两阶段课程

- **链接：** §III-A–D；Fig. 2；Fig. 5；Table II
- **摘录要点：** 三要素缺一不可——无课程难收敛；无接触传感则 quiet 阶段易学停走；固定 PD 可降噪但不如可变 PD。
- **对 wiki 的映射：**
  - [Learning Quiet Walking（aibo）实体页](../../wiki/entities/paper-learning-quiet-walking-aibo.md)
  - [Kp/Kd 设置 query](../../wiki/queries/legged-humanoid-rl-pd-gain-setting.md)
  - [可变阻抗接触 RL](../../wiki/entities/paper-variable-impedance-contact-rl.md)

### 3) 与后人形 QuietWalk（GRF/PINN）对照

- **链接：** 与 [quietwalk_arxiv_2604_23702.md](./quietwalk_arxiv_2604_23702.md) 对照
- **摘录要点：** 本文是 **运动学代理（接触速度）+ 消费级四足 aibo**；后人形 QuietWalk 是 **PINN 估计竖直 GRF + G1**。同名项目页 QuietWalk 指本文；勿混。
- **对 wiki 的映射：**
  - [QuietWalk 人形实体页](../../wiki/entities/paper-quietwalk-humanoid-locomotion.md)

## BibTeX

```bibtex
@inproceedings{watanabe2025quietwalking,
  title     = {Learning Quiet Walking for a Small Home Robot},
  author    = {Watanabe, Ryo and Miki, Takahiro and Shi, Fan and Kadokawa, Yuki
               and Bjelonic, Filip and Kawaharazuka, Kento and Cramariuc, Andrei
               and Hutter, Marco},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2025},
  note      = {arXiv:2502.10983}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-learning-quiet-walking-aibo.md`](../../wiki/entities/paper-learning-quiet-walking-aibo.md)
- 项目页：[`sources/sites/sony-quietwalk-github-io.md`](../sites/sony-quietwalk-github-io.md)
- 互链：[Locomotion](../../wiki/tasks/locomotion.md)、[四足机器人](../../wiki/entities/quadruped-robot.md)、[Locomotion 奖励设计指南](../../wiki/queries/locomotion-reward-design-guide.md)、[人形 QuietWalk（GRF）](../../wiki/entities/paper-quietwalk-humanoid-locomotion.md)、[Kp/Kd 设置](../../wiki/queries/legged-humanoid-rl-pd-gain-setting.md)、[Sim2Real](../../wiki/concepts/sim2real.md)、[Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)、[Disney Olaf](../../wiki/methods/disney-olaf-character-robot.md)

## 当前提炼状态

- [x] 摘要、三要素、奖励课程、声学/消融/PD 分析要点摘录
- [x] 项目页与 GitHub 开源边界核查
- [x] wiki 实体页与奖励指南 / 人形 QuietWalk 交叉链接规划
- [ ] 若作者后续发布训练代码再补 `sources/repos/` 与运行时序图

## 2026-11-07 通读复核补充（出自论文原文）

- 网络与频率：actor / critic 均为 3 层 × 128 单元 ELU 的 MLP，PPO；仿真 400 Hz、控制 100 Hz，Isaac Gym。
- 选择「目标关节位置 + PD 增益缩放」作动作而非直接力矩的两条理由（论文）：先前工作报告力矩作动作学习运动不高效；先前工作发现估计目标位置与 PD 增益更适合需要柔软接触的敏感接触任务。策略对每个关节只输出一个增益缩放 $x_i$，同时调节 P 与 D：$P_i=P^*+\alpha\sigma(x_i)$，$D_i=D^*+\beta\sigma(x_i)$，$\alpha=4.0,\ \beta=0.02,\ P^*=3.0,\ D^*=0.03$。
- 无课程时：直接用 quiet 阶段惩罚训练不收敛（aibo 学会不动）；只用 noisy 阶段参数的无课程版本只获得约一半的降噪；去掉开关接触传感器则训练收敛但 aibo 完全不走；固定 PD 增益的版本鲁棒性相当、但声级不如可变增益。
- 论文自述局限：策略鲁棒性低于 baseline（最大可爬坡度约 7° 的是最响的 baseline）；建议结合感知估计地面特性选择策略；展望为把「仿真里相关的代理量」用于其他难以仿真的真实指标（如以力矩近似电池寿命）。
