# doorman_opening_sim2real_arxiv_2512_01061

> 来源归档（ingest）

- **标题：** Opening the Sim-to-Real Door for Humanoid Pixel-to-Action Policy Transfer（DoorMan）
- **类型：** paper
- **来源：** arXiv:2512.01061（HTML v1；CVPR 2026）
- **入库日期：** 2026-05-17
- **一句话说明：** 以人形 **RGB-only** 开门为 loco-manipulation 压力测试：Isaac Lab 中 **特权教师 PPO + 分阶段重置探索** → **DAgger 视觉学生** → **GRPO 微调** 缓解部分可观测性；大规模 **物理 + 外观** 程序化门与域随机化，全仿真训练后真机零样本泛化，并在相同 WBC 栈上相对人类遥操作提升成功率与完成时间。

## 核心论文摘录（MVP）

### 1) 问题设定与总贡献（Abstract / Introduction）

- **链接：** <https://arxiv.org/abs/2512.01061>
- **核心贡献：** 将 **铰接物体（门）交互** 作为高难 loco-manipulation 代表；提出 **teacher–student–bootstrap**：分阶段奖励的 **特权教师 RL**、**DAgger** 将教师蒸馏到 **RGB+本体** 学生、再用 **GRPO** 在 **二值成功 + 简单正则** 下微调以改善长时域闭环与部分可观测；声称首个仅靠 **纯 RGB** 在真机上完成 **多样化铰接 loco-manipulation** 的人形 Sim2Real 策略（Unitree G1，底层 WBC 来自预训练 **HOMIE** 类栈）。
- **对 wiki 的映射：**
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [Privileged Training](../../wiki/concepts/privileged-training.md)
  - [Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md)
  - [DoorMan 论文实体](../../wiki/entities/paper-doorman-opening-sim2real-door.md)

### 2) 教师观测与动作接口；分阶段重置探索（Sec.2.1–2.2）

- **链接：** <https://arxiv.org/html/2512.01061v1#S2>
- **核心贡献：** 教师除本体外可访问 **根–门、左右手–把手** 等 **特权几何**、**手部接触 wrench**、根线速度等；动作为 **目标关节角**（论文给出 G1 上 **高维关节目标** 与 **50 Hz** 推理需求，依托预训练 WBC 跟踪）。长时域任务用 **阶段化奖励**；提出 **staged-reset**：进入新阶段时把仿真快照写入滚动缓冲（论文用 **最近 100 步**），reset 时以混合律从各阶段分布初始化，以抬高后期状态占用、避免早期「错误抓把手 → 高惩罚 → 不敢再进阶段」的探索塌陷。
- **对 wiki 的映射：**
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md)
  - [Curriculum Learning](../../wiki/concepts/curriculum-learning.md)
  - [GR00T-VisualSim2Real 实体](../../wiki/entities/gr00t-visual-sim2real.md)

### 3) 学生架构与 GRPO 微调（Sec.2.1 / 2.3）

- **链接：** <https://arxiv.org/html/2512.01061v1#S2>
- **核心贡献：** 学生用 **ResNet 视觉编码**（与策略 **联合微调**）+ 本体，经 **两层 LSTM（512）** 与 **三层 MLP** 输出关节目标；蒸馏为 **DAgger**。GRPO 用 **组内相对优势**（无单独 value 网络）做 **clip PPO 式** 策略更新；微调期主要 **任务成功二值信号** + 关节速度/加速度/动作变化率惩罚；经验上学生可学会 **保持操作区域在视野内** 等教师未显式演示的补偿行为。
- **对 wiki 的映射：**
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)
  - [Domain Randomization](../../wiki/concepts/domain-randomization.md)

### 4) 大规模仿真随机化与真机对比（Sec.2.4 / Sec.3）

- **链接：** <https://arxiv.org/html/2512.01061v1#S2.F4>
- **核心贡献：** Isaac Lab **程序化门**：多 **门型/尺寸/铰链阻尼/把手位/阻力矩/闩锁动力学** 等物理随机化；**PBR 材质 + 大量 dome light**、**相机内外参微扰**、RTX 实时渲染与运动模糊等；消融表明 **无纹理/无穹顶光** 时成功率可跌至 **个位数～20%**，全量纹理 + 穹顶光在未见门上约 **81–86%** 子任务成功率。真机与 **同 WBC 栈** 下 **VR 遥操作** 对比：报告约 **83% vs 专家 80% / 非专家 60%** 成功率，完成时间快 **约 23–32%**。
- **对 wiki 的映射：**
  - [Isaac Gym / Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)

## 其他公开资料（非 PDF 正文）

- **项目页（演示与管线叙述）：** <https://doorman-humanoid.github.io/> — 归档见 [sources/sites/doorman-humanoid-github-io.md](../sites/doorman-humanoid-github-io.md)
- **代码与复现入口：** <https://github.com/NVlabs/GR00T-VisualSim2Real> — 归档见 [sources/repos/gr00t_visual_sim2real.md](../repos/gr00t_visual_sim2real.md)

## 当前提炼状态

- [x] 论文摘要与核心方法摘录
- [x] wiki 页面映射与姊妹资料互链
