# viral-humanoid-visual-sim2real

> 来源归档（ingest）

- **标题：** VIRAL: Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation
- **类型：** paper
- **来源：** arXiv:2511.15200v1（CVPR 2026）
- **入库日期：** 2026-05-15
- **最后更新：** 2026-05-15
- **一句话说明：** 全仿真视觉 Sim2Real 人形 loco-manipulation：特权教师 RL（PPO + delta 命令驱动预训练 WBC）→ 大规模仿真中 RGB 学生（DAgger 与 BC 混合）→ 零样本真机，强调算力规模、视觉域随机化与灵巧手 / 相机 real-to-sim 对齐。

## 核心论文摘录（MVP）

### 1) Teacher–Student 总线与算力规模（Abstract / Sec.2 概述）

- **链接：** <https://arxiv.org/abs/2511.15200v1>
- **核心贡献：** 特权教师在全状态上学习长时域 loco-manipulation；学生在 **分块渲染的大规模仿真** 中用 **在线 DAgger 与行为克隆的混合** 模仿教师；作者发现将仿真扩展到 **数十块 GPU（学生阶段至多 64）** 对教师与学生训练的可复现性至关重要，低算力 regime 常失败。
- **对 wiki 的映射：**
  - [Privileged Training](../../wiki/concepts/privileged-training.md)
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [论文实体 VIRAL](../../wiki/entities/paper-viral-humanoid-visual-sim2real.md)

### 2) 教师：delta 动作空间、WBC 作 API、RSI（Sec.2.1）

- **链接：** <https://arxiv.org/abs/2511.15200v1>
- **核心贡献：** 教师输出对预训练 **HOMIE WBC** 的 **delta 命令**（线速度、yaw 角速度、臂与手指关节增量），而非从裸力矩学起；用约 **200 条仿真遥操作 demo** 做 **Reference State Initialization**，每回合从 demo buffer 采样中间状态 reset，显著改善长时域探索与成功率消融。
- **对 wiki 的映射：**
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md)
  - [GR00T-VisualSim2Real 实体](../../wiki/entities/gr00t-visual-sim2real.md)

### 3) 学生：DAgger–BC 混合与视觉骨干（Sec.2.2）

- **链接：** <https://arxiv.org/abs/2511.15200v1>
- **核心贡献：** 蒸馏损失在教师 rollout 与学生 rollout 诱导的观测分布上做 **α 混合**（默认 α=0.5）：纯 BC 收敛快但脆弱，引入学生 rollout（DAgger 侧）提高 Isaac→MuJoCo 与真机成功率；视觉骨干采用 **DINOv3** 等 SOTA 编码器，并消融历史窗口 / LSTM。
- **对 wiki 的映射：**
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)
  - [Domain Randomization](../../wiki/concepts/domain-randomization.md)

### 4) Sim2Real：灵巧手 SysID、相机外参、视觉随机化（Sec.2.3）

- **链接：** <https://arxiv.org/abs/2511.15200v1>
- **核心贡献：** 对 G1 三指 **高减速比灵巧手** 做 **SysID**（惯量、刚度、阻尼）对齐抓放轨迹；**相机外参** 做轻量 real-to-sim 标定并在训练中随机化；配合光照、材质、图像质量、延迟等 **大规模视觉域随机化**。
- **对 wiki 的映射：**
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [System Identification](../../wiki/concepts/system-identification.md)

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 关联 wiki 页面的参考来源段落已添加 ingest 链接
