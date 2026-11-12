# TacCoRL（arXiv:2606.11743）

> 来源归档（paper，全文通读版）

- **标题：** TacCoRL: Integrating Tactile Feedback into VLA via Simulation
- **类型：** paper
- **作者 / 机构：** Siyu Ma、Yuqi Liang、Chang Yu（共同一作，Chang Yu 为 project lead）、Yunuo Chen、Hao Su、Yixin Zhu、Yin Yang、Chenfanfu Jiang；UCLA、UCSD、UESTC、北京大学、犹他大学（首页署名）
- **arXiv：** <https://arxiv.org/abs/2606.11743>（v1，2026-06-10，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2606.11743v1>
- **项目页：** <https://tac-corl.github.io/>（论文给出；代码仓库未在论文中给出，本次未找到公开仓库）
- **入库日期：** 2026-11-12
- **一句话说明：** 在 π0.5 式 VLA 上加触觉 token 与接触门控，用「仿真+真机联合训练 → 仿真稀疏奖励 PPO（带真机监督锚）」两阶段后训练，四个双臂接触密集任务真机平均成功率 72.5%（对照 50.0%）。

- **核对范围：** 通读 arXiv v1 PDF 正文 §1-§6（含 Fig.1-7、Table 1-2）与补充材料 A-E（机器人配置、SysID、触觉与相机对齐、基线失败分析、Table 3-4 训练参数、Table 5 域随机化）；参考文献表仅扫读未逐条核对；未看项目页视频，未读任何代码。
- **通读：** 已完成

## 核心摘录（均为论文原文内容）

1. **问题与主张（Abstract、§1）：** 视觉常缺失接触局部状态；提出不做大规模触觉预训练、不做真机在线 RL，而用「real-aligned simulator」作闭环训练环境。关键思想是「学习接触读数如何调制近失败状态下的动作响应」（原文称这类状态在演示中稀少、真机采集有风险）。
2. **策略结构（§3.2，式 1-5）：** π0.5 式 VLA；触觉窗口 L 帧经编码器 Eτ 与投影 Wτ 得到 M 个触觉 token；二值接触门控（任一时刻有 ≥m 个 taxel 超过阈值 λf 才放行，否则从注意力移除触觉 token）；两条注入路径：对视觉-语言-本体 token 做 cross-attention，以及作为 action expert 的条件 token。
3. **阶段 1 联合训练（§3.3，式 6-7）：** 仿真遥操作 D_sim^teleop 经 MimicGen 扩增为 D_sim^Mimic，按与评测相同的成功判据过滤；损失 = α·仿真 flow-matching + (1−α)·真机 flow-matching。
4. **阶段 2 后训练（§3.4，式 8）：** 标准 PPO，任务级成功/失败判据作稀疏奖励，外加 β 加权的真机行为克隆项作为锚；部署时丢弃 critic、奖励与特权状态。实现沿用 RLinf 与 πRL 的 flow-policy RL（附录 D）。
5. **对齐（§3.1、§4.1、附录 B）：** 控制器 SysID（SPSA 优化 Kp、Kd、Tref，Fig.9）；触觉用弹簧阻尼法向读数模型（式 10），只对齐法向通道、不重建剪切力；相机外参离线标定。对齐质量只以定性图与一个直方图展示（Fig.4、10、11），无统一定量误差指标。
6. **设置（§4.2、附录 A、D）：** 双 AgileX PiPER 6-DoF 臂，每爪一块 FlexiTac-V2（32×12 taxel）；每任务 50 条真机演示 + 20 条仿真遥操作（MimicGen 扩至 200）；仿真评测 128 并行环境，真机每任务 20 次试验。
7. **仿真结果（§4.3，Table 1）：** 直接从基座 VLA 做 RL 全任务 0.00；联合训练后均值：视觉 33.0%、视觉触觉 40.5%（按四任务表值平均）；RL 后视觉 60.5%、视觉触觉 78.5%。
8. **真机结果（§4.4，Table 2）：** RL 后平均 50.0% → 72.5%（视觉 vs 视觉触觉）；Do Puzzle 视觉触觉：真机微调 15%、联合训练 25%、RL 45%。
9. **消融（§4.5，Fig.7，仅 Assembly #2）：** α 从 0.95 降到 0.5，最佳仿真成功率 42.9% → 70.3%，真机 40% → 45%；α=0（无仿真演示）仿真 14.1%、真机 25%；β=0.1 与 1.0 把真机从 45% 提到 80%；β=5.0 监督损失最低但仿真与真机成功率都下降。
10. **局限（§6，作者自述）：** 仍需少量真机触觉演示作锚；任务级数字孪生需人工；只覆盖刚体 / 近刚体接触，可变形物体与流体需额外建模。

## 对 wiki 的映射

- [paper-sa-2606-11743-taccorl-integrating-tactile-feedback-into-vla-vi](../../wiki/entities/paper-sa-2606-11743-taccorl-integrating-tactile-feedback-into-vla-vi.md)
- 旧的清单索引级摘录仍保留：[sun_awesome_touch_2606_11743_taccorl-integrating-tactile-feedback-int](./sun_awesome_touch_2606_11743_taccorl-integrating-tactile-feedback-int.md)
