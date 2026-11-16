# MPAIL2：Online World Modeling Enables Real-World Inverse Reinforcement Learning from Observation（arXiv:2602.24121）

> 来源归档（paper）

- **标题：** Online World Modeling Enables Real-World Inverse Reinforcement Learning from Observation（v1 原标题为 *Planning from Observation and Interaction*）
- **类型：** paper
- **作者：** Tyler Han, Bat Nemekhbold, Harine Ravichandiran, Siyang Shen, Sanghun Jung, Rohan Baijal, Richard Ebock, Kevin Huang, Byron Boots（v3 署名顺序；华盛顿大学）
- **arXiv：** <https://arxiv.org/abs/2602.24121>（v1 2026-02-27，v3 2026-10-18；arXiv 页注释 CoRL 2026）
- **PDF：** <https://arxiv.org/pdf/2602.24121v3>
- **项目页：** <https://uwrobotlearning.github.io/mpail2/>
- **代码：** https://github.com/UWRobotLearning/mpail2
- **入库日期：** 2026-11-12
- **一句话说明：** 用「编码器 + 潜空间动力学 + 对抗式逆强化学习奖励 + 价值 + 多步策略 + MPPI 规划」的在线世界模型，在真机上只靠任务观测（无奖励、无示范动作、无预训练、无仿真）从零学视觉操作。

## 核对范围

- **核对范围：** 下载 arXiv v3（2026-10-18，39 页）PDF，用 pdftotext 通读正文第 1–7 节（引言、相关工作、方法、实验、讨论、结论、局限）与附录 A（算法与架构，含 Algorithm 与网络结构）、B（实验设置与 Table 3/4 全部检查点结果）、C（超参数与基线实现）、D（补充学习曲线、消融、预测性/传播/鲁棒性分析、世界模型诊断）、E（补充讨论 E.1–E.5）；Table 1/2/3/4、Fig. 2 与 Fig. 3 另渲染页面图查看，其余附录图（Fig. 10–23）只读图注与正文描述，未逐图细看。另通读 v1（2026-02-27，仅 12 页正文 + 附录）作版本对照。核对项目页（含 BibTeX 标题与代码仓库链接）与 arXiv 摘要页的版本信息。未做：未运行代码、未读 GitHub 仓库内容、未看项目页视频；论文引用的外部文献（如 MPAIL、TD-MPC2、RLPD/SERL）未逐篇核对。
- **通读：** 已完成

## 版本差异（读哪一版很重要）

1. **v1（2026-02-27）**：标题 *Planning from Observation and Interaction*；真机任务 2 个（Block Push、Pick-and-Place）+ 迁移实验；Block Push 示范 10 条，Pick-and-Place 示范 10 条；摘要写「under an hour」；Fig. 5 图注写单次训练约 90 分钟（Block Push）与 70 分钟（Pick-and-Place）；无 Mug-on-Plate、无仅视频示范、无 [−O]（MPAIL）消融。
2. **v3（2026-10-18）**：标题即清单所用标题；新增 Mug-on-Plate、仅外部相机的 Block Push（Video）、[−O]（MPAIL 式）消融、Predictability / Propagation / Robustness 三条分析（附录 D.3/D.4）；摘要改为「under 40 minutes … pick-and-place … 82%」。本归档的数字以 v3 为准。

## 核心摘录（均出自 v3，括号为章节 / 表号）

1. **问题设定（§3）：** 只给任务观测轨迹 D，无动作无奖励；用学徒学习式目标 `arg max_r E_πE[r] − max_π E_π[r]` 推断奖励（式 1）。MDP 状态即机器人观测，转移、奖励、终止均未知。
2. **组件（§3，式 2–6）：** 编码器 e_ω、潜空间动力学 f_ψ（确定性）、奖励 r_θ(z,z')、价值 Q_ζ（Q 集成，附录 C.1 为 5 头）、多步策略 π_φ（输出 H 步计划，用于给规划器热启动并支撑离策略价值估计）、MPPI 规划器。编码器与动力学只用多步潜空间预测损失（停梯度目标）训练，不接收奖励 / 价值梯度。
3. **奖励（§3，式 3）：** 示范转移得分高于回放缓冲区转移得分，加 WGAN 式梯度惩罚；奖励离策略训练，使用全部历史交互。
4. **超参（附录 C.1，Table 5）：** H=7，λ=0.95，γ=0.99，MPPI：512 条 rollout、64 个精英、5 轮迭代、策略计划占比 0.05、温度 2.0；潜维 512；真机动作频率 10.28 ± 0.45 Hz（附录 C）；总参数 7114494（Fig. 6，两路 RGB 加本体感觉）。
5. **真机任务（Table 2，附录 B）：** 64×64 RGB（外部相机 + 腕部相机）+ 本体感觉；Franka：Block Push（10 条示范，1,451 转移）、Mug-on-Plate（15 条，1,802 转移）、Block Push（Video）（仅外部相机，10 条，914 转移）；Kinova Gen3：Pick-and-Place（10 条，1,025 转移）。示范由人用空间鼠标或键盘遥操作采集（B.2–B.4）。每任务 3 个种子、每检查点 50 次试验。
6. **仿真任务（Table 2）：** IsaacLab Franka，5 个种子；Block Push（state / image）示范 27 条，Pick-and-Place 30 条；附录另有 Gymnasium 的 Ant / Hopper / Humanoid（合成示范，Fig. 10）。
7. **真机主结果（Table 1 / Table 4，成功率 %，3 种子均值 ± 标准差）：** Block Push：MPAIL2 最佳 100±0 / 末检查点 61.8±13.5，BC 94.1±5.9，[−P] 64±17.7（最佳）。Pick-and-Place：MPAIL2 最佳 82.0±12.2 / 末 68.0±13.6，BC 12.0±5.8，[−P] 16.0±14.8。Mug-on-Plate：MPAIL2 最佳 54.9±14.7 / 末 35.3±17.3，BC 33.5±12.3，[−P] 3.9±2.8。[−PM]（DAC）与 RLPD 在所有真机检查点均为 0。
8. **迁移（§4 Q3，Fig. 3，Table 4）：** 先训 Block Push 再换成反方向（+y）推；迁移后最佳 90.2±4.7（60 个 episode，约 20 分钟），从零训 88.2±5.7（140 个 episode，约 50 分钟）。Pick-and-Place 反向迁移最佳 94±10.2 对从零 53±36.7（约省 20 分钟）。只迁移编码器与动力学（Dynamics Transfer）的效果「与完整迁移大致相同」（附录 E.4）。BC 与 [−P] 出现负迁移。
9. **仅视频示范（§4 Q3，附录 B.5）：** 仅外部相机视角，Block Push 最佳 62.7±10.0 / 末 58.8±14.4，BC 25.5±5.5（正文写 26%）。局限节指出这些视频示范里出现的是机器人自己（§7）。
10. **分析（§5，附录 D.3/D.4）：** 奖励与价值梯度回传进编码器（TD-MPC2 式）时，仿真 Block Push（image）5 个种子中 4 个出现潜表示塌缩，累计成功数由 244 降到 100（D.3）；规划器 vs 单用策略：训练早期规划器成功率高约 30 个百分点，后期单用策略反而在 5 个种子中的 4 个更好（Table 7，D.3）；策略在 MPPI 精英得分中占比始终低于 20%（附录 E.3，Fig. 21）；真机世界模型预测奖励的误差随滚动步数从 7% 升到 43%（k=1 到 7，D.4）。
11. **作者自述的局限（§7，附录 E.5）：** 操作员仍需指定任务观测、手动复位物体、事后挑选最佳检查点；对抗训练可能后期衰退；只做 64×64 图像下的单臂桌面操作，未测更高分辨率与更长时域；想象奖励随滚动深度退化；未学跨本体；是否可与大型预训练世界 / 策略 / 奖励模型结合未解决。

## 对 wiki 的映射

- [paper-mpail2-irlfo](../../wiki/entities/paper-mpail2-irlfo.md)
