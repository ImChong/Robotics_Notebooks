# 四足多技能：RL+模仿联合蒸馏与对抗任务选择 论文来源归档

## 元数据

- **题目：** Teaching a Robot Dog New Tricks: Diverse Quadruped Skills via Combined Reinforcement and Imitation Learning with Adversarial Task Selection
- **作者：** Lemon Foxmere, Anthony Furman, Yizheng Du, Oliver Chang, Leilani Gilpin, Steve McGuire
- **单位：** arXiv 页面未列机构；周报标注为加州大学圣克鲁兹分校（UCSC）研究团队（未在 arXiv 页面核实）
- **预印本：** https://arxiv.org/abs/2610.10601
- **HTML：** https://arxiv.org/html/2610.10601v1
- **版本：** v1，2026-10-06，cs.RO；12 页 5 图
- **补充视频：** https://youtu.be/V9yX04EBcFA
- **代码与项目页：** arXiv 页面未列项目页或代码仓库；截至 2026-10-12 未发现官方公开代码。

## 论文要解决的问题

让一个策略同时掌握行走、疾驰（gallop）、跳跃（hop）、挖掘（dig）、抬腿、身体高度/俯仰/偏航调整等差异很大的技能，并能把它们组合（如行走+俯仰）。单阶段 PPO 在这种多任务设定下样本效率低、任务间梯度冲突；先蒸馏后微调（Distill-then-Finetune, DtF）又会在 RL 微调阶段偏离教师动作风格。

## 方法与训练信息

1. **8 个单任务教师：** Walk、Gallop、Hop、Body Height、Body Tilt、Body Yaw、Leg Lift、Dig，各自用 PPO 训练；可用特权信息（如足端接触力）；每个教师约 18–34 项奖励项（平均 26）；MLP 256-128-64；训练完成后冻结。
2. **学生策略：** MLP 512-256-128；输入本体感知、14 维命令向量（速度、角速度、俯仰、偏航、抬腿标志、足端目标、步态选择、挖掘力、站姿高度）和 4 个步态时钟项（1.7 Hz 与 3.4 Hz 两组正余弦）。
3. **联合目标：** L(θ) = L_PPO(θ) − λ·L_BC(θ)；L_BC 为学生动作均值与教师动作的 MSE，数据由 DAgger 采集，只作用于动作均值而非标准差；复合任务无对应教师时不产生 BC 样本。
4. **两阶段调度：** ILP（Imitation Learning Primary，λ_IL=1.0，仅单任务，奖励只含稳定与平滑项，并加动作噪声防止标准差塌缩）；RLP（Reinforcement Learning Primary，λ_RL=0.25，单任务与复合任务，任务可在回合中切换，先冻结 actor 预训练 critic，奖励面向跟踪，全部 22 任务共 32 项奖励项）。
5. **对抗任务选择：** 每个任务激活一组误差通道，复合任务取并集；通道归一化 CNorm(e)=tanh(a·v_e/τ_e)，a=atanh(0.2)，τ_e 为教师在该通道的最终误差；任务亏损 Ē_i 为激活通道均值；采样概率 p_i = κ·Ē_i^β / Σ_j Ē_j^β + (1−κ)/|𝒴|，κ=0.85，β=1.5，全程恒定。
6. **22 个训练任务：** 8 个单任务 + 14 个两任务复合（如 Walk+Tilt、Gallop+Height、Dig+Yaw 等）；三任务、四任务组合不参与训练，只用于泛化测试。
7. **训练设置：** NVIDIA Isaac Lab，平地，各方法使用相同域随机化；每种方法 RL 段 12.29 亿次转移；学习率 5e-5 线性衰减到 5e-6（前 80%）；初始动作标准差本文方法与 DtF 微调为 0.5，PPO 为 1.0。并行环境数、墙钟时间、GPU 型号原文未报告。

## 论文报告的结果

- **对比对象：** Teacher Router（按任务路由到对应教师，无法做复合任务）、单阶段 PPO、DtF（沿用 Rudin 等的做法，使用相同教师/架构）；3 个种子。
- **测试 1：** 22×22=484 个 A→B 任务序列，每序列 40 次试验（每模型每种子 19,360 次）；A 保持 10 s，B 保持 5 s；带观测噪声和 0–300 N、0.1–0.2 s 的外力扰动。
- **单任务（Table VI）：** 本文方法生存率 99.14%、显式任务分 0.608、步态误差 0.147、站姿分 0.927；DtF 96.99% / 0.474 / 0.383 / 0.895；PPO 99.72% / 0.440 / 0.478 / 0.600；Teacher Router 80.34% / 0.500 / 0.118 / 0.809。DtF 的隐式分（0.741）略高于本文（0.716）。
- **复合任务：** 本文 99.23% / 显式 0.605 / 步态误差 0.158；DtF 96.96% / 0.496 / 0.384；PPO 99.42% / 0.471 / 0.476。
- **教师动作 L2 误差（单任务）：** 本文在 8 个任务上 0.64–1.42，DtF 1.45–4.20，PPO 2.63–7.09。DtF 在 Dig 与 Hop 的任务分高于本文（Hop 0.676 vs 0.543），但动作误差大得多。
- **任务切换（A→B）：** 本文生存率 99.05%、显式分 0.558；DtF 97.66% / 0.395；Router 80.82% / 0.394；PPO 99.81% / 0.448。
- **任务规模：** 9 任务训练时本文与 DtF 接近（显式 0.694 vs 0.679）；扩到 22 任务后在 22 任务集上本文显式 0.608，DtF 0.474。
- **OOD 组合（7 个命令，各 60 次）：** 本文生存率 96.35%–100%，DtF 在 Tilt+Height+Gallop 上仅 83.02%（本文 99.44%）；个别组合（Dig+Height、Dig+Tilt+Yaw）DtF 隐式分更高。
- **消融（Table X）：** 本文对初始动作标准差更稳健（1.0 STD 下显式 0.595），对初始学习率更敏感（1e-4 时显式降至 0.537，站姿分 0.706）。
- **真机（Unitree B1）：** 仅本体感知；机载 Raspberry Pi 4B 100 Hz 推理；平坦混凝土、草地、松土；展示钻低矮空间、开门、推电池、挖掘、疾驰、跳跃、俯仰、抬腿按按钮。真机结果为定性展示，无量化指标。

## 复现与解释边界

- 无公开代码与权重；并行环境数、训练时长、完整观测与奖励权重原文未给出，难以复现。
- 仅 Unitree B1 一种本体、MLP 架构、平地训练、纯本体感知；论文自述局限包括对初始学习率敏感。
- 3 个种子，只报均值与标准差，无显著性检验；0.5 分的“教师水平”是归一化选择，不是实测上限。
- 真机无试验次数与失败率。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-quadruped-skills-adversarial-task-selection.md
- 任务入口：wiki/tasks/locomotion.md
- 汇总页：wiki/overview/frontier-humanoid-quadruped-2026-10-05-10-09.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.10601>
- arXiv HTML 全文：<https://arxiv.org/html/2610.10601v1>
- 补充视频：<https://youtu.be/V9yX04EBcFA>
