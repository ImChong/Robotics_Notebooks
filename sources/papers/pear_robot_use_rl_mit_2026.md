# Training Reasoning Models in Simulation to Natively Use Real Robots（PEAR / RURL）

> 来源归档：论文全文核对（2026-11-15）

- **标题：** Training Reasoning Models in Simulation to Natively Use Real Robots（在仿真中训练推理模型，使其原生操作真实机器人）
- **类型：** paper（预印本）
- **论文 PDF：** <https://mit-han-lab.github.io/pear-blog/paper.pdf>
- **项目博客：** <https://mit-han-lab.github.io/pear-blog/>
- **代码：** <https://github.com/mit-han-lab/pear>
- **模型权重：** <https://huggingface.co/collections/mit-han-lab/pear>
- **作者：** Jiaming Tang、Zhuoyang Zhang、Junxian Guo、Fangzhou Zhao、Yilong Zhao、Zekai Li、Xingyang Li、Yuyang Zhao、Song Bian、Zhijian Liu、Song Han
- **机构：** MIT、UC Berkeley、UCSD、NVIDIA、Inco AI
- **入库日期：** 2026-11-15
- **核对范围：** 21 页预印本：摘要、引言、相关工作、§3 RURL/PEAR 算法和课程、§4 各项实验、§5 局限、§6 结论、附录 A–H（训练设定、分项结果、扰动、RoboDojo、通用能力、真机逐次试验和提示词模板）；对照官方博客的示意图、代码库 README 与训练/部署入口。
- **通读：** 已完成
- **一句话说明：** 开放 27B 推理模型通过仿真中的多轮机器人操作强化学习，无示范数据、无额外感知/规划 harness，学习观察→推理→末端姿态动作，并迁移到未见过的真实双臂机器人。

---

## 原文摘录与 wiki 映射

### 1. Robot-Use Agent（RUA）的接口

- **论文机制（§3.1、附录 H）：** 每回合输入任务指令、摄像头帧和机器人各部件状态；模型产生自然语言推理以及 JSON 形式的各部件目标**相对变化量**（长度 cm、角度 rad）；逆运动学与控制器执行，再返回新观测。模型不直接产生关节力矩，也不依赖外部抓取检测器、运动规划器或技能库。
- **核心边界：** 本文的 RUA 与示范轨迹训练的动作块 VLA 不同；物理执行仍依赖每个机器人本地 IK/控制器，不能把“无 harness”误读为“不需要控制栈”。
- **对 wiki 的映射：** [PEAR / RURL](../../wiki/entities/paper-pear-robot-use-rl.md) 的接口、流程与工程实践。

### 2. 仿真 RL 与课程训练

- **论文机制（§3.2–3.3、附录 A）：** simulator 按任务完成与否给 episode 级二值奖励；同任务同初态分组的 GRPO 归一化优势赋给各回合推理及动作 token；丢弃全成功/全失败组；完全异步训练用 score centering 校正采样策略与更新策略差异。PEAR 在 veRL、vLLM/Ray、LMCache 之上实施长上下文多回合 rollout。
- **课程：** 阶段 1 为 LIBERO Goal/Spatial/Object，阶段 2 加入 LIBERO-Long，阶段 3 再加 RoboDojo 的 general pickup / align blocks；论文训练分别约 148、77、115 步。
- **资源边界：** 官方 quickstart 举例 32 个 H100（每节点 8 GPU）或 GB300（每节点 4 GPU）计算节点，不能误写成普通单机即可重训；单卡部署与分布式训练是不同硬件条件。
- **对 wiki 的映射：** [PEAR / RURL](../../wiki/entities/paper-pear-robot-use-rl.md) 的训练机制与源码时序图。

### 3. 作者报告的实验结果（非本次独立复现）

| 设定 | 作者报告 | 注意口径 |
|------|----------|----------|
| LIBERO（四套汇总） | 基座 1.6% → RL 74.4% | xhigh 推理努力；π₀.₅ 在同基准 96.9% |
| LIBERO-Pro 新任务、旧场景 | PEAR 57.4%，π₀.₅ 0.75% | 特定指令重定向扰动，不是整个 LIBERO-Pro 总分 |
| LIBERO-Long（未训练该套时） | 0.4% → 39.0% | 先在另外三套 LIBERO 学会操作，零样本迁移 |
| RoboLab-120 | 最终模型 37.3%，LIBERO-only 22.1%，基座 5.4% | 120 任务；不能与 LIBERO 成功率混算 |
| 真实双臂 YAM | 基座 0/30、LIBERO-only 7/30、最终模型 18/30 = 60% | 10 次对齐积木 + 20 次抓取，共 30 次；非新硬件大样本测评 |
| 通用能力 | 九项推理/代码/视觉基准均值 81.64→81.49 | 各项配对 95% 区间均覆盖零；不代表绝对没有变化 |

- **对 wiki 的映射：** [PEAR / RURL](../../wiki/entities/paper-pear-robot-use-rl.md) 的评测与结论。

### 4. 局限与开源状态

- **论文 §5：** 任务/仿真覆盖有限；每一步先长链推理，平均约 4.3 秒，慢于连续低层控制；依赖详尽本体/坐标系 prompt；仅输出逐步位姿而非可靠动作块，IK/控制器质量决定执行误差；NVFP4 量化影响精度，因此论文各评测用 BF16。
- **真机故障（附录 G）：** 最终模型 12 次失败中，包括步数耗尽、积木未对齐、碰倒对象、网球滚走；气泵与刷子均未成功抓取。
- **开源核对（2026-11-15）：** 官方项目页链接 Apache-2.0 GitHub 仓库；提供三阶段训练、评测、vLLM server、YAM/SO-101 station 及 HF checkpoint，部分仿真数据/硬件仍需独立配置。
- **对 wiki 的映射：** [PEAR / RURL](../../wiki/entities/paper-pear-robot-use-rl.md) 的局限、部署安全与原始材料链接。
