# ManiUnit 论文来源归档

## 元数据

- **题目：** ManiUnit: A Manipulation Skill Dataset and Benchmark for Long-Horizon Tasks
- **作者：** Guoting Wei, Dawei Yan, Xia Yuan, Gengming Zhang, Yelin He, Guodong Du, Jiaquan Ye, Heng Zhang, Xinming Wei, Xianbiao Qi, Chunxia Zhao, Haokui Zhang, Rong Xiao
- **单位：** 据 arXiv HTML：南京理工大学、西北工业大学、云天励飞（Intellifusion）
- **预印本：** https://arxiv.org/abs/2610.12089
- **HTML：** https://arxiv.org/html/2610.12089v1
- **版本：** v1，2026-10-08，cs.RO
- **项目页与代码：** 摘要与可读正文均未提供项目页、GitHub、Hugging Face 链接，也无发布声明（附录部分未读到）；截至 2026-10-12 未核实到公开数据或代码。

## 论文要解决的问题

长程移动操作需从一条自然语言指令出发跨房间执行一串操作技能。作者指出三类困难：同一指令下相似观测使技能选择有歧义；上一个技能成功后机器人状态偏离示范起始状态，拖累下一技能；任务级指标无法定位是哪一个技能失败，且早期失败使后续技能无从测试。

## 方法与训练信息

1. **数据来源与切分：** 基于 2025 BEHAVIOR Challenge 对 50 个 BEHAVIOR-1K 活动的示范，用 312 条手写抽取规则和 48 条合并规则从标注动作区间切出技能段，再用臂姿启发式（每臂 3 个关节、至少偏离 15 度、至少 5 帧）加夹爪事件保持来细化边界。
2. **规模：** 137,899 个片段、21 种技能类型、417 个子任务，平均片段时长 17.9 s；每个片段含头部及双腕 RGB、机器人状态、动作，并配有显式子任务指令。技能类型含 Pick Up、Place、Chop、Open/Close Door、Pour、Sweep、Turn On/Off Switch、Open/Close Lid、Spray、Insert、Release、Open/Close Drawer、Attach、Ignite、Press、Wipe、Hang。Lite 子集 10,588 个片段、7 类技能，取自 8 个成功率最高的活动。
3. **基准协议：** 测试实例 e=(s0, l, g)，s0 为恢复的起始状态，l 为子任务指令，g 为局部 BDDL 成功谓词。通过回放已存回合历史重建场景、恢复辅助抓取约束并先 25 步稳定再 60 步保持。条件有三：Original、Base Perturbation（基座平移）、Joint Perturbation（部分臂/躯干关节），扰动状态做碰撞和物体关系检查。Full 共 1,260 个实例（21 类 x 20 个锚点），Lite 186 个；局部目标须连续保持 10 帧且在该子任务平均示范时长的 3 倍内完成；每实例 Full 重复 3 次、Lite 重复 5 次，按技能类型等权平均。
4. **被评策略与训练：** StarVLA-PI、StarVLA-GR00T（Qwen3.5-2B 骨干 + 随机初始化动作头）、StarVLA-Cosmos2（Cosmos-Predict2 骨干）、π0.5（用 RLinf 从多机器人预训练权重微调）；均训练 5 个 epoch、跨技能类型混合训练。学习率、批大小、硬件、机器人本体与仿真器名称在可读正文中未给出。
5. **规划器协调：** 规划器读取机器人状态与视觉历史，决定何时交出控制权并给出哪条子任务指令；具体模型与架构在附录 D.3，本次未读到。

## 论文报告的结果

- Full（1,260）成功率 Original/Base/Joint/Overall：StarVLA-PI 60.1/51.2/26.5/45.9；StarVLA-GR00T 63.5/51.3/28.2/47.6。关节扰动相对 Original 下降约 55.9%（PI）和 55.6%（GR00T），摘要概括为整体约 56%。
- Lite（186）：StarVLA-PI 70.1/67.3/26.8/54.7；GR00T 77.8/64.0/29.6/57.1；Cosmos2 63.2/59.0/13.8/45.3；π0.5 80.0/65.8/53.0/66.3。相似总分掩盖技能差异，如 Full 上 Pick Up GR00T 54.4% 对 PI 44.4%；Lite 上 Place PI 81.9% 对 GR00T 70.5%、Pour GR00T 40.0% 对 PI 12.4%。
- Pick Up 对象变化（Lite）：π0.5 91.4 / 74.3（位置）/ 80.0（颜色）/ 80.0（干扰物）；PI 68.6 / 40.0 / 60.0 / 62.9；位置变化降幅最大。
- 技能策略对任务策略（两个活动 Turning On Radio、Picking Up Trash，75 个测试实例，均从同一 π0.5 权重训练 5 epoch）：任务策略 49.3%（Original 60.0、Base 56.0、Joint 32.0）；技能策略 78.7%（84.0、76.0、76.0）。Press radio 由 13.3% 到 80.0%，Pick up trash 由 60.0% 到 93.3%。
- 规划器协调后全任务成功率（Radio/Trash/总）：仅任务策略 4.0/4.0/4.0%，组合系统 24.0/12.0/18.0%；平均 Q-score 4.0/18.0/11.0 升至 24.0/26.0/25.0；每活动 10 回合 x 5 次重复。

## 复现与解释边界

- 全任务结论只覆盖两个长程活动；规划器的任务进度估计、子任务选择和切换时机仍是开放问题（作者自述）。
- 可读正文未给出数据/代码发布信息，附录被截断；视为“未核实”，不得写成已开源。
- 超参、硬件缺失，无法评估复现成本；片段由规则切分，边界质量依赖启发式。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-maniunit.md
- 相关实体：wiki/entities/behavior-1k.md、wiki/methods/star-vla.md
- 周报汇总：wiki/overview/frontier-manipulation-2026-10-05-10-09.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.12089>
- arXiv HTML：<https://arxiv.org/html/2610.12089v1>
