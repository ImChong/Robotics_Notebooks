# CHOREO: Every Humanoid Skill as a Trajectory

## 元数据

- **arXiv：** <https://arxiv.org/abs/2609.22274>
- **PDF：** <https://arxiv.org/pdf/2609.22274>
- **HTML：** <https://arxiv.org/html/2609.22274>
- **DOI：** <https://doi.org/10.48550/arXiv.2609.22274>
- **提交：** 2026-09（arXiv v1）
- **机构：** 中国海洋大学（OUC）；中国科学院大学（UCAS）；中国科学院自动化研究所（CASIA）
- **平台：** Unitree G1 · MuJoCo · 低层跟踪 **GMT**（50 Hz）
- **项目页 / 代码（2026-09-27 核查）：** arXiv 摘要页 **未列** GitHub 或独立项目页；论文正文未承诺 release 条款 → **截至入库日未开源**

## 核心摘录

1. **问题：** RL、模仿学习与生成模型各自产出人形技能，但表示、控制器接口与执行假设互不兼容；「再训一个更大通用策略」每加技能都要改训练分布，扩展性差。
2. **观察：** 无论技能来源如何，**可执行机器人轨迹** 是共同行为输出 — 统一 **行为层** 而非统一 **策略架构**。
3. **SkillMotion：** 规范轨迹 \(\boldsymbol{\tau}\)（关节/根/足接触 @ 30 Hz）+ 语义描述 \(\mathbf{z}\) + 入/出边界窗 \(\mathbf{b}^{\mathrm{in/out}}\) + 执行描述 \(\mathbf{e}\)（机体、Tracker、校验证据）；离线 Source Adapter 注册进库 \(\mathcal{L}\)。
4. **在线组合：** LLM 任务规划器输出技能 **语义序列**（非轨迹）；运行时按边界 mismatch 得分 \(d_i\) 选 **直接拼接**、**局部 quintic seam**（Hermite 边界）或 **预验证站立 bridge** + 双 seam；Tracker 适配器把合成参考喂给冻结 \(\pi_k\)（实验主用 GMT）。
5. **异源入库：** 运动库 **2969** 条、RL rollout **50**、扩散生成 **50** — 全部 **3069/3069** 通过 import；闭环 rollout 子集 **353/356**（**99.2%**）成功执行。
6. **长程基准（130 固定序列 · 552 边界 · 冻结 GMT · seed 20260726）：** 整体序列成功率 **95.4%**（124/130）；2/3-action **100%**；5-action **91.7%**；8-action **93.8%**；切换成功率 **96.7%**；fall rate **4.6%**；\(\Delta q\) **0.0170 rad**。最强 baseline（Motion Matching）8-action 仅 **31.2%**。
7. **多 Tracker：** 同一 30 条准入轨迹经适配器在 GMT / OpenTrack / TWIST2 / HoloMotion / SONIC / H-ACT 上评测 — GMT、HoloMotion、SONIC、H-ACT **30/30**；TWIST2 **26/30**；OpenTrack **14/30**（测 **接口兼容性**，非系统排名）。
8. **消融：** 去掉 **state–entry compatibility** 匹配时 8-action 成功率 **75.0%**、fall **16.2%** — 长序列下「当前状态对齐技能入口」最关键。

## 对 wiki 的映射

- 实体页：[CHOREO（SkillMotion 异源技能组合）](../../wiki/entities/paper-choreo.md)
- 交叉：[GMT](../../wiki/entities/paper-gmt.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[Switch 技能切换](../../wiki/methods/switch-framework.md)、[HumanoidArena 分层 GMT](../../wiki/entities/paper-humanoidarena.md)
