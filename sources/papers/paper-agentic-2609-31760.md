# What Stops Recursive Self-Improvement in Robotics? Lessons from 123 Rounds of Agentic Skill Discovery

- **作者：** Jiaming Wang（正文署名 Wang Jiaming）
- **机构：** National University of Singapore（新加坡国立大学）
- **版本 / 载体：** arXiv:2609.31760v1 [cs.RO]（2026-09-23 提交，Technical report，20 页，7 图，8 表，含附录 A–E；CC BY 4.0）
- **原文：** [arXiv HTML 全文](https://arxiv.org/html/2609.31760)、[arXiv 摘要页](https://arxiv.org/abs/2609.31760)
- **项目页 / 代码：** 未确认开源代码（arXiv:2609.31760v1 全文及当前公开索引未列出官方 GitHub 仓库或独立项目页链接）
- **核对范围：** 通读 arXiv HTML 全文第 1–11 节（Abstract、Introduction、System overview、123 轮运行分析、三条件诊断模型、Lessons 1–3、12 条设计建议 D1–D12、Related work、Limitations、Conclusion）、Tables 1–8 及 Appendices A–E（含逐轮记录表 Table 4、主文测量汇总 Table 5、阶段统计 Table 6、外部模型调用统计 Table 7 与第 123 轮终止状态 Table 8）
- **通读：** 已完成

## 原文核对笔记

- **实验定位与问题设定：** 本文是一篇长周期实证技术报告，围绕“编程智能体能否像改进软件一样，在无人类编写机器人代码的前提下通过失败反馈递归自改进机器人操作能力”展开，重点考察三项问题：（1）能否仅从失败中自主发现缺失能力；（2）跨轮次的局部改进能否累积为任务端到端成功；（3）若不能累积，瓶颈究竟在于智能体自身还是外围系统。
- **五层软件与四类迭代主体（§2, Fig. 2, Table 1）：**
  - 系统包含五个部分：**任务程序（Task program）**、**技能库（Skill library）**、**外部模型（Providers）**、**路由规则（Routing）**与**评测/执行支架（Harness）**。其中前四者分别由**编排智能体（Orchestrator）**、**提案智能体（Proposer，多模态）**、**研究智能体（Research agent）**及日志拟合自动更新；**仅 Harness 由人类维护**，且严格禁止人类编写任何技能、任务程序或模型封装代码。
  - 所有智能体均运行 OpenAI GPT 模型于沙盒化 Codex CLI 容器中：第 1–75 轮为 `gpt-5.6-sol`（最高推理强度），第 76–84 轮为 `gpt-6-astra`（低强度），第 85–123 轮为 `gpt-6-astra`（中等强度）。机器人运行时本身不调用 GPT，仅通过头/腕 RGB-D 相机与关节读数调用技能及其后端 Provider。
  - 任务基于 RoboCasa 中的 PandaOmron 移动操作机器人，核心目标任务为约 15 步的 `Fridge`（将台面上的芥末与蛋黄酱放入冰箱顶层搁架，若顶层已有物品则先腾移到其他层；第 60 轮加入 `Cabinet`，第 66 轮加入 `Sink`，另有仅用于门禁测试的保留任务 `cabinet to counter`）。
- **123 轮三阶段演化结果（§3, Tables 4/5/6/7/8）：**
  - 总计完成 `972 / 993` 次仿真 rollout，`847` 次提案尝试，`57` 次技能库变更通过全部门禁（其中 `51` 次位于感知层），`86` 次外部模型注册（覆盖 `20` 个独立模型身份），人类共执行 `113` 次编号 Harness 修复。
  - **正面发现：** 智能体并非瓶颈，且具备自主发现缺失能力与精准诊断的能力——第 2 轮修复可见性归因后，第 5 轮自主申请 `target_free_active_view_base_placement` 并修复起点占用 Bug，第 16 轮完全消除“目标不在视野”失败；累计 8 个早期外部模型在部署中获得数十至上千次调用（如 `relational_rgbd_localizer` 调用 `1,863` 次）。
  - **核心失败：** `Fridge` `0/486`、`Cabinet` `0/256`、`Sink` `14/230`（其中 `9` 次伴随中间步骤失败、属误放跌落）。第 107 轮之后注册的 `12` 个外部模型（含第 114–121 轮注册的 `11` 个视觉-语言区域定位模型）在后续 rollout 中的调用次数全部为 `0`。第 123 轮结束时 `Fridge` 仍有 `3/4` 种子停在第 0 步 `fixture_region_localized: wrong_region`。
- **三大结构性阻塞与三条件框架（§§4–7）：**
  - 论文用任务链概率模型 $S(\Phi) = \prod_{j=1}^m p_j(\Phi)$ 提出有效自改进的三项前提：**可表征性（Representability）**、**可表达性（Expressibility）**、**可分辨性（Resolvability）**。
  - **教训 1（感知串联不理解关系，§5）：** 概念分割（SAM 3）与指代表达分割（Florence-2）加深度提升均不表征“最上层搁架”等序数/空间关系，导致代码智能体出现**语义位移（Semantic displacement）**——用 `3,056` 行代码、`33` 个几何阈值（组合 `5,032` 行、`48` 个阈值的物体定位器）去过拟合 2 个失败样本。人工在 4 帧离线测试中验证 `Qwen2.5-VL 7B` 可 4/4 正确定位（约 1 s/次），表明出路在于将“切换模型类别”纳入智能体可执行动作。
  - **教训 2（技能链将学习锁死在第一步，§6）：** 由 $\Delta S = \pi_j \Delta p_j \prod_{i>j} p_i$，上游不可靠同时衰减下游修复的收益与可观测证据样本量（**双重衰减 / Frontier lock**）：`Fridge` 57% 失败在第一步、仅 2.7% 通过四步、0.9% 通过五步；第 118 轮在未见/保留场景复现胜出的抓取改进，在随后 60 次 rollout 中仍贡献 0 次端到端成功。
  - **教训 3（Harness 决定学习方向，§7）：** 包括**评测器捕获**（0.85 m 球形可达判定导致 276 次成功卡在 0.849–0.850 m；缺失躯干升降说明遭投诉 34 次；底盘标称速度 0.0016 m/step 与实测 0.037 m/step 相差逾 20 倍）、**不可分辨筛选**（16/36 次Fresh差异重跑不可复现；22 次饱和基线零胜零负晋级）、**证据不动点**（第 90–102 轮结果逐轮完全一致）、**记忆污染**（88 次含模型请求提案被当代码门禁并记为失败，导致智能体 15 次拒绝重提）、**有能力未调用**与**措辞高度敏感**（改 1 段提示词使编辑率从 0/8 升至 11/12）。
- **12 条可证伪设计建议（§8, Table 3）：** D1–D3 恢复可表征性（按能力归因、实测具身接口、反馈执行轨迹），D4–D6 恢复可表达性（VLM 先推理后几何校验、语义位移触发跨类模型升级、允许编排层门控修改技能链），D7–D9 恢复可分辨性（越过前沿采样状态、统计分辨率感知门控、分离回归哨兵与探索预算），D10–D12 闭合递归（分型记忆、模型准入绑定集成义务、将 Harness 自身纳入档案回放门控迭代）。
- 对 wiki 的映射：[What Stops Recursive Self-Improvement in Robotics? Lessons from 123 Rounds of Agentic Skill Discovery](../../wiki/entities/paper-agentic-2609-31760.md)
