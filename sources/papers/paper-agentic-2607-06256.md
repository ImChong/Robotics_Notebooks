# Diagnosing Semantic Handoff Failures in Agent-Orchestrated Vision-Language-Action Skill Composition

- **作者：** Ke Rui\*, Yushen Zuo\*, Jiawei Wang, Haoran Jia, Jinming Ma, Weitao Zhou, Minglei Li（\* Equal contribution；通信作者：`{wangjiawei, liminglei}@simpleai.tech`）
- **机构：** 简易人工智能（Simple AI / SimpleAI）
- **版本 / 载体：** arXiv:2607.06256v2 [cs.RO]（2026-07-07 提交 v1，2026-07-15 修订 v2，arXiv HTML 全文，CC BY 4.0）
- **原文：** <https://arxiv.org/abs/2607.06256> / <https://arxiv.org/html/2607.06256>
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27 核查：arXiv:2607.06256v2 正文及元数据未公布独立项目页或官方 GitHub 仓库链接；抓取元信息中的 `huggingface.co` 通用链接非本项目仓库）
- **核对范围：** 通读 `arXiv:2607.06256v2` HTML 全文，涵盖 Abstract、Sec. I（Introduction）、Sec. II（Semantic Handoff Problem）、Sec. III（Semantic Execution Harness）、Sec. IV（Diagnostic Evaluation）、Sec. V（Discussion）、Sec. VI（Related Work）、Sec. VII（Conclusion）、Appendix（训练/路由细节、Table IV–IX）及 Fig. 1–2 图注。
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [Diagnosing Semantic Handoff Failures 论文实体](../../wiki/entities/paper-agentic-2607-06256.md)

### 1. 问题定义与研究动机（Sec. I–II）
- 长时程家居任务需要按语义顺序组合多个语言条件技能，但相邻技能之间的边界通常未显式刻画。
- **Semantic Handoff Problem（语义交接问题）**：前一个技能 $s_t$ 在满足自身局部后置条件 $\phi_t$ 后，仍可能将机器人位姿、物体状态或相机视角停留在下一技能 $s_{t+1}$ 无法可靠起步的状态。例如 `move_to(can)` 后目标物体在远处可见但未进入机械臂可达抓取范围；或 `open_door` 打开柜门后把下一目标推离当前视野。
- **技能边界分布偏移**：与经典模仿学习单条策略内的误差累积（Ross et al., 2011）不同，此处偏移发生在「前序策略的合法终止状态」与「后序策略训练时的精选边界快照」之间。
- 当前系统将交接就绪条件近似写入 `move_to` 的自然语言后置条件（要求目标物体在头部相机中近距离、显著且处于臂展可达距离），并在 Sec. V-C 与 Table IX 提出形式化的下一技能就绪谓词模板 $\phi_t \wedge \rho(s_t, s_{t+1})$。

### 2. 语义执行框架与技能库训练（Sec. III & Appendix）
- **双层解耦架构（Fig. 2）**：
  - **Agent Layer**：维护任务脚本与状态，下发带类型化参数的技能契约，调度多视角 VLM 验证器决定推进（advance）、重试（retry）或重规划（replan），并记录可回放轨迹证据（日志、视频、VLM 判定与恢复提示）。
  - **Embodiment Layer**：负责 BEHAVIOR-1K / OmniGibson 仿真环境、$\pi_{0.5}$ VLA 推理与底层动作块（action chunk）执行。
- **语义技能契约**：$s_t = (\text{name}, \text{args}, \text{prompt}, B_s, K_s, \phi_t)$，包含 9 个技能名（`move_to`、`pick_up_from`、`place_on`、`place_on_next_to`、`place_in`、`press`、`turn_on_switch`、`open_door`、`close_door`；轨迹日志中 `move_to` / `pick_up_from` 原名记录为 `navigate_to` / `grasp`）。
- **$\pi_{0.5}$ VLA 技能库**：PaliGemma 2B 视觉语言骨干 + 300M 参数流匹配动作专家；输入为头部相机 + 左右腕部相机共三路 $224 \times 224$ RGB 图像及本体状态，输出多步动作块。从清洗后的 BEHAVIOR-1K 遥操作片段（每任务 180 训练 / 20 验证集）经两阶段训练：50k 步全技能中期混合训练，再以峰值学习率 $2.5 \times 10^{-5}$（cosine 衰减、全局 batch 256）按 6 个技能组做专精后训练（Table V）。
- **Step-Level Plan-Act-Verify-Replan 循环（Sec. III-B）**：每执行 $K_s = 200$ 步仿真后调用 `gemini-2.5-flash` 审视头/左腕/右腕三视角与 $\phi_t$；仅当正面判定置信度 $> 0.6$ 时推进下一技能，否则继续执行至单技能预算 $B_s$（Table V：多数 800 步，`place_on_next_to` 1200 步，`open_door` 1500 步）或触发重规划（全局回合预算为人类演示长度的 2 倍）。

### 3. 诊断评测与关键结果（Sec. IV & Appendix Table I–VIII）
- **单技能隔离基准（Table I，恢复验证集技能边界快照，多视角 VLM 判定 + 人工复核）**：`place_on` 6/6（100.0%）、`open_door` 7/7（100.0%）、`pick_up_from` 28/29（96.5%）、`turn_on_switch` 7/8（87.5%）、`place_in` 11/13（84.6%）、`move_to` 27/35（77.1%）、`close_door` 4/6（66.7%），`press` 与 `turn_on_switch` 共享权重组未单独评测。
- **端到端组合进度（Table II，10 个 BEHAVIOR-1K 测试任务 × 3 轮共 30 条轨迹，对照 teacher-hint 参考序列人工核查）**：平均进度 **19.5%**（`Turn on radio` $50.0 \pm 43.3\%$、`Make microwave popcorn` $45.8 \pm 7.2\%$、`Move boxes to storage` $30.0 \pm 0.0\%$、`Pick up trash` $22.2 \pm 4.8\%$、`Put shoes on rack` $11.9 \pm 4.1\%$、`Bring in wood` $10.3 \pm 4.4\%$、`Cook hot dogs` $10.3 \pm 4.4\%$、`Set mousetraps` $8.3 \pm 0.0\%$、`Freeze pies` $6.4 \pm 2.2\%$、`Hide Easter eggs` $0.0 \pm 0.0\%$）。
- **代表性单轮全量轨迹诊断（Table III & VIII，10 个任务各 1 实例，$K=200$）**：共 232 次技能尝试（100 成功 / 132 失败）、137,823 仿真帧、196 次 VLM 调用，BDDL 最终任务成功 `1/10`（仅 `Turn on radio` 完成）。剔除 2 次验证器侧案例后，130 次 VLA 侧失败归因于三类：
  1. **Control execution（58 次）**：`grasp` 31 次（接近但未闭合/抓稳）、`actuation` 15 次、`placement` 12 次。
  2. **Target grounding / scene search（37 次）**：导航或开门后目标移出视野或绑定错误实例。
  3. **Next-skill readiness / navigation-to-ready（35 次）**：导航停在臂展外导致无法交接。
- **受控消融（Table VI，Reached-area vs. Arm-reach 导航交接标准）**：收紧至臂展可达后，`Turn on radio` 从失败转为成功（$0/10 \to 1/10$，平均任务分 $\bar{q}: 0.01 \to 0.10$），暴露额外 12 次 readiness 失败（$23 \to 35$）并触发多出 25 次的 `move_to` 重导航尝试（$74 \to 99$）。
- **人工盲审验证器（Table VII）**：双作者对非 `unclear` 样本共识一致性达 90%；21 条达成共识的失败判定中 20 条为真失败，失败判定可靠度 0.95、过严率（over-strict rate）0.05。

### 4. 开源核查结论（步骤 2.5）
- arXiv HTML 全文、摘要与仓库补查结果均未提供官方代码仓库、模型权重或项目主页链接；当前核查状态为 **未确认开源代码**，相关复现需依据论文描述与外部 BEHAVIOR-1K / $\pi_{0.5}$ 生态自行实现。
