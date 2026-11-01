# Explore, Execute, Evolve: A Skill Acquisition and Reuse Loop for Embodied Agents（arXiv:2609.37810）

> 来源归档（paper）

- **标题：** Explore, Execute, Evolve: A Skill Acquisition and Reuse Loop for Embodied Agents
- **arXiv：** <https://arxiv.org/abs/2609.37810>（v1，2026-09-29）
- **代码：** <https://github.com/SII-dannyXSC/RoboSkill>
- **作者：** Sicheng Xie、Yitong Chen、Haidong Cao、Shunlin Lu、Zuxuan Wu、Yu-Gang Jiang
- **许可证：** MIT（依据官方仓库 LICENSE）
- **入库日期：** 2026-10-27
- **一句话说明：** RoboSkill 将具身智能体的技能获取和复用组织成 Explore、Execute、Evolve 闭环，并结合触觉反馈与可复用代码，降低新任务探索和执行开销。
- **对 wiki 的映射：** [RoboSkill 实体页](../../wiki/entities/paper-roboskill.md)

## 核心论文摘录

### 1. Explore–Execute–Evolve 技能闭环

- **出处：** 摘要。
- **要点：** 智能体先探索并收集与任务相关的信息，再执行任务并根据反馈调整；随后根据执行记录进化技能库。后续任务重新利用已积累的技能来引导探索和执行，形成持续改进循环。
- **对 wiki 的映射：** 见实体页「核心原理」与「流程总览」。

### 2. 触觉反馈与可复用代码

- **出处：** 摘要。
- **要点：** 视觉之外加入触觉反馈，以降低物理交互中的不确定性；技能复用时以可复用代码补充文本指导，减少推理开销。
- **对 wiki 的映射：** 见实体页「工程实践」与「局限与风险」。

### 3. 评测结果

- **出处：** 摘要。
- **要点：** 在 LIBERO-10 上，RoboSkill 对四种 agent 的首回合成功率提升 12.5–25.0 个百分点，平均运行时间降低 7.6–72.4%；真机实验成功率提升 8.3 个百分点，成功试验平均运行时间至少降低 14.4%。
- **解读边界：** 这些是论文摘要报告的对比结果；不同 agent / 任务设置的绝对值与运行条件应以论文正文表格为准，不把跨设置的百分比直接当作同一基线的横向排名。
- **对 wiki 的映射：** 见实体页「实验与评测」及「结论」。

## 官方资料核查

- arXiv 摘要给出 RoboSkill 方法与主要实验结论；对应官方代码仓库为 <https://github.com/SII-dannyXSC/RoboSkill>。
- 仓库 README 将发布内容描述为论文使用的原生 Agent CLI 执行路径，包含 LIBERO Gateway、Harness 生命周期管理和实验调度配置；代码以 MIT 许可证发布。
- 仓库 README 明确说明该发布不包含运行工作区、日志、会话、视频、凭据、历史结果、生成的经验包或技能。因此，源码开放不等于这些生成数据和完整历史实验产物一并开放。
- **对 wiki 的映射：** 复现入口、源码边界与运行路径见实体页「工程实践」「源码运行时序图」。

## 对 wiki 的映射

- 统一实体：[wiki/entities/paper-roboskill.md](../../wiki/entities/paper-roboskill.md)
- 官方代码导航：[sources/repos/roboskill.md](../repos/roboskill.md)
