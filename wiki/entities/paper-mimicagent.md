---
type: entity
tags: [paper, quadruped, imitation-learning, llm-agents, example-guided-rl, sim2real, cmu]
status: complete
updated: 2026-10-30
arxiv: "2609.24145"
related:
  - ../methods/reinforcement-learning.md
  - ../tasks/hybrid-locomotion.md
  - ../concepts/sim2real.md
  - ../concepts/ai-auto-research.md
sources:
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-09-21_25.md
  - ../../sources/papers/mimicagent_arxiv_2609_24145.md
  - ../../sources/sites/mimic-agent-github-io.md
  - ../../sources/repos/mimicagent-code.md
summary: "MimicAgent（CMU, arXiv:2609.24145）：LLM agent 从文本生成粗参考轨迹，再 example-guided RL 学四足动态技能；87% prompt 语义对齐，用户偏好常优于 Eureka。"
---

# MimicAgent（arXiv:2609.24145）

**MimicAgent**（*Quadruped Skills via Prompt-to-Trajectory Generation*，CMU，[arXiv:2609.24145](https://arxiv.org/abs/2609.24145)）提出 **prompt-to-trajectory** agentic 管线：用 LLM coding agent 生成 **运动学参考，可能不满足动力学约束** 的参考轨迹，再交给 **example-guided RL**（DeepMimic 族）在仿真与 **Unitree Go2 / Go2-W** 真机落地；相对 Eureka 等 **LLM reward design**，强调 **轨迹合成比 reward shaping 更易泛化**。

## 一句话定义

用 LLM 把自然语言技能变成可执行 reference trajectory，再用模仿式 RL 补全动力学，让四足（及扩展人形）从文本获得多样动态技能。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 参考轨迹跟踪与 sim2real |
| FK | Forward Kinematics | MuJoCo 运动学执行，无 physics |
| LLM | Large Language Model | motion planner / code generator agent |
| Sim2Real | Simulation to Real | Go2 / Go2-W 真机部署 |
| EGRL | Example-Guided RL | 参考运动 + RL 跟踪（DeepMimic 族） |

## 为什么重要

- **四足数据缺口：** 人形有大规模 mocap，四足公开参考库远小；MimicAgent 用 LLM **合成** 参考，绕过 reward engineering。
- **与 Eureka 对照：** 同一文本 prompt，轨迹法在 **7 技能** 与 **57 人用户研究** 中多次优于 LLM reward 与手工 keyframe（Bound / RYP 等例外见项目页表）。
- **Agent 范式：** motion planner → code gen → unit tests → self-improvement，与 [ai-auto-research](../concepts/ai-auto-research.md) S3 代码生成环同构但目标域为 **motor skill**。

**核查状态：** 本次未能获取论文全文，既有论文细节和主流程图尚待通读原文复核；本次源码时序图仅针对公开的人形扩展。

## 核心信息

| 字段 | 内容 |
|------|------|
| 机构 | 卡内基梅隆大学（CMU） |
| arXiv | [2609.24145](https://arxiv.org/abs/2609.24145) |
| 项目页 | <https://luckykantnayak.github.io/mimic-agent/> |
| 开源状态 | 四足仓库占位；人形运动学生成代码已公开，见 [源码归档](../../sources/repos/mimicagent-code.md) |
| 真机 | Unitree Go2、Go2-W（wheeled quadruped） |

## 流程总览

```mermaid
flowchart LR
  P["文本 skill prompt"] --> MP["Motion planner agent\n分阶段技能描述"]
  MP --> CG["Code generator\nbase + foot 轨迹代码"]
  CG --> FK["MuJoCo FK 执行\n无 physics sim"]
  FK --> UT["Task-agnostic unit tests"]
  UT -->|fail| RD["Diagnostic + 代码修订\n≤3 轮"]
  RD --> CG
  UT -->|pass| REF["参考轨迹 τ"]
  REF --> RL["Example-guided RL"]
  RL --> SIM["仿真 rollout"]
  RL --> REAL["真机 Go2 / Go2-W"]
```

### Unit tests（技能无关）

- Base height envelope
- Joint limit violations
- Foot ground penetration
- Non-aerial 脚接触约束

## 工程实践

| 检查项 | 建议 |
|--------|------|
| 参考质量 | 87% prompt 语义对齐（Claude Fable 5.1 + harness）；仍依赖 unit test 过滤退化轨迹 |
| 粗轨迹可用性 | 动力学不可行轨迹 **足够** 作 RL 目标 — 勿要求 LLM 输出 physics-valid |
| 开源 | 四足仓库尚无实现；人形仓库公开生成及回放代码，不等于四足训练代码已发布 |
| 人形扩展 | 项目页展示 G1 运动学参考；这不证明策略可跟踪或人形真机部署 |

## 源码运行时序图

以下为人形仓库 commit `daac0331fd3cbc32c24e63449ca6ed6c900fbcc7` 的生成入口；四足仓库仍无可运行实现。

```mermaid
sequenceDiagram
  participant U as 用户
  participant R as agent/run.py
  participant G as pipeline.py
  participant P as 规划与代码生成
  participant E as 执行与测试
  participant D as 诊断与修复
  U->>R: --skill 与 --name
  R->>G: prepare_skill 与 build_graph
  G->>P: goal contract、筛选、规划、生成
  P->>E: 执行生成代码并进行检查
  alt 需要修复
    E->>D: 检查结果
    D->>E: 修订代码并重新执行
  end
  E->>G: 通过或达到预算
  G->>R: finalize 与 outcome
  R->>U: 轨迹、报告和 metadata
```

此图概括正常路径与诊断修复分支；路由也可能触发重新规划或提前结束。生成结果是运动学参考，不保证动力学可行或控制器可跟踪。

## 实验与评测读法

- **7 技能 baseline：** Eureka、Manual Keyframing、MimicAgent — Trot/Bound/Flip/AC/CDS/RYP 等（Go2 vs Go2-W 分工见项目页）。
- **用户研究：** n=57，1–5 偏好分；MimicAgent 在 Side Flip、Front Flip、AC、CDS 等领先。
- **真机：** four-leg trot、two-leg walk/skate/roll、handstand 过渡等 clip — 证明 sim2real 非仅仿真。

## 与其他工作对比

| 维度 | MimicAgent（本页） | Eureka 式 **LLM reward design** | 人工动捕参考 + DeepMimic |
|------|---------------------|----------------------------------|---------------------------|
| LLM 产出物 | **参考轨迹**（kinematically feasible、dynamics-infeasible） | 奖励函数代码 | 不用 LLM |
| 下游学习 | example-guided RL 补全动力学 | 常规 RL 用生成的奖励 | 模仿参考动作 |
| 失败可诊断性 | 高：轨迹可用 **技能无关单元测试** 过滤（高度包络 / 关节限位 / 足部穿透 / 非腾空接触） | 低：奖励错了要靠训练结果反推 | 高，但受限于动捕覆盖 |
| 获取新技能的成本 | 写一句 prompt | 写一句 prompt | 需新动捕 |

- **本文的核心主张是「轨迹比奖励好生成」：** 奖励函数的错误要经过一整轮 RL 才暴露，而参考轨迹可以在 **执行前** 用运动学单元测试筛掉——这是它相对 reward-design 路线在泛化上的结构性优势，而不是模型更强。
- **「动力学不可行」是特性不是缺陷：** 参考只需给出 **目标形状**，动力学由 RL 补；要求 LLM 直接输出 physics-valid 轨迹反而会把任务变难。
- **读数边界：** **87% prompt 语义对齐** 与用户偏好优于 Eureka 属 **作者报告**；四足实现仍待发布；人形代码提供运动学参考生成，主实验与指标在四足，不要将人形预览当作真机结论。

## 结论

**MimicAgent 把四足技能学习从「reward 景观搜索」改写成「参考轨迹合成 + EGRL」，对缺 mocap 的 embodiment 尤具启发。**

1. LLM **生成参考** 比 **生成 reward** 更易跨技能/形态泛化（相对 Eureka）。
2. **粗、动力学不可行** 轨迹仍可作为 DeepMimic 族有效监督 — 降低对 mocap 精度依赖。
3. Agentic 环（planner → code → test → revise）是 scalable skill library 的关键工程壳。
4. 四足实现待发布，人形生成代码已公开；部署前仍需验证跟踪与动力学。
5. 与 [reinforcement-learning](../methods/reinforcement-learning.md) / DeepMimic 对照阅读，明确 **参考从哪来** 这一变量。

## 局限与风险

- **Wheeled quadruped：** Go2-W 技能（flip、skating）对 unit test 与 RL 稳定性要求更高；Eureka 在 RYP 上仍可能更优。
- **人形：** G1 预览为运动学参考，不能证明控制器可跟踪；人形真机未作为主结果。
- **LLM 依赖：** prompt / 模型版本变更会影响 87% 对齐率的可复现性。

## 关联页面

- [reinforcement-learning](../methods/reinforcement-learning.md)
- [hybrid-locomotion](../tasks/hybrid-locomotion.md)
- [sim2real](../concepts/sim2real.md)
- [ai-auto-research](../concepts/ai-auto-research.md)

## 参考来源

- [mimicagent_arxiv_2609_24145.md](../../sources/papers/mimicagent_arxiv_2609_24145.md)
- [mimic-agent-github-io.md](../../sources/sites/mimic-agent-github-io.md)
- [官方仓库核查](../../sources/repos/mimicagent-code.md)
- [arXiv:2609.24145](https://arxiv.org/abs/2609.24145)

## 推荐继续阅读

- [MimicAgent 项目页](https://luckykantnayak.github.io/mimic-agent/)
- [DeepMimic（arXiv:1804.00657）](https://arxiv.org/abs/1804.00657)
- [Eureka（arXiv:2310.12931）](https://arxiv.org/abs/2310.12931)
