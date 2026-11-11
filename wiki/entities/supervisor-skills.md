---
type: entity
project_id: supervisor-skills
tags: [ai-auto-research, research-skills, academic-writing, llm-agents, hkust]
status: complete
updated: 2026-11-11
code: https://github.com/HKUSTDial/Supervisor-Skills
original:
  - ../../sources/repos/supervisor-skills.md
sources:
  - ../../sources/repos/supervisor-skills.md
related:
  - ../concepts/ai-auto-research.md
  - ./paperspine.md
  - ./nature-polishing.md
  - ./aris-auto-research-in-sleep.md
summary: "Supervisor-Skills 通过 Handbook 与 SKILL.md 将科研选题、证据约束、论文论证、图表设计和投稿前评审封装为可供不同宿主 Agent 调用的科研副导师技能。"
---

# Supervisor-Skills（科研副导师技能库）

## 一句话定义

Supervisor-Skills 是由 HKUSTDial 维护的 **Handbook 理论指南 + Agent Skills 可执行清单**，用于协助研究者从 Idea 评估、文献调研、技术论文逻辑骨架，推进到有证据约束的写作与投稿检查。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| LLM | Large Language Model | 宿主大模型执行 Markdown 技能的基础 |
| AI | Artificial Intelligence | 课题评估与写作辅助方式 |
| SKILL | Agent Skill | 本仓库的结构化可复用流程文本，非训练模型 |
| MECE | Mutually Exclusive, Collectively Exhaustive | deep-research 的调研分类原则 |

## 为什么重要

科研新手常见的问题不是缺一句英文润色，而是课题是否成立、论证链是否一致、主张是否有证据。官方 README 明确以导师的选题、写作、审稿经验设计 Skill；对机器人论文的基线实验、方法动机与论文组织也有借鉴价值。

## 核心原理

1. **Guide 与执行技能分工：** `handbook/` 讲论文质量、Idea 生命周期、Introduction 及图表；`skills/` 通过 `SKILL.md` 将这些检查拆成可调用步骤。
2. **选题到结构：** `idea-evaluator` 与 `deep-research` 先核对研究问题与文献；`tech-paper-template` 对齐 Background → Limitations → Key Idea/Goal → Challenges → Methodology Modules → Contributions。
3. **写作与证据门控：** `intro-drafter`、`paper-writer`、`paper-polish` 分工不同。Paper Writer 明确所有事实性论断须来自作者材料、本次核验或不包含具体量化细节的领域常识。
4. **交付核查：** `figure-designer`、`pre-submission-reviewer`、`rebuttal-guidance` 参与可视化、投稿前审核与审稿意见回复。不同技能是可选调用，不保证自动端到端运行。

## 流程总览

```mermaid
flowchart LR
  Idea["研究者提出 Idea"] --> Eval["idea-evaluator 评估"]
  Eval --> Lit["deep-research 文献与证据"]
  Lit --> Logic["tech-paper-template 论文逻辑骨架"]
  Logic --> Intro["intro-drafter Introduction"]
  Intro --> Write["paper-writer 正文"]
  Write --> Polish["paper-polish 语言修订"]
  Logic --> Figure["figure-designer 图表规划"]
  Polish --> Review["pre-submission-reviewer 投稿检查"]
  Figure --> Review
  Review --> Rebuttal["rebuttal-guidance 审稿回应"]
```

图示按官方 `skills/README.md` 的上下游关系提炼，表示建议的技能组合而非仓库内自动调度器。

## 源码运行时序图

**不适用（独立训练/推理运行时）**：项目主体是由宿主 Agent 读取的 `SKILL.md` 与 Handbook，没有声明自带一个自动把全流程跑完的独立科研执行器。运行时步骤由宿主 Agent 及所选 Skill 控制；未对官方辅助脚本做完整运行验证，不伪造源码级时序。

## 工程实践

- 入门可先阅读 `handbook/02_Idea_Generation/` 和 `03_Paper_Writing/`，按需安装技能。
- 研究思路模糊先使用 `idea-evaluator`；主张与方法不匹配先用 `tech-paper-template`；有材料才由 `paper-writer` 起草。
- 对实验结果、引用和模型生成内容保持人工核验，论文作者承担科学责任。
- 官方上游 `HKUSTDial/Supervisor-Skills` 已核对非 Fork；本页版本 `4bd3fef393e5`。

## 局限与风险

Skill 文件不是可证明有效的导师替代品；独立可复现实验与论文接受率没有由本次核验建立。README 声明 CC BY-NC-SA 4.0，商用前核对具体文件授权；任何 AI 写作均不得添加未知的引用、数值和实验机制。

## 关联页面

- [PaperSpine](paperspine.md)：更重论文贡献—动机—交付任务状态。
- [nature-polishing](nature-polishing.md)：聚焦已写论文的证据约束润色。
- [ARIS](aris-auto-research-in-sleep.md)：侧重真实实验运行与独立审稿循环。
- [AI Auto-Research](../concepts/ai-auto-research.md)：研究自动化全生命周期背景。

## 参考来源

- [官方源码归档](../../sources/repos/supervisor-skills.md)
- [HKUSTDial/Supervisor-Skills 原仓库](https://github.com/HKUSTDial/Supervisor-Skills)

## 推荐继续阅读

- [官方 Handbook 与 Skills 目录（固定版本）](https://github.com/HKUSTDial/Supervisor-Skills/tree/4bd3fef393e5)
- [Paper Writer 证据门控原文](https://github.com/HKUSTDial/Supervisor-Skills/blob/4bd3fef393e5/skills/paper-writer/SKILL.md)
