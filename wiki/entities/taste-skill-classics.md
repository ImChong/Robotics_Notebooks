---
type: entity
project_id: leonxlnx-taste-skill-classics
code: "https://github.com/Leonxlnx/taste-skill-classics"
tags: [llm-agents, coding-agents, skills, frontend, ux, open-source]
status: complete
updated: 2026-11-22
original:
  - ../../sources/repos/leonxlnx-taste-skill-classics.md
related:
  - ../methods/coding-agent-verification-loops.md
  - ../methods/claude-code-dynamic-workflows.md
  - ./taste-skill.md
  - ./anthropic-frontend-design-skill.md
  - ./impeccable.md
  - ./claude-code-frontend-skills.md
sources:
  - ../../sources/repos/leonxlnx-taste-skill-classics.md
summary: "Taste Skill Classics 保存从 Taste 主仓分出的七组历史风格 Skill，尤其极简和工业粗野主义；需显式选择，不能与当前 Taste v2.5 的默认行为混同。"
---

# Taste Skill Classics｜历史风格包与工业粗野主义

## 一句话定义

将 Taste Skill 的七套**历史/专项视觉规则**独立保存为按名称显式调用的 Skills，适合需要明确风格审查的项目，不是新版 Taste 的可执行引擎。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| UI | User Interface | 视觉风格与组件 |
| UX | User Experience | 排版、信息密度与互动体验 |
| CRT | Cathode-Ray Tube | Tactical Telemetry 技能借鉴的终端视觉语汇 |
| HUD | Head-Up Display | 信息密集型控制台视觉范式 |
| GPT | Generative Pre-trained Transformer | `gpt-taste` 针对的模型类别 |

## 为什么重要

对工业控制台、运动控制 dashboard、机器人参数看板，可用**机械式表格/遥测视觉**表达用途；但粗野风格应该是明确选择，不能自动覆盖既有站点。对于普通文档/产品页面，极简风格包的扁平版式可能更稳妥。

## 核心原理

| 独立技能 | 源码规约的核心侧重点 |
|---|---|
| `high-end-visual-design` | 低对比、留白和柔和运动的高端表现 |
| `minimalist-ui` | 编辑排版、克制的单色/低饱和配色与平面 Bento |
| `industrial-brutalist-ui` | Swiss Industrial Print 或 Tactical Telemetry/CRT；**一项目仅选一种视觉原型** |
| `gpt-taste` | 更严格的 GPT/Codex 设计风格指令 |
| `stitch-design-taste` | Google Stitch 的设计与可选 `DESIGN.md` |
| `design-taste-frontend-v2-experimental` | 主仓早期三旋钮实验版 |
| `design-taste-frontend-v1` | 原始版本兼容 |

## 流程总览

```mermaid
flowchart LR
  A["确定品牌与界面用途"] --> B{"是否需要历史风格？"}
  B -->|否| C["使用当前 Taste Skill"]
  B -->|是| D["显式选择一个 Classics Skill"]
  D --> E["阅读所选 SKILL.md"]
  E --> F["按风格约束生成"]
  F --> G["检查可读性及实际页面"]
```

## 源码运行时序图

**不适用**：所核对的 `skills/brutalist-skill/SKILL.md`、`skills/minimalist-skill/SKILL.md` 是提示规约而非统一运行程序；执行与加载依赖外部 Agent harness，不能虚构内部调用时序。

## 工程实践

- 示例：`npx skills add https://github.com/Leonxlnx/taste-skill-classics --skill "industrial-brutalist-ui"`。
- 极简风格适合白底文档阅读，工业风格适合明确要求的硬件展示；若已有设计系统或品牌配色，先以品牌约束为准。
- 和当前 [Taste Skill](taste-skill.md) 分仓维护；不把 Classics 的三旋钮描述成当前 v2.5 默认证据。

## 局限与风险

- 旧版规则可能和现代无障碍规范、产品品牌或 [Emil Skills](emil-kowalski-skills.md) 的低频才动原则冲突。
- 风格包并不提供软件运行时、用户测试或跨设备验收；需要浏览器实际评估。

## 关联页面

- [Coding Agent 验证闭环](../methods/coding-agent-verification-loops.md) — 历史风格规则不能取代已实现页面的实际验收
- [Claude Code 动态工作流](../methods/claude-code-dynamic-workflows.md) — 将显式选风格、实现与审查拆成有边界的任务阶段

- [Taste Skill](taste-skill.md) — 当前主线
- [frontend-design](anthropic-frontend-design-skill.md) — 主题驱动的基座审美
- [Impeccable](impeccable.md) — 成品审查
- [Claude Code Frontend Skills](claude-code-frontend-skills.md) — 按任务抑制冲突的路由层

## 参考来源

- [Classics 官方 README/Skill 归档](../../sources/repos/leonxlnx-taste-skill-classics.md)

## 推荐继续阅读

- [官方七个 Skills 目录](https://github.com/Leonxlnx/taste-skill-classics)
- [工业粗野主义 SKILL.md](https://github.com/Leonxlnx/taste-skill-classics/blob/main/skills/brutalist-skill/SKILL.md)
