---
type: comparison
tags:
  - llm-agents
  - coding-agents
  - skills
  - frontend
  - ux
  - agent-infrastructure
status: complete
updated: 2026-09-30
related:
  - ../entities/skillry.md
  - ../entities/taste-skill.md
  - ../entities/impeccable.md
  - ../entities/anthropic-frontend-design-skill.md
  - ../entities/find-skills-skill.md
  - ../../docs/checklists/frontend-optimization-v1.md
sources:
  - ../../sources/sites/skillry-dev.md
  - ../../sources/repos/leonxlnx-taste-skill.md
  - ../../sources/repos/pbakaus-impeccable.md
summary: "Agent 前端/交付物能力选型：Skillry 是闭源交付物市场；Taste Skill 是 MIT 生成约束（三旋钮+禁令）；Impeccable 是 Apache-2.0 设计语言+24 命令+61 detector。与 Anthropic frontend-design 可叠加。"
---

# Skillry vs Taste Skill vs Impeccable（Agent 前端与交付物选型）

三者都服务 **coding agent 产出更好看的界面或媒体**，但 **治理层不同**：

| 产品 | 隐喻 | 开源 | 核心机制 |
|------|------|------|----------|
| [Skillry](../entities/skillry.md) | **交付物市场** | 否（Skill 私有包） | ~150 精选 Skill；Web/Slides/Image/Video；先看效果再装 |
| [Taste Skill](../entities/taste-skill.md) | **生成约束层** | MIT | 三旋钮 + 硬禁令 + pre-flight |
| [Impeccable](../entities/impeccable.md) | **设计语言 + 检测器** | Apache-2.0 | 24 命令 + PRODUCT/DESIGN.md + 61 detector hooks |

[Anthropic frontend-design](../entities/anthropic-frontend-design-skill.md) 仍是 **官方 UI skill 基线**；Impeccable 自述由其演进，Taste 与 Impeccable 常 **叠在** frontend-design 或裸 agent 之上。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| UI | User Interface | 三产品主要改善 HTML/CSS/React 观感 |
| CLI | Command-Line Interface | skillry-cli / npx skills / npx impeccable |
| MIT | MIT License | Taste Skill 协议 |
| OAuth | Open Authorization | Skillry 登录方式 |
| PR | Pull Request | Impeccable CLI 可对接 CI |

## 核心特性对比

| 维度 | Skillry | Taste Skill | Impeccable |
|------|---------|-------------|------------|
| **主要产出** | Landing、deck、OG、视频等 **成品流** | **前端代码** 审美 | **前端代码** + 设计文档 |
| **发现/安装** | skillry.dev + `skillry-cli login` | `npx skills add` Git URL | `npx impeccable install` 或 skills add |
| **费用** | 订阅（页显 ~$9.99/月） | 免费 | 免费（开源） |
| **约束形式** | 每 Skill 独立指令包 | SKILL 内规则 + 3 dials | 命令 + MD 文件 + detector |
| **可审计性** | 闭源 ZIP | 公开 SKILL.md | 公开 skill + 规则源码 |
| **迭代已有 UI** | 视 Skill 而定 | redesign audit 协议 | polish / distill / audit |
| **自动化检测** | 无统一引擎 | pre-flight（agent 自检） | **61 确定性规则** + hooks |
| **非 Web 产出** | **Slides / Image / Video** | image 参考板 skill | 弱（偏 UI 工程） |

## 如何选型？

### 何时优先 Skillry？

1. **目标是一次性高完成度交付物**（发布会 landing、Starlit deck、片头），且愿为 **curated 工作流** 付订阅。
2. **希望先看 catalog 演示与 installs**，而不是自己拼 SKILL。
3. **需要 Video/Image/Slides 四类之一**，开源双雄不覆盖同等成品库。

### 何时优先 Taste Skill？

1. **从零生成前端**，要强 **反模板** 与 **brief 推断**（行业/情绪/布局家族）。
2. 想用 **旋钮** 快速在「对称干净 ↔ 不对称实验」「hover ↔ scroll 动效」「留白 ↔ 仪表盘密度」间切换。
3. 团队 **只愿意引入 MIT SKILL**，不要闭源包或二进制引擎。

### 何时优先 Impeccable？

1. **长期维护同一产品 UI**，需要 `PRODUCT.md` / `DESIGN.md` 与 **polish、typeset、distill** 等 **共享动词**。
2. 要在 **agent 编辑循环或 PR** 里 **实证** 去掉 AI tell（61 detector，无 API key）。
3. 需要 **live 浏览器** 点选迭代或 **design direction** 案例库作起点。

### 叠加建议（非互斥）

- **Impeccable init + Taste 默认 skill：** Impeccable 管项目真相与检测；Taste 管 **生成瞬间** 的 layout/motion 方言。
- **frontend-design + Impeccable：** 保留 Anthropic 计划式 tell 清单，用 Impeccable 命令与 hook **落地与验收**。
- **Skillry + Impeccable：** Skillry 出第一版 landing；Impeccable **polish** 对齐既有 `DESIGN.md`（若站点要长期统一）。

## 与本站 Robotics_Notebooks 的关系

- **wiki 知识 vs docs 展示：** 三工具都不替代 [Ingest Workflow](../../schema/ingest-workflow.md)；它们优化 **读者看到的静态站与 agent 维护体验**（见 [前端体验优化清单](../../docs/checklists/frontend-optimization-v1.md)）。
- **机器人本体栈：** 仿真、RL、Sim2Real **不依赖** 这些 skill；价值在 **项目页、图谱 UI、对外 demo**。

## 关联页面

- [Skillry](../entities/skillry.md)
- [Taste Skill](../entities/taste-skill.md)
- [Impeccable](../entities/impeccable.md)
- [frontend-design（Anthropic）](../entities/anthropic-frontend-design-skill.md)
- [find-skills](../entities/find-skills-skill.md)

## 参考来源

- [Skillry 站点归档](../../sources/sites/skillry-dev.md)
- [Leonxlnx/taste-skill 归档](../../sources/repos/leonxlnx-taste-skill.md)
- [pbakaus/impeccable 归档](../../sources/repos/pbakaus-impeccable.md)

## 推荐继续阅读

- [skills.sh](https://skills.sh/) — 公开 skill 排行榜（与 Skillry 独立）
- [Impeccable 安装文档](https://impeccable.style) — hooks 与 Copilot 内置说明
