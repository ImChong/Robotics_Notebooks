# Taste Skill（tasteskill.dev）

- **标题：** Taste Skill — The Anti-Slop Frontend Framework for AI Agents
- **类型：** site / project-page
- **URL：** <https://tasteskill.dev>
- **入库日期：** 2026-09-30
- **代码：** <https://github.com/Leonxlnx/taste-skill>（归档 [`sources/repos/leonxlnx-taste-skill.md`](../repos/leonxlnx-taste-skill.md)）
- **开源状态：** **已开源**（MIT）

## 一句话摘要

开源 **Agent Skill 包**，用 **brief 推断 + 三旋钮（方差 / 动效 / 密度）+ 硬禁令 + 起飞前检查** 约束前端生成，减少「AI 模板味」；默认 skill 安装名 **`design-taste-frontend`**（v2 experimental 为当前默认）。

## 公开信息要点（截至入库日 2026-09-30）

- **安装：** `npx skills add https://github.com/Leonxlnx/taste-skill --skill "design-taste-frontend"`
- **技能族：** `taste-skill`（默认）、`taste-skill-v1`（兼容旧行为）、`gpt-tasteskill`、`output-skill`、`minimalist-skill`、`brutalist-skill` 等；另含 **image-generation skills** 做参考板。
- **v2 机制（站点）：** §0 brief inference；§2 brief→设计系统映射（Material / shadcn / GOV.UK 等）；§8 暗色协议；§11  redesign audit-first；§14 **hard pre-flight check**。
- **三旋钮（README Settings）：** `DESIGN_VARIANCE`、`MOTION_INTENSITY`、`VISUAL_DENSITY` — 分别控制布局实验度、动画深度、信息密度。
- **兼容：** 宣称 Cursor、Claude Code、Codex、Gemini CLI、v0、Lovable、OpenCode 等一切支持 `SKILL.md` 的工具。

## 为何值得保留

- 与 [Anthropic frontend-design](../../wiki/entities/anthropic-frontend-design-skill.md)、[Impeccable](../../wiki/entities/impeccable.md) 形成 **开源前端审美约束** 对照轴；stars 量级高（GitHub API 入库日约 91k+），社区采用面大。

## 关联资料

- 代码归档：[`sources/repos/leonxlnx-taste-skill.md`](../repos/leonxlnx-taste-skill.md)
- wiki：[`wiki/entities/taste-skill.md`](../../wiki/entities/taste-skill.md)
