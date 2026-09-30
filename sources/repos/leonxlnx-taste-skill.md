# Taste Skill（Leonxlnx/taste-skill）

> 来源归档

- **标题：** Taste Skill
- **类型：** repo
- **作者：** Leonxlnx（[@Leonxlnx](https://github.com/Leonxlnx)）
- **链接：** https://github.com/Leonxlnx/taste-skill
- **官方站：** https://tasteskill.dev
- **分发：** `npx skills add https://github.com/Leonxlnx/taste-skill --skill "design-taste-frontend"`（及 `--skill` 选其它子 skill）
- **入库日期：** 2026-09-30
- **协议：** MIT
- **一句话说明：** 开源 **反 AI 模板** 前端 Agent Skill 库：默认 **taste-skill v2** 在生成前做 brief 推断与设计系统选型，用 **DESIGN_VARIANCE / MOTION_INTENSITY / VISUAL_DENSITY** 三旋钮与 **硬禁令 + §14 pre-flight** 约束输出；含 brutalist / minimalist / GPT 变体与 image 参考板 skill。
- **为什么值得保留：** 高 star 社区默认「去 slop」技能之一；与本站 [docs/ 展示层](../../docs/checklists/frontend-optimization-v1.md) 及 [Impeccable](../../wiki/entities/impeccable.md) 选型直接相关。
- **沉淀到 wiki：** 是 → [`wiki/entities/taste-skill.md`](../../wiki/entities/taste-skill.md)

## README 要点（归纳）

- **定位：** *Anti-Slop Frontend Framework for AI Agents* — 提升 layout / typography / motion / spacing，避免 boilerplate UI。
- **默认 skill：** `design-taste-frontend`（taste-skill v2 experimental，安装名稳定、规则迭代中）。
- **Settings（仅 taste-skill）：** 三 dial — 方差（对称 vs 不对称）、动效深度（hover vs scroll/magnetic）、视觉密度（留白 vs 仪表盘）。
- **v2 结构：** brief inference；design-system map；dark mode protocol；redesign audit-first；block library schema；hard pre-flight。
- **硬约束示例：** em-dash ban；canonical GSAP skeletons（与动效强度旋钮联动）。
- **其它 skills：** v1 保留、`gpt-taste`（GPT/Codex 更严）、`output-skill`（防占位符半成品）、minimalist / brutalist 视觉方言。
- **Agent Skills 兼容：** 徽章指向 vercel-labs/agent-skills 生态。

## 项目页核查（步骤 2.5）

- **已开源：** 全仓 MIT；SKILL.md 与 assets 公开可审计。
- **无官方闭源 runtime：** 依赖 harness 读取 SKILL；无单独付费引擎（站点赞助与 API 联盟链接不等于 skill 闭源）。

## 与本站 sources 的其它锚点

- 项目页：[`sources/sites/tasteskill-dev.md`](../sites/tasteskill-dev.md)
- 对照：[`wiki/comparisons/skillry-taste-skill-impeccable.md`](../../wiki/comparisons/skillry-taste-skill-impeccable.md)
