# frontend-design（anthropics/skills）

> 来源归档

- **标题：** frontend-design — Anthropic 官方前端审美与设计 lead 技能
- **类型：** repo（monorepo 子目录）
- **链接：** https://github.com/anthropics/skills/tree/main/skills/frontend-design
- **分发：** https://skills.sh/anthropics/skills/frontend-design（`npx skills add anthropics/skills --skill frontend-design`）
- **入库日期：** 2026-09-30
- **代码：** **已开源**（仓库许可见 `LICENSE.txt`；技能正文即规约，无权重）
- **一句话说明：** 把 **反模板化 UI** 的设计原则（字体、布局、动效、文案、自我 critique）编译进 `SKILL.md`，要求先出 token 级设计计划、对照 brief 查「AI 默认审美 tell」，再写代码。
- **为什么值得保留：** 本站 `docs/` 静态站与 [frontend-optimization 清单](../../docs/checklists/frontend-optimization-v1.md) 迭代时，代理易产出 cream/terracotta、SaaS 卡片 kit 等 **生成页指纹**；本技能与 [claude-api 实体页](../../wiki/entities/anthropic-claude-api-skill.md) 同属 Anthropic 官方 skills 包，但域为 **视觉与 UX 交付**。
- **沉淀到 wiki：** 是 → [`wiki/entities/anthropic-frontend-design-skill.md`](../wiki/entities/anthropic-frontend-design-skill.md)

## SKILL 要点（归纳）

- **流程：**  grounding 主题 → 设计 token 计划（色/型/布局/原则）→ **与 brief 对照修订 generic default** → 实现 CSS 特异性纪律 → 单点 bold + 可访问性底线 → 可选截图自评。
- **显式反模式：** 暖 cream + terracotta  accent、acid-green on black、broadsheet 滥用、统一圆角卡片阴影、→ 链式 CTA、headline 单词高亮等「AI 页 tell」。
- **文案：** 界面文字按 **用户可理解** 命名与错误/空态指导，与视觉同等 intentional。

## 关联资料

- skills.sh 页：[`sources/sites/skills-sh-frontend-design.md`](../sites/skills-sh-frontend-design.md)
- 同仓 API 技能：[`anthropics-claude-api-skill.md`](anthropics-claude-api-skill.md)
- 站内曾引用：[`docs/frontend-redesign-plan.md`](../../docs/frontend-redesign-plan.md)
