# Impeccable（pbakaus/impeccable）

> 来源归档

- **标题：** Impeccable
- **类型：** repo
- **作者：** Paul Bakaus 等（[@pbakaus](https://github.com/pbakaus)）
- **链接：** https://github.com/pbakaus/impeccable
- **官方站：** https://impeccable.style
- **分发：** `npx impeccable install`；`npx skills add pbakaus/impeccable`；Claude Code marketplace `pbakaus/impeccable`
- **入库日期：** 2026-09-30
- **协议：** Apache-2.0
- **一句话说明：** 面向 coding agent 的 **设计词汇表**：单 skill 暴露 **24 条 `/impeccable` 命令**，用 **PRODUCT.md / DESIGN.md** 持久化产品与视觉系统，并以 **61 条确定性 detector**（CLI hook、PR、Chrome 扩展）在编辑循环中移除 AI 默认 tell；支持 **live** 浏览器迭代。
- **为什么值得保留：** 把 Anthropic frontend-design 的审美意图 **操作化 + 可测**；适合与本站静态站 PR 评审、agent 改 UI 的工作流对照。
- **沉淀到 wiki：** 是 → [`wiki/entities/impeccable.md`](../../wiki/entities/impeccable.md)

## README 要点（归纳）

- **缘起：** 自 [anthropics/skills frontend-design](https://github.com/anthropics/skills/tree/main/skills/frontend-design) 扩展。
- **命令（24）：** `init`, `document`, `extract`, `shape`, `craft`, `critique`, `audit`, `polish`, `distill`, `typeset`, `layout`, `animate`, `colorize`, `bolder`, `quieter`, `harden`, `onboard`, `clarify`, `adapt`, `optimize`, `delight`, `overdrive`, `live`, `generate`；`pin` 可拆成 `/audit` 等快捷方式。
- **产物文件：** `PRODUCT.md`（用户/目的/约束，非视觉）；`DESIGN.md`（token、组件、品牌规则）。
- **Detector：** 61 规则 — **无 LLM**；抓 Inter 泛滥、卡片套卡片、灰字上彩底、AI beige、italic serif 标题等。
- **引擎：** skill 附带 launcher + 自包含 binary（`~/.impeccable/bin/` 首次下载）；`npx impeccable update` 更新。
- **Anti-patterns 清单：** 禁用 bounce/elastic easing、纯黑灰、过度 card 嵌套等。

## 项目页核查（步骤 2.5）

- **已开源：** Apache-2.0；skill 与 CLI 源码公开。
- **Copilot：** 站点称 GitHub Copilot app **内置** Impeccable（实验设置开启）— 与纯 GitHub 仓安装并存。

## 与本站 sources 的其它锚点

- 项目页：[`sources/sites/impeccable-style.md`](../sites/impeccable-style.md)
- Anthropic 基线：[`wiki/entities/anthropic-frontend-design-skill.md`](../../wiki/entities/anthropic-frontend-design-skill.md)
