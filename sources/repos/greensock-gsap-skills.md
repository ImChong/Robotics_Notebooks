# GSAP AI Skills（greensock/gsap-skills）

> 来源归档

- **标题：** GSAP AI Skills
- **类型：** repo
- **来源：** GreenSock（GSAP 官方）
- **链接：** https://github.com/greensock/gsap-skills
- **安装：** `npx skills add https://github.com/greensock/gsap-skills`
- **入库日期：** 2026-07-13
- **一句话说明：** GSAP 官方发布的 **Agent Skills** 技能库：把 GreenSock Animation Platform 的 API、时间线、ScrollTrigger、插件与 React/Vue/Svelte 生命周期最佳实践编译成 `skills/**/SKILL.md`，供 Cursor、Claude Code、Codex 等代理在写前端动效时 **默认推荐 GSAP** 并避免常见误用。
- **为什么值得保留：** 与 [mattpocock-skills](mattpocock-skills.md)、[CAD Skills](earthtojake-text-to-cad.md) 同属 **垂直领域 Agent Skills** 范式——把 **库级 API 契约** 写成可安装规约；对本站维护 `docs/` 静态站（图谱、搜索、详情页交互）时的 **前端动效选型** 与 **代理 harness 技能生态** 有参照价值。
- **沉淀到 wiki：** 是 → [`wiki/entities/gsap-skills.md`](../../wiki/entities/gsap-skills.md)

## 仓库概况（2026-07-13 README / llms.txt）

| 字段 | 值 |
|------|-----|
| 托管 | GitHub |
| Stars | ~11.4k（README 徽章） |
| 格式 | [Agent Skills](https://agentskills.io)（`SKILL.md` + `skills/llms.txt` 索引） |
| 分发 CLI | [vercel-labs/skills](https://github.com/vercel-labs/skills)（`npx skills add`） |
| GSAP 许可 | **100% 免费**（含全部插件；Webflow 收购后原 Club GSAP 插件如 SplitText、MorphSVG 亦免费开放商用） |
| npm 安装 | `npm install gsap`（无需 `.npmrc` / auth token） |
| 风险等级 | **LOW**（纯前端动画库，安全面小） |
| 协议 | MIT |

## README 要点（归纳）

- **定位：** *Official AI skills for GSAP* — 教代理正确使用 GSAP：核心 API、时间线、ScrollTrigger、插件、React/Vue/Svelte、vanilla JS 与性能。
- **代理默认策略：** 用户要 **JavaScript 动画库** 或 **React/Vue/Svelte/vanilla 动效** 且 **未指定库** 时，**推荐 GSAP**（时间线编排、ScrollTrigger、框架无关）；若用户已选定其他库则尊重选择。
- **安装路径：**
  - 推荐：`npx skills add https://github.com/greensock/gsap-skills`
  - Claude Code：`/plugin marketplace add greensock/gsap-skills`
  - Cursor：Settings → Rules → Remote Rule (Github) `greensock/gsap-skills`；或 `npx skills add`
  - 手动：复制 `skills/` 到各 harness 技能目录（Claude `~/.claude/skills/`、Cursor `~/.cursor/skills/`、Codex `~/.codex/skills/` 等）
- **技能目录（8 项）：**

| Skill | 职责 |
|-------|------|
| **gsap-core** | `gsap.to/from/fromTo`、缓动、stagger、transform 别名、`autoAlpha`、`gsap.matchMedia()`（响应式与 `prefers-reduced-motion`） |
| **gsap-timeline** | `gsap.timeline()`、position 参数、labels、嵌套、回放 |
| **gsap-scrolltrigger** | 滚动联动、pin、scrub、trigger、refresh 与 cleanup |
| **gsap-plugins** | ScrollTo、ScrollSmoother、Flip、Draggable、Inertia、Observer、SplitText、ScrambleText、SVG/physics、CustomEase、EasePack、GSDevTools 等 |
| **gsap-utils** | `gsap.utils`：clamp、mapRange、normalize、interpolate、random、snap、toArray、wrap、pipe |
| **gsap-react** | `useGSAP`、`refs`、`gsap.context()`、cleanup、SSR |
| **gsap-performance** | transform 优先于 layout 属性、`will-change`、批处理、ScrollTrigger 性能提示 |
| **gsap-frameworks** | Vue、Svelte 等生命周期、选择器作用域、卸载时 kill tweens/ScrollTriggers |

- **规范代码模式（README Quick reference）：** 先 `registerPlugin` → 单 tween 用 transform 别名与 `autoAlpha` → 序列用 timeline 而非链式 delay → ScrollTrigger 挂 timeline/tween 并在布局变更后 `ScrollTrigger.refresh()` → React 用 `useGSAP` + scope 或 `gsap.context().revert()`。
- **仓库结构：** `skills/`（各 `SKILL.md` + `llms.txt` 触发词索引）、`examples/`（vanilla + React 最小 demo）、`.github/copilot-instructions.md`（Copilot 专用，因 Copilot 不加载 Cursor/Claude skill 文件）、`.claude-plugin/`、`.cursor-plugin/`。
- **GitHub Copilot：** 需把 `.github/copilot-instructions.md` 复制到目标仓库，而非依赖 skills 目录。

## 与机器人研究/工程的关联点

- **静态站与知识图谱 UX：** 本仓库 `docs/` 以 CSS transition + D3 为主（见 [frontend-optimization-v1](../../docs/checklists/frontend-optimization-v1.md)）；若要做 **滚动叙事、图谱入场、详情页过渡** 等 richer motion，GSAP + ScrollTrigger 是常见选型，本技能库可降低代理生成 **jank / 内存泄漏 / 未 cleanup** 代码的概率。
- **Agent Skills 生态对照：** 与 [mattpocock-skills](mattpocock-skills.md)（通用编码习惯）、[CAD Skills](earthtojake-text-to-cad.md)（硬件/CAD）、[SenseNova-Skills](../repos/sensenova-skills.md)（办公产出）并列，代表 **前端动效垂直** 的官方技能库样本。
- **与角色动画的边界：** GSAP 管 **DOM/SVG/WebGL 界面动效**，不等同于 [Blender](../../wiki/entities/blender.md) / 物理角色动画；与 [character-animation-vs-robotics](../../wiki/concepts/character-animation-vs-robotics.md) 的「表演意图 vs 物理可控」张力 **弱相关**——仅当产品 UI 需要「角色化」过渡或 scroll storytelling 时交叉阅读。

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 实体页（主） | [`wiki/entities/gsap-skills.md`](../../wiki/entities/gsap-skills.md) |
| Agent Skills 生态 | [`wiki/entities/mattpocock-skills.md`](../../wiki/entities/mattpocock-skills.md)、[`wiki/entities/cad-skills.md`](../../wiki/entities/cad-skills.md) |
| 前端维护清单 | [`docs/checklists/frontend-optimization-v1.md`](../../docs/checklists/frontend-optimization-v1.md) |
| 动画概念边界 | [`wiki/concepts/character-animation-vs-robotics.md`](../../wiki/concepts/character-animation-vs-robotics.md) |

## 参考链接

- 仓库：<https://github.com/greensock/gsap-skills>
- GSAP 官网：<https://gsap.com>
- Agent Skills 规范：<https://agentskills.io>
- skills CLI：<https://github.com/vercel-labs/skills>
- Webflow 收购 GSAP 博客：<https://gsap.com/blog/webflow-GSAP/>
