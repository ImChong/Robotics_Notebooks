# Three.js Game Skills（majidmanzarpour/threejs-game-skills）

> 来源归档

- **标题：** Three.js Game Skills
- **类型：** repo（Agent Skills + Vite/TypeScript 游戏脚手架 + QA 脚本）
- **作者：** Majid Manzarpour
- **链接：** https://github.com/majidmanzarpour/threejs-game-skills
- **安装：** `npx skills add majidmanzarpour/threejs-game-skills --skill '*' -a codex -g -y`（Claude Code 将 `-a claude-code`）
- **入库日期：** 2026-09-07
- **一句话说明：** 面向 **可玩、可发布** 的浏览器 Three.js 游戏的 **九件套 Agent Skills**：`threejs-game-director` 自动路由玩法、AAA 画面、UI、Tripo/Gemini/ElevenLabs 资产生成、调试与 Playwright QA；内置 Vite+TS 脚手架、确定性测试钩子与证据清单脚本。
- **开源状态：** **已开源** — MIT；主仓含 `skills/**/SKILL.md`、`install.sh`、`scripts/` 校验与 helper 单测；可选 Tripo / Gemini / ElevenLabs API 为闭源付费服务，需环境变量。
- **沉淀到 wiki：** 是 → [`wiki/entities/threejs-game-skills.md`](../../wiki/entities/threejs-game-skills.md)

## 为何值得保留

- **Agent Skills × 完整游戏交付管线：** 与 [img2threejs](img2threejs.md)（单图→程序化 Three.js 工厂）、[GSAP Skills](greensock-gsap-skills.md)（DOM 动效）、[image-blaster](image-blaster.md)（单图→Marble splat + Hunyuan 资产包）并列，代表 **「从零到可发布浏览器游戏」** 的垂直技能包——导演技能自动拉齐玩法、画面、UI、资产生成与 QA，而不是让用户手动选九个专家技能。
- **证据驱动完成标准可读：** README 要求 `npm run build`、浏览器运行、Playwright 截图、canvas 非空像素、viewport、bot playtest、visual scorecard 等 **按改动范围缩放** 的验证清单；`check_evidence.py --manifest` 与 `probe_asset_credentials.sh` 把「声称 premium/AAA」与可审计 artifact 绑定。
- **对本站维护者的交叉价值：** 若 `docs/` 静态站需要 **WebGL 小游戏 demo、交互原型或 Three.js 可视化 sandbox**，本技能包的 **Vite 脚手架 + 测试钩子** 是可复用参照；与机器人仿真栈无直接耦合，但同属 **编码代理 + 浏览器 3D** 生态。

## README 要点（归纳，2026-09-07）

| 字段 | 值 |
|------|-----|
| 托管 | GitHub |
| Stars | ~1.8k（入库日） |
| 格式 | [Agent Skills](https://agentskills.io)（9× `SKILL.md` + `references/` + `scripts/`） |
| 分发 CLI | `npx skills add majidmanzarpour/threejs-game-skills`；本地 `./install.sh --codex|--claude|--all` |
| 运行时脚手架 | `skills/threejs-gameplay-systems/assets/threejs-vite-game/`（Vite + TS + Playwright 模板） |
| 核心入口 | `threejs-game-director` — 路由 sibling skills，按用户 scope 缩放验证强度 |
| 协议 | MIT |

### 九项技能分工

| Skill | 职责 |
|-------|------|
| **threejs-game-director** | 端到端游戏：scope、专家路由、委托、连续性笔记、证据清单 |
| **threejs-gameplay-systems** | 可玩循环、架构、关卡/遭遇、输入、相机、物理选型、打击感 |
| **threejs-aaa-graphics-builder** | 视觉记分卡、技术美术预算、材质/VFX/光照、渲染抛光 |
| **threejs-game-ui-designer** | HUD、菜单、响应式/触控 UI、安全区、图标与文字适配 |
| **threejs-debug-profiler** | 黑屏、运行时错误、移动端、性能与 draw call/纹理/内存 |
| **threejs-qa-release** | 生产构建、截图、canvas 像素指标、bot playtest、发布风险 |
| **threejs-3d-generator** | Tripo API：文/图→3D、纹理、rig、动画、GLB/FBX |
| **threejs-image-generator** | Gemini：概念图、纹理、天空、图标、GUI 艺术、图生 3D 输入 |
| **threejs-audio-generator** | ElevenLabs：SFX、环境音、UI 音、TTS/配音 |

### 可选 API（非必需）

| Provider | 环境变量 | 用途 |
|----------|----------|------|
| Tripo | `TRIPO_API_KEY` | 英雄载具、武器、建筑、生物等高质量 3D |
| Gemini | `GEMINI_API_KEY` | 概念图、纹理、天空、图标、图生 3D 源图 |
| ElevenLabs | `ELEVENLABS_API_KEY` | SFX、环境循环、UI 反馈、配音 |

缺 key 时导演技能 **继续用程序化/本地资产** 并报告 `probe_asset_credentials.sh` 输出，不静默降级为「已完成 premium 3D」。

### 演示游戏（README Demos）

| 游戏 | 在线 |
|------|------|
| Neon Ridge Drift | [ridgedrift.netlify.app](https://ridgedrift.netlify.app) |
| Championship Snooker Arena | [snookerarena.netlify.app](https://snookerarena.netlify.app) |
| Starship Dogfight | [starshipdogfight.netlify.app](https://starshipdogfight.netlify.app) |
| Tide Singer | [tidesinger.netlify.app](https://tidesinger.netlify.app) |
| Ripcore | [ripcore.netlify.app](https://ripcore.netlify.app) |

### 打包资源（自包含）

- 每项技能自带 `SKILL.md`、`references/`、`scripts/`、`assets/`；**不依赖** 仓库根目录文档。
- `create_threejs_game.py` — 从空目录生成 Vite 游戏（含 `__THREE_GAME_TEST_HOOKS__`、seeded RNG、`tests/` 模板）。
- `inspect-threejs-canvas.mjs` — canvas 像素指标与 render budget。
- `check_evidence.py --manifest` — 校验声明的截图/状态捕获与 run ID。

## 与机器人研究/工程的关联点

- **浏览器 3D 原型层：** 适合 **产品 demo、遥操作 UI 小游戏化、数据可视化 sandbox**；**不是** Isaac/MuJoCo 仿真资产或 URDF 操作链（对照 [CAD Skills](../../wiki/entities/cad-skills.md)、[img2threejs](../../wiki/entities/img2threejs.md)）。
- **Agent Skills 生态：** 与 [mattpocock-skills](mattpocock-skills.md)、[Superpowers](../../wiki/entities/superpowers-obra.md) 同属 **可安装 `SKILL.md` 规约**；本仓是 **游戏垂直 + 内置 QA 证据** 的完整样本。
- **生成式 3D 资产消费：** Tripo/Gemini/ElevenLabs 与 [image-blaster](image-blaster.md)（Marble + FAL）同属 **API 闭源生成**；本仓更聚焦 **可玩循环 + 发布验证**，而非 splat 世界包。

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 实体页（主） | [`wiki/entities/threejs-game-skills.md`](../../wiki/entities/threejs-game-skills.md) |
| 程序化 Three.js 对照 | [`wiki/entities/img2threejs.md`](../../wiki/entities/img2threejs.md) |
| Web 动效 Agent Skills | [`wiki/entities/gsap-skills.md`](../../wiki/entities/gsap-skills.md) |
| 单图→世界资产包 | [`wiki/entities/image-blaster.md`](../../wiki/entities/image-blaster.md) |
| Agent Skills 通用对照 | [`wiki/entities/mattpocock-skills.md`](../../wiki/entities/mattpocock-skills.md) |

## 参考链接

- 仓库：<https://github.com/majidmanzarpour/threejs-game-skills>
- Agent Skills 规范：<https://agentskills.io>
- skills CLI：<https://github.com/vercel-labs/skills>
- Tripo API：<https://platform.tripo3d.ai/docs/quick-start>
- Gemini API keys：<https://ai.google.dev/gemini-api/docs/api-key>
- ElevenLabs API：<https://elevenlabs.io/docs/eleven-api/quickstart>
