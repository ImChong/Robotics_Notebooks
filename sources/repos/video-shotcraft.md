# video-shotcraft（Vincentwei1021/video-shotcraft）

> 来源归档

- **标题：** video-shotcraft
- **类型：** repo
- **来源：** Vincentwei1021（个人；Trendshift 榜单上榜仓库）
- **链接：** https://github.com/Vincentwei1021/video-shotcraft
- **项目页 / Gallery：** https://vincentwei1021.github.io/video-shotcraft/
- **安装：** `npx skills add Vincentwei1021/video-shotcraft`
- **入库日期：** 2026-07-31
- **一句话说明：** 面向 Claude Code / Codex 的 **Agent Skill**：把镜头配方卡、Remotion 参考实现、音频资产与六/八阶段制作流水线打包成可安装技能，让代理完成产品宣传片的分镜、动画与音效设计。
- **为什么值得保留：** 与 [GSAP Skills](greensock-gsap-skills.md)、[img2threejs](img2threejs.md)、[mattpocock-skills](mattpocock-skills.md) 同属 **垂直领域 Agent Skills**；本库把 **电影感产品视频（Remotion + 2.5D 运镜 + 节拍卡点）** 写成可执行规约，对本站维护者做 demo/宣传片、以及理解「配方卡 + 已验收模板 + 独立终检」的代理交付范式有直接对照价值。
- **沉淀到 wiki：** 是 → [`wiki/entities/video-shotcraft.md`](../../wiki/entities/video-shotcraft.md)
- **地址更正：** 用户触发路径写作 `trendshift/video-shotcraft`（**404**）；核实为 [Trendshift](https://trendshift.io/repositories/88911) 榜单徽章/入口，官方 GitHub 仓为 [`Vincentwei1021/video-shotcraft`](https://github.com/Vincentwei1021/video-shotcraft)。

## 开源状态（项目页 + 仓库核查，2026-07-31）

- **代码：** 已开源 — Apache-2.0，GitHub 完整仓（`SKILL.md`、`references/`、`demos/`、`template/`、`gallery/`、`assets/`）。
- **Gallery：** 已部署 — [vincentwei1021.github.io/video-shotcraft](https://vincentwei1021.github.io/video-shotcraft/)（GitHub Pages；样片 mp4 自 2026-07-26 起移至 release，瘦身主仓）。
- **运行时依赖：** 渲染依赖 [Remotion](https://www.remotion.dev/)（**自有许可**：个人/小团队免费，公司可能需付费）；音频素材多来自 Mixkit（见仓内 `assets/audio/ATTRIBUTION.md`）。
- **结论：已开源**（技能规约 + Remotion 模板/demo + Gallery 站点 + 音频资产）；成片质量仍依赖宿主代理、浏览器渲染与产品素材脱敏。详见 [`sources/sites/video-shotcraft-gallery.md`](../sites/video-shotcraft-gallery.md)。

## 仓库概况（2026-07-31 README / API）

| 字段 | 值 |
|------|-----|
| 托管 | GitHub |
| Stars / Forks | ~2984 / ~257（API 快照） |
| 语言 | TypeScript |
| 协议 | Apache-2.0（仓库本体）；Remotion / Mixkit 另有条款 |
| 格式 | [Agent Skills](https://agentskills.io)（根目录 `SKILL.md` + `references/`） |
| 分发 CLI | [skills.sh](https://skills.sh/) / `npx skills add Vincentwei1021/video-shotcraft` |
| 锁定计数（README_CN / 树扫描） | **104** 张镜头配方卡（`references/shots/`）；**161** 个样式 / **161** 条动态样片（Gallery）；~153 个 `.tsx` demo；**Ink Press** 模板 36.2s · 1920×1080 · 30fps · 10 镜头；音频 `bgm/` 5 首 + `sfx/` 149 条（16 类） |
| 说明 | 部分外链/榜单文案写「106 卡 · 162 样式」；**以入库日官方 README 与仓库树为准（104 / 161）** |

## README / SKILL 要点（归纳）

- **定位：** 把 Claude Code / Codex 变成动效工作室——真实页面截图、2.5D 运镜、节奏卡点、电影级 SFX；当前主场景是 **Web/桌面产品宣传片**，镜头卡也可单独用于单镜头动效。
- **三种完整宣传片模式（互不合并）：**
  1. **Ink Press 模板** — 换截图/文案/品牌，最快出片；
  2. **自主自由创作** — 读 `references/pipeline.md`，Agent 连续推进阶段 0–7；
  3. **共同创作** — 读 `guided-free-creation.md`，产品简报→视觉方向→镜头映射→分镜逐级确认后再进入制作。
- **八阶段流水线：** 产品理解 → styleframe → 功能到镜头映射 → 分镜放行 → 素材采集 → 逐镜头 Remotion 实现 → 声音设计 → **独立 subagent 终检**。
- **质量硬约束（SKILL 核心理念摘要）：** 复刻页面必须真实截图；视觉语言从产品设计 tokens 生长；每镜头一种动效主角；强节奏 BGM 必须卡拍；用卡必读准确 demo TSX；确定性渲染（禁 `Date.now()`/`Math.random()`）；交付前独立终检。
- **Headless / CI：** 低核机 `--concurrency=1`；用 chrome-headless-shell；CDN 不可达时 `--browser-executable=` 指定本地二进制。

## 仓库结构（摘要）

```text
video-shotcraft/
├── SKILL.md                 # Agent 入口与核心规则
├── references/
│   ├── pipeline.md          # 八阶段流水线
│   ├── shots/               # 104 张镜头配方卡
│   ├── sequences/           # 全片结构与桥段
│   ├── aesthetic-rules.md / sound-design.md / music-beat-sync.md
│   └── guided-free-creation.md / final-review.md
├── demos/                   # 镜头卡 Remotion 参考实现
├── gallery/                 # 在线样片静态站
├── template/                # Ink Press 成片模板
└── assets/                  # Remotion 组件、采集脚本、音频
```

## 与机器人研究/工程的关联点

- **弱直接、强对照：** 本库不做机器人策略或仿真；价值在 **Agent Skills 交付范式**（配方卡 + 已验收模板 + 贯穿 QA + 独立终检）与 **产品/研究 demo 视频** 制作。
- **与角色动画边界：** Remotion 管 **程序化时间线视频**（截图纹理 + 2.5D 相机），不等同 [Manim](../../wiki/entities/manim.md) 数学片或物理角色动画；与 [character-animation-vs-robotics](../../wiki/concepts/character-animation-vs-robotics.md) 仅在「产品叙事视觉」层弱交叉。
- **与 GSAP Skills：** GSAP 技能管 **DOM/Scroll 交互动效**；video-shotcraft 管 **离线成片（Remotion）**——同属前端/运动设计技能生态，输出物不同层。

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 实体页（主） | [`wiki/entities/video-shotcraft.md`](../../wiki/entities/video-shotcraft.md) |
| Gallery 站点归档 | [`sources/sites/video-shotcraft-gallery.md`](../sites/video-shotcraft-gallery.md) |
| Agent Skills 生态 | [`wiki/entities/gsap-skills.md`](../../wiki/entities/gsap-skills.md)、[`wiki/entities/mattpocock-skills.md`](../../wiki/entities/mattpocock-skills.md)、[`wiki/entities/img2threejs.md`](../../wiki/entities/img2threejs.md) |
| 离线示意动画对照 | [`wiki/entities/manim.md`](../../wiki/entities/manim.md) |

## 参考链接

- 仓库：<https://github.com/Vincentwei1021/video-shotcraft>
- Gallery：<https://vincentwei1021.github.io/video-shotcraft/>
- Trendshift 榜单条目：<https://trendshift.io/repositories/88911>
- Remotion：<https://www.remotion.dev/>
- Agent Skills 规范：<https://agentskills.io>
- skills CLI：<https://skills.sh/>
