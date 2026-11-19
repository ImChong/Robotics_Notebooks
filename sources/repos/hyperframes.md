# HyperFrames（heygen-com/hyperframes）

> 来源归档

- **标题：** HyperFrames
- **类型：** repo（HTML/CSS/动画 → 确定性 MP4；Agent Skills + CLI）
- **作者：** HeyGen（heygen-com）
- **链接：** https://github.com/heygen-com/hyperframes
- **文档 / Playground：** https://hyperframes.heygen.com/ · https://hyperframes.studio/
- **npm：** `hyperframes`（Node ≥22）
- **许可：** Apache-2.0
- **入库日期：** 2026-10-01
- **复核日期：** 2026-11-19
- **核对范围：** 官方 GitHub README（定位、Skills 当前发布集、命令、功能模块及与 Remotion 对照）、`docs/quickstart.mdx`、`packages/cli/README.md`（CLI 用法与依赖）、`packages/cli/src/cli.ts`（命令分发和动态加载入口）；核对许可入口与发布边界。
- **通读：** 已完成
- **Trendshift（用户触发，2026-10）：** 约 **+11.3k stars/月**；GitHub API 2026-10-01 约 **54.8k** stars
- **一句话说明：** **写 HTML 就能渲染视频**：可 seek 的动画 + 媒体管线 lint/preview/render 成 MP4；为 **编码 Agent** 提供 **9 个公开发布 skills**（`/hyperframes` 路由 + 域技能）与 Claude/Cursor/Codex 等插件；与 [Archify](../../wiki/entities/archify.md)（静态系统图 HTML）形成 **动效视频 vs 可校验架构图** 分工。
- **为什么值得保留：** Demo 视频、课程片段、机器人实验回放等 **Agent 可编程视频工件** 的新默认栈之一。
- **沉淀到 wiki：** 是 → [`wiki/entities/hyperframes.md`](../../wiki/entities/hyperframes.md)

## README 要点（归纳）

- **Agent 安装：** `claude plugin marketplace add heygen-com/hyperframes`；或 `npx skills add heygen-com/hyperframes`（Core Skills 组）；`npx hyperframes skills update` 按 CLI 对应的 release tag 同步 skills（不等于追踪 main）。
- **生产环：** plan → HTML → seekable animations → media → lint → preview → render。
- **插件 vs standalone：** 插件走 agent 更新管理；standalone 默认 lean core set，按需拉 workflow skill。

## 开源状态

- **已开源（Apache-2.0）** — CLI、skills、文档站；HeyGen 托管 authoring 为商业延伸。

## 对 wiki 的映射

| 目标 | 链接 |
|------|------|
| 实体页 | [`wiki/entities/hyperframes.md`](../../wiki/entities/hyperframes.md) |
| 静态图 Skill | [`wiki/entities/archify.md`](../../wiki/entities/archify.md) |


## 2026-11-19 一手资料复核

- **当前技能发布口径：** 官方 README 列 **9 个公开发布 Skills**，包含默认核心集 6 个（`hyperframes`、`hyperframes-cli`、`hyperframes-registry`、`media-use`、`reverse-any-video`、`style-sampling`）与按需 3 个（`figma`、`slideshow`、`remotion-to-hyperframes`）。另有 **6 个仓库内部 Skills**，默认不作为公开独立安装集；先前的“21 个”仅保留为历史记录，不代表目前公开技能数。
- **版本同步：** `npx hyperframes skills update` 安装**与当前 CLI 版本匹配的 release tag** 技能，不是无条件同步 GitHub `main`。
- **运行入口：** `npx hyperframes init my-video` → `cd my-video` → `npx hyperframes lint ./index.html --verbose`（或项目路径）→ `npx hyperframes preview` → `npx hyperframes render -o output.mp4`；要求 Node.js ≥22、FFmpeg。CLI 中 `packages/cli/src/cli.ts` 的 `commandLoaders` 按需加载 `commands/render.js` 等命令。渲染链路涉及 `@hyperframes/core`、`@hyperframes/engine`、`@hyperframes/producer`，README 分别标明 parser / page-to-video / capture+encode+audio mix 职责。
- **商业能力边界：** 核心框架 Apache-2.0；Studio 基础应用可免费使用，**Pro** 加入 HeyGen 生成模型；不应把这些云端增值能力写作 Apache-2.0 开源仓库的默认免费能力。
- **官方路径：** [README](https://github.com/heygen-com/hyperframes/blob/main/README.md) · [Quickstart](https://github.com/heygen-com/hyperframes/blob/main/docs/quickstart.mdx) · [CLI README](https://github.com/heygen-com/hyperframes/blob/main/packages/cli/README.md) · [CLI 入口](https://github.com/heygen-com/hyperframes/blob/main/packages/cli/src/cli.ts) · [Apache 2.0 许可证](https://github.com/heygen-com/hyperframes/blob/main/LICENSE)。
