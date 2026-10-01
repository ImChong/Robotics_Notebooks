# HyperFrames（heygen-com/hyperframes）

> 来源归档

- **标题：** HyperFrames
- **类型：** repo（HTML/CSS/动画 → 确定性 MP4；Agent Skills + CLI）
- **作者：** HeyGen（heygen-com）
- **链接：** https://github.com/heygen-com/hyperframes
- **文档 / Playground：** https://hyperframes.heygen.com/ · https://www.hyperframes.dev/
- **npm：** `hyperframes`（Node ≥22）
- **许可：** Apache-2.0
- **入库日期：** 2026-10-01
- **Trendshift（用户触发，2026-10）：** 约 **+11.3k stars/月**；GitHub API 2026-10-01 约 **54.8k** stars
- **一句话说明：** **写 HTML 就能渲染视频**：可 seek 的动画 + 媒体管线 lint/preview/render 成 MP4；为 **编码 Agent** 提供 **21 个 skills**（`/hyperframes` 路由 + 域技能）与 Claude/Cursor/Codex 等插件；与 [Archify](../../wiki/entities/archify.md)（静态系统图 HTML）形成 **动效视频 vs 可校验架构图** 分工。
- **为什么值得保留：** Demo 视频、课程片段、机器人实验回放等 **Agent 可编程视频工件** 的新默认栈之一。
- **沉淀到 wiki：** 是 → [`wiki/entities/hyperframes.md`](../../wiki/entities/hyperframes.md)

## README 要点（归纳）

- **Agent 安装：** `claude plugin marketplace add heygen-com/hyperframes`；或 `npx skills add heygen-com/hyperframes`（Core Skills 组）；`npx hyperframes skills update` 从 main 同步最新 skill。
- **生产环：** plan → HTML → seekable animations → media → lint → preview → render。
- **插件 vs standalone：** 插件走 agent 更新管理；standalone 默认 lean core set，按需拉 workflow skill。

## 开源状态

- **已开源（Apache-2.0）** — CLI、skills、文档站；HeyGen 托管 authoring 为商业延伸。

## 对 wiki 的映射

| 目标 | 链接 |
|------|------|
| 实体页 | [`wiki/entities/hyperframes.md`](../../wiki/entities/hyperframes.md) |
| 静态图 Skill | [`wiki/entities/archify.md`](../../wiki/entities/archify.md) |
