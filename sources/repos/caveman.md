# Caveman（JuliusBrussee/caveman）

> 来源归档

- **标题：** caveman
- **类型：** repo
- **作者：** Julius Brussee
- **链接：** https://github.com/JuliusBrussee/caveman
- **许可：** MIT
- **入库日期：** 2026-05-19
- **一句话说明：** 面向 30+ 编码代理 harness 的可安装技能/插件：用「洞穴人」式极简输出压缩代理回复与部分上下文（`CLAUDE.md` 等），宣称平均约 **65% 输出 token 节省**且保持技术准确度；含多档压缩、`/caveman-commit`/`/caveman-review`、会话统计、MCP 描述压缩（caveman-shrink）与 cavecrew 子代理等。
- **为什么值得保留：** 与本知识库 **Karpathy LLM Wiki + schema/ingest** 维护方式强相关：长期 ingest / `make ci-preflight` 会反复读写 `AGENTS.md`、`CLAUDE.md` 与大量 markdown；caveman 直接针对 **代理输出与记忆文件 token 成本**，与 [Superpowers](obra-superpowers.md)（交付流程技能）、[Hermes Agent](nousresearch_hermes_agent.md)（常驻运行时）形成 **成本 / 流程 / 运行时** 三角对照。
- **沉淀到 wiki：** 是 → [`wiki/entities/caveman.md`](../wiki/entities/caveman.md)

## README 要点（归纳，2026-05-19）

- **定位：** *why use many token when few token do trick* — Claude Code skill/plugin，亦支持 Codex、Gemini、Cursor、Windsurf、Cline、Copilot、OpenClaw 等；一键 `install.sh` / `install.ps1`（Node ≥18）。
- **机制：** 技能指示代理去掉 filler、保留实质、用片段式表达；Claude Code 另有 **session hook** 写 flag，使会话从首条消息起即压缩，无需每次 `/caveman`。
- **压缩档位：** `lite`（去废话）、`full`（默认洞穴语）、`ultra`（电报体）、`wenyan`（文言极简）；`/caveman` 切换，`normal mode` 退出。
- **附属能力：**
  - `/caveman-commit` — Conventional Commit，subject ≤50 字符
  - `/caveman-review` — 单行 PR 评论（如 `L42: 🔴 bug: ...`）
  - `/caveman-stats` — 会话 token 统计与累计节省（Claude Code 可读 session log）
  - `/caveman-compress <file>` — 将 `CLAUDE.md` 等记忆文件改写为洞穴语（宣称 ~46% 输入 token 节省，代码/URL/路径字节级保留）
  - **caveman-shrink**（npm）— MCP 中间件，压缩工具描述
  - **cavecrew-*** — 洞穴语子代理（investigator/builder/reviewer），宣称比 vanilla 子代理少 ~60% token
- **基准：** README `benchmarks/` 表：10 条提示平均 1214→294 tokens（**65%**）；强调只减 **输出** token，thinking/reasoning 不变；并引用 2026 论文 [Brevity Constraints Reverse Performance Hierarchies](https://arxiv.org/abs/2604.00025)（简短约束有时提升准确率）。
- **生态：** cavemem（跨代理记忆）、cavekit（spec 驱动构建）、cavegemma（Gemma 微调洞穴语对）；OpenClaw 集成见 README「Lobster, Meet Rock」。
- **维护：** 详细安装矩阵见 `INSTALL.md`；hook/CI 见仓库 `CLAUDE.md`。

## 对 wiki 的映射

- 沉淀 **[`wiki/entities/caveman.md`](../../wiki/entities/caveman.md)**；安装命令与 star 数等易变指标以克隆时上游为准。
