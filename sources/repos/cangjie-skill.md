# Cangjie Skill（kangarooking/cangjie-skill）

> 来源归档

- **标题：** Cangjie Skill
- **类型：** repo
- **作者：** 袋鼠帝 kangarooking（AI 博主 / 独立开发者）
- **链接：** https://github.com/kangarooking/cangjie-skill
- **入库日期：** 2026-07-07
- **一句话说明：** 把书、长视频、播客、访谈等高价值文本蒸馏成可独立调用、可组合、可压力测试的 Agent Skills 工具包；核心流水线 **RIA-TV++**（便签拆书法 + 三重验证 + 面向 agent 的可执行步骤与边界）。
- **为什么值得保留：** 与 [nuwa-skill](nuwa-skill.md)（蒸馏人）和 [darwin-skill](darwin-skill.md)（进化 skill）构成「人 / 书 / 进化」三角生态；方法论强调 **结构化复用** 而非摘要压缩，与本站 [Karpathy LLM Wiki](../../wiki/references/llm-wiki-karpathy.md) 的「知识编译进可维护文件」同构；对维护本仓库时把论文/课程 **编译成 wiki** 有流程对照价值。
- **沉淀到 wiki：** 是 → [`wiki/entities/cangjie-skill.md`](../wiki/entities/cangjie-skill.md)

## README 要点（归纳）

- **定位：** *把书、长视频、播客里的方法论，蒸馏成可调用的 AI Skills* — 读完/看完/听完之后带走 **可触发的方法论模块**，而非躺在收藏夹。
- **RIA-TV++ 七阶段流水线：**
  1. **Adler 分析阅读** — 结构/解释/批判/应用 → `BOOK_OVERVIEW.md`
  2. **五路并行提取** — 框架、原则、案例、反例、术语
  3. **三重验证筛选** — 跨域佐证（≥2 处）、预测力（能答未明说问题）、独特性（非常识）；通过率通常 25–50%
  4. **RIA++ 构造** — R/I/A1/A2/E（可执行步骤）/B（边界与盲点）
  5. **Zettelkasten 链接** — `INDEX.md` 与 skill 间依赖/对比/组合
  6. **压力测试** — `test-prompts.json` 含诱饵题与跨 skill 混淆测试，未通过回炉
  7. **交付** — `DIGEST.md` 精华长文 + 安装到 Claude Code / Cursor skills 目录
- **命名拆解：** RIA（赵周便签拆书法）+ TV（Triple Verification）+ ++（Execution + Boundary，面向 agent）
- **输入范围：** 不仅书籍，亦适用有字幕/转写的视频、播客、课程、长文；视频建议搭配 [video-downloader](https://github.com/kangarooking/kangarooking-skills/tree/main/video-downloader) 先取文本。
- **已产出 skill packs：** 巴菲特致股东信、穷查理宝典、影响力、吴恩达 AI for Everyone 等 20+ 独立仓库（见上游 README 表）。
- **生态咬合：** [nuwa-skill](https://github.com/alchaincyf/nuwa-skill) 蒸馏人 → **cangjie** 蒸馏书/长内容 → [darwin-skill](https://github.com/alchaincyf/darwin-skill) 持续进化。
- **平台：** OpenClaw、Claude Code、Cursor 等；元规范在根 `SKILL.md`，`methodology/`、`extractors/`、`templates/` 可复用。
- **协议：** MIT。

## 对 wiki 的映射

| 主题 | 目标页面 |
|------|----------|
| 实体页（主） | [`wiki/entities/cangjie-skill.md`](../../wiki/entities/cangjie-skill.md) |
| 蒸馏人（姊妹） | [`wiki/entities/nuwa-skill.md`](../../wiki/entities/nuwa-skill.md) |
| skill 进化 | [`wiki/entities/darwin-skill.md`](../../wiki/entities/darwin-skill.md) |
| 知识编译范式 | [`wiki/references/llm-wiki-karpathy.md`](../../wiki/references/llm-wiki-karpathy.md) |
| 实验自动化对照 | [`wiki/entities/karpathy-autoresearch.md`](../../wiki/entities/karpathy-autoresearch.md) |

## 与本站 sources 的其它锚点

- 人物蒸馏姊妹：[nuwa-skill.md](nuwa-skill.md)
- skill 优化姊妹：[darwin-skill.md](darwin-skill.md)
- 编码工程技能对照：[obra-superpowers.md](obra-superpowers.md)、[mattpocock-skills.md](mattpocock-skills.md)
