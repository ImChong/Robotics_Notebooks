# i-have-adhd（ayghri/i-have-adhd）

> 来源归档

- **标题：** i-have-adhd
- **类型：** repo（Agent Skill / 输出格式契约）
- **作者：** ayghri
- **链接：** https://github.com/ayghri/i-have-adhd
- **入库日期：** 2026-10-01
- **Trendshift（用户触发，2026-10）：** 约 **+26.8k stars/月**；GitHub API 2026-10-01 约 **52.4k** stars
- **一句话说明：** 编码代理技能：用 **10 条 ADHD 友好规则** 约束回复结构——**结论/下一步先行**、多步编号、抑制 tangent、每轮重述状态、分钟级时间估计、可见小胜利；不要求用户有 ADHD 诊断。
- **为什么值得保留：** 与 [Caveman](caveman.md)（压缩措辞 token）、[Ponytail](ponytail.md)（减 LOC）正交，专注 **可读性与行动导向**；适合长 ingest / 多步 CI 会话里减少「答案埋在废话里」。
- **沉淀到 wiki：** 是 → [`wiki/entities/i-have-adhd.md`](../../wiki/entities/i-have-adhd.md)

## README 要点（归纳）

- **安装：** 复制提示安装 skill，或见 `INSTALL.md`；技能路径 `skills/i-have-adhd/SKILL.md`。
- **Before/After 示例：** 从长段解释改为「先给命令 + 编号步骤 + 单一 next step」。
- **规则摘要：** Lead with action；Number steps；One concrete next step；Suppress tangents；Restate state；Specific time estimates；Make wins visible 等（完整 10 条见 SKILL.md）。
- **多语言 README** 在 `.github/readme/`。

## 开源状态

- **已开源** — 仓库即技能源；以 `LICENSE` badge 为准。

## 对 wiki 的映射

| 目标 | 链接 |
|------|------|
| 实体页 | [`wiki/entities/i-have-adhd.md`](../../wiki/entities/i-have-adhd.md) |
| 输出压缩对照 | [`wiki/entities/caveman.md`](../../wiki/entities/caveman.md) |
