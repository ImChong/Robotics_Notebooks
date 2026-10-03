# ASD-STE100 Skill（danyuchn/asd-ste100-skill）

- **标题：** ASD-STE100 Skill — Simplified Technical English for Agent Output
- **类型：** repo
- **作者：** danyuchn
- **仓库：** <https://github.com/danyuchn/asd-ste100-skill>
- **许可证：** MIT（仓库 LICENSE）
- **入库核查：** 2026-10-03；公开 GitHub 仓库，含 `SKILL.md`、示例、规则参考和结构检查脚本。
- **独立详情节点：** [ASD-STE100 Skill](../../wiki/entities/asd-ste100-skill.md)
- **相关标准：** [ASD-STE100 官方站归档](../sites/asd-ste100.md)

## README 摘要

该项目把 ASD-STE100 的受控语言原则整理为 Claude Code Skill，用于澄清面向代理的英文文本，例如工具描述、错误消息、系统提示与代理间指令。它提供 **Strict** 和 **STE-flavored** 两种模式：前者用于误读代价较高的程序步骤等内容；后者保留短句、主动表达和清晰结构，但不锁定词汇表。

仓库还提供确定性结构检查脚本。其 README 明确说明：检查器只覆盖选定的结构模式，不验证改写是否保留全部语义，也不复现官方词典。

## 阅读入口

- [README](https://github.com/danyuchn/asd-ste100-skill#readme)
- [SKILL.md](https://github.com/danyuchn/asd-ste100-skill/blob/master/SKILL.md)
- [规则说明](https://github.com/danyuchn/asd-ste100-skill/blob/master/references/writing-rules.md)
- [ASD-STE100 官方站](https://www.asd-ste100.org/)
