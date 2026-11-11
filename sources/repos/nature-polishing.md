# nature-polishing（Nature Skills 官方子模块）

- **类型：** repo 内独立 Agent Skill
- **上游原始仓库：** https://github.com/Yuan1z0825/nature-skills
- **唯一模块路径：** https://github.com/Yuan1z0825/nature-skills/tree/main/skills/nature-polishing
- **上游身份：** GitHub REST `fork: false`，不是独立的 nature-polishing 仓库
- **核验版本：** `a1a8220366e8`（main，2026-11-10），`manifest.yaml` 声明技能版本 `6.6.0`
- **入库日期：** 2026-11-11
- **许可：** 主仓 Apache-2.0
- **开源状态：** SKILL.md、manifest.yaml、static / references 与共享 `nature-shared` 已公开；属于宿主 Agent 调度的文本技能，不提供独立模型权重
- **核对范围：** 已全文阅读 `skills/nature-polishing/SKILL.md`、同目录 `README.md` 和 `manifest.yaml`，并阅读共享 `skills/nature-shared/core/ethics.md`。未测试实际润色样本。
- **通读：** 已完成
- **知识页：** [nature-polishing](../../wiki/entities/nature-polishing.md)

## 一手资料

- [Skill 原文](https://github.com/Yuan1z0825/nature-skills/blob/a1a8220366e8/skills/nature-polishing/SKILL.md)
- [模块 README](https://github.com/Yuan1z0825/nature-skills/blob/a1a8220366e8/skills/nature-polishing/README.md)
- [路由 manifest](https://github.com/Yuan1z0825/nature-skills/blob/a1a8220366e8/skills/nature-polishing/manifest.yaml)
- [共享学术伦理约束](https://github.com/Yuan1z0825/nature-skills/blob/a1a8220366e8/skills/nature-shared/core/ethics.md)

## 可复核结论

`nature-polishing` 首先确定 paper_type、section、language、journal 四轴，从 manifest 选择对应 `static/` 碎片与共享伦理/术语约束；仅据原始手稿做润色、缩写、翻译和版式修正，不允许新增原文缺失的结果、机制、统计显著性或引用。它明确把“从零起草章节”交给 `nature-writing`。

## 边界

“Nature 风格”不是 Nature 官方编辑背书；README 自述部分规则由发表论文语料归纳，投期刊必须对照实时官方 Author Guidelines。