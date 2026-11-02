# ub-mode（unclebob/ub-mode）

> 来源归档（ingest）

- **类型：** GitHub 仓库 / 编码代理技能
- **仓库：** <https://github.com/unclebob/ub-mode>
- **已核对原文：** [README.md](https://github.com/unclebob/ub-mode/blob/master/README.md)（blob `c54164fb5a21a0f917e081fd54c959d4d235ea01`）；[SKILL.md](https://github.com/unclebob/ub-mode/blob/master/SKILL.md)（blob `9ac940b23b61d77e8e5192721d7b6066357f892c`）
- **入库日期：** 2026-11-02
- **开放状态：** 仓库公开，包含 README 与技能文本；配套工具位于独立仓库。
- **许可：** 所核对的仓库文件列表未见 LICENSE，不推断授权条款。
- **一句话说明：** 以三阶段原则指导编码代理，并提出使用独立质量工具检查复杂度、重复和变异测试。
- **沉淀到 wiki：** [ub-mode](../../wiki/entities/ub-mode.md)

## 原文要点

1. First Make it Work：最小改动、数据形状、TDD、架构边界、幂等和共享状态隔离等。
2. Then Make it Right：复现并修复根因、以可验证的小单元推进、测试外部行为；用 Crapper、Dryer、Mutator 辅助检查，再重构。
3. Practice：记录操作及原因、控制上下文、清理工作树、以小而有序的提交推进。
4. README 指示把 SKILL.md 放进代理的技能目录并调用 `/ub-mode`；实际发现及执行方式取决于代理环境。

## 配套仓库

- [Crapper](https://github.com/unclebob/crapper)：复杂度、覆盖率与 CRAP 分数。
- [Dryer](https://github.com/unclebob/dryer)：结构重复检测。
- [Mutator](https://github.com/unclebob/mutator)：变异测试。
- [UML Viewer](https://github.com/unclebob/uml-viewer)：可视化组件及质量指标。

这些工具不包含在 ub-mode 仓库内；本次未验证它们的安装、语言支持或实际运行效果。
