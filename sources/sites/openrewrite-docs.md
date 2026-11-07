# OpenRewrite — 官方文档归档

- **类型：** site / 官方技术文档
- **项目：** OpenRewrite
- **官网：** <https://docs.openrewrite.org/>
- **源码：** <https://github.com/openrewrite/rewrite>
- **核对日期：** 2026-11-07
- **核对范围：** 通读官方 Introduction、Lossless Semantic Trees、Recipes、Visitors、Licensing、Quickstart 中与工作原理、执行方式、许可证和依赖仓库配置相关的正文及示例；未覆盖整站 Recipe catalog。
- **通读：** 已完成
- **开源状态：** 源码公开；核心大部分 Apache-2.0，部分语言模块与 Recipes 为 Moderne Source Available License，另有商用专有 Recipe。
- **实体：** [OpenRewrite](../../wiki/entities/openrewrite.md)
- **仓库归档：** [openrewrite/rewrite](../repos/openrewrite-rewrite.md)

## 官方原文入口

- [Introduction](https://docs.openrewrite.org/)
- [Lossless Semantic Trees](https://docs.openrewrite.org/concepts-and-explanations/lossless-semantic-trees)：带类型信息且保留空白/格式的源码表示；本地运行每次重新构造内存中的 LST。
- [Recipes](https://docs.openrewrite.org/concepts-and-explanations/recipes)：命令式 Java Recipe、声明式 YAML Recipe、组合 Recipe，`getVisitor()` 实施具体编辑。
- [Visitors](https://docs.openrewrite.org/concepts-and-explanations/visitors)：`TreeVisitor.visit`、`Cursor`、语言专用 visitor；不要绕开通用 visit 生命周期。
- [Quickstart](https://docs.openrewrite.org/running-recipes/getting-started)：插件启用、选择 Recipe、dry run 与实际修改。
- [Licensing](https://docs.openrewrite.org/licensing/openrewrite-licensing)：分层许可证说明。

## 经核实的事实

1. OpenRewrite 本地创建 LST → 运行 Recipe/Visitor → 按修改结果重新打印源码；已更改的文件才需要写回，下一次本地运行会重新解析。
2. LST 相比单纯 AST 更注重类型归属与格式保留，因此可以有条件地匹配语义代码模式。
3. Recipe 可以独立执行、组合执行、配置 YAML 声明式序列，遍历和变换工作由 Visitor 完成。
4. 官方 Quickstart 当前提及 Code Genome Project 制品仓库及认证令牌要求。不同 Recipe 对应的许可和可下载权限不同；配置时要保留 Maven Central 供其他依赖解析，凭证不可提交进代码库。
5. 官方声明本地 LST 需要装入内存；Moderne 的跨仓库规模化运行及持久化缓存不应当作开源引擎单仓插件的默认能力。

## 使用边界

此次未实机执行示例，也未验证所有语言解析器/Recipe 的许可明细及可用发行版本；以实际所需模块的许可证、仓库和官方当前版本页为准。
