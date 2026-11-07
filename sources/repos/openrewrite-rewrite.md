# OpenRewrite / rewrite — 官方源码仓库归档

- **类型：** repo / 自动化代码重构引擎
- **项目：** OpenRewrite（由 Moderne 维护）
- **代码：** <https://github.com/openrewrite/rewrite>
- **官方文档：** <https://docs.openrewrite.org/>
- **核对版本：** [08560af3c93b7a87d1cd798a97c1b212e303b16e](https://github.com/openrewrite/rewrite/tree/08560af3c93b7a87d1cd798a97c1b212e303b16e)（2026-11-07 main）
- **核对日期：** 2026-11-07
- **开源状态：** 核心框架及原有语言模块主要为 Apache-2.0；部分新增语言模块为 Moderne Source Available License；商业扩展另行授权，须按模块核对许可。
- **核对范围：** 通读仓库 README；逐段核验 `Recipe.java` 的 `run/getVisitor`、`RecipeScheduler.java` 的 `scheduleRun/runRecipeCycles`、`RecipeRunCycle.java` 的 `scanSources/generateSources/editSources/editSource`，并阅读 `ScanningRecipe.java`、`TreeVisitor.java`、`JavaParser.java` 关键接口及 `ChangeType.java` visitor 实现。未声称阅读整个大型仓库或实际运行其构建与重构任务。
- **通读：** 已完成
- **实体：** [OpenRewrite](../../wiki/entities/openrewrite.md)
- **文档归档：** [OpenRewrite Docs](../sites/openrewrite-docs.md)

## 项目定位与模块

仓库 README 把 OpenRewrite 定义为面向源码的自动重构生态：用可组合的 Recipe 对带类型信息、保留格式的 Lossless Semantic Tree（LST）执行变换，适合语言/框架升级、安全修复和风格统一。

- `rewrite-core`：Recipe、ScanningRecipe、TreeVisitor、RecipeScheduler、RecipeRunCycle、SourceFile 等引擎抽象与调度。
- `rewrite-java`：Java 解析、Java LST 和 Java 相关变换；`JavaParser` 接口提供语言解析能力。
- 其他语言解析器和 Recipe 的支持与许可各有边界，不能将整个生态一概归为 Apache-2.0。
- Maven/Gradle 插件位于配套官方插件仓库，不是该引擎源码目录中的一条 CLI 主程序。

## 源码调用路径核验

1. `Recipe.run(LargeSourceSet, ExecutionContext)` 调用 `new RecipeScheduler().scheduleRun(...)`。
2. `RecipeScheduler.runRecipeCycles(...)` 按循环创建 `RecipeRunCycle`，依次调用 `scanSources`、`generateSources`、`editSources`。
3. 扫描和生成是 `ScanningRecipe` 的特有能力，不应描述为所有 Recipe 必经的有效变换；普通 Recipe 的扫描、生成阶段可以不做事。
4. `RecipeRunCycle.editSource(...)` 的普通路径调用 `recipe.getVisitor()`，先检查 `isAcceptable` 再对单个 `SourceFile` 调用 `visitor.visit(...)`；源码还存在 RPC 批处理分支，图中只画普通路径。
5. `RecipeScheduler.scheduleRun` 返回 `RecipeRun`（含 changeset 与 data table store）；最终把修改写回磁盘的步骤由使用引擎的工具链负责，不能虚构为引擎源码某个统一的自动磁盘写入函数。
6. `ChangeType.getVisitor()` 通过类型预检查以及 Java/引用 visitor 修改类型引用；源码展示了语义匹配而非纯字符串替换。

## 源码定位（固定版本）

- [README](https://github.com/openrewrite/rewrite/blob/08560af3c93b7a87d1cd798a97c1b212e303b16e/README.md)
- [Recipe.java](https://github.com/openrewrite/rewrite/blob/08560af3c93b7a87d1cd798a97c1b212e303b16e/rewrite-core/src/main/java/org/openrewrite/Recipe.java)
- [RecipeScheduler.java](https://github.com/openrewrite/rewrite/blob/08560af3c93b7a87d1cd798a97c1b212e303b16e/rewrite-core/src/main/java/org/openrewrite/RecipeScheduler.java)
- [RecipeRunCycle.java](https://github.com/openrewrite/rewrite/blob/08560af3c93b7a87d1cd798a97c1b212e303b16e/rewrite-core/src/main/java/org/openrewrite/scheduling/RecipeRunCycle.java)
- [ScanningRecipe.java](https://github.com/openrewrite/rewrite/blob/08560af3c93b7a87d1cd798a97c1b212e303b16e/rewrite-core/src/main/java/org/openrewrite/ScanningRecipe.java)
- [TreeVisitor.java](https://github.com/openrewrite/rewrite/blob/08560af3c93b7a87d1cd798a97c1b212e303b16e/rewrite-core/src/main/java/org/openrewrite/TreeVisitor.java)
- [JavaParser.java](https://github.com/openrewrite/rewrite/blob/08560af3c93b7a87d1cd798a97c1b212e303b16e/rewrite-java/src/main/java/org/openrewrite/java/JavaParser.java)
- [ChangeType.java](https://github.com/openrewrite/rewrite/blob/08560af3c93b7a87d1cd798a97c1b212e303b16e/rewrite-java/src/main/java/org/openrewrite/java/ChangeType.java)

## 使用边界

本次为官方源码定向审读，未运行 Maven/Gradle 插件、未实测迁移成功率或执行完整单测。机器人应用场景是对训练/部署工具链里受支持的 Java 或配置文件代码进行迁移的潜在用途，非项目已经验证的 ROS 2、C++ 控制器或机器人物理仿真方案。
