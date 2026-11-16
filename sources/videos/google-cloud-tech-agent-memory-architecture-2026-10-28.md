# Google Cloud Tech：AI Agent Memory Architecture（Fashion Mind 演示线索）

- **类型：** video / Google Cloud Tech 官方视频与演示代码短链接
- **英文标题：** AI Agent Memory Architecture: Structured SQL, Semantic Vector Search & Memory Bank
- **中文标题：** AI Agent 记忆架构：结构化 SQL、语义向量检索与 Memory Bank
- **发布方：** Google Cloud Tech
- **主讲人：** Annie Wang（见视频索引资料；官方字幕尚未核验）
- **视频发布日期：** 2026-10-28（公开的视频索引标注，待与官方播放页核对）
- **官方视频：** <https://www.youtube.com/watch?v=8C4bh59FtsI>
- **用户提供的代码短链接：** <https://g.dev/ai/fashion-mind>
- **短链接可见跳转目标：** <https://me.developers.google.com/url/ai/fashion-mind>（当前抓取工具无法打开，不等于链接失效）
- **入库日期：** 2026-11-16
- **核对范围：** 已核对上述视频标题、Google Cloud Tech 署名及公开索引中的视频说明，并尝试打开用户提供的短链接。尚未直接读取官方视频完整字幕、短链接最终目标或任何实际源码；下述内容仅限视频说明明确列出的主题，不等于完成原文/源码核验。
- **通读：** 未完成
- **代码状态：** **待核验**。视频说明将短链接称为代码入口，但尚未取得最终 GitHub / Colab / Codelab 地址，不能据此声称代码已公开、可运行或已确认许可。
- **关联知识页：** [Agent 记忆与 dreaming](../../wiki/concepts/agent-memory-and-dreaming.md)

## 已确认的主题（据视频公开说明，非源码分析）

- **短期与长期：** 区分会话期间的短期记忆与跨会话保留的长期记忆。
- **结构化事实与语义理解：** 精确事实查询应使用适合确定性检索的结构化存储（示例包括 PostgreSQL、MySQL、Cloud SQL），偏好、趋势等语义关联可借助 pgvector、AlloyDB 或 Vertex AI Agent Engine Memory Bank。
- **多模态边界：** 原始相机画面和音频波形不自动等价于可复用的语义记忆；视频讨论如何处理感知结果。
- **时延路径：** 介绍面向实时语音 agent 的双轨读取/写入：事实可通过同步工具调用访问；语义抽取应考虑在时延敏感路径之外异步执行。

这些是**架构教学主题**，不是该演示工程的已验证运行机制。当前不写实现入口、具体 API、性能数字、模型版本或运行时序图。

## 与机器人知识库的关系

这条资料可作为 **具身 Agent 长时记忆设计** 的跨领域参照：机器人中的精确任务状态、物体 ID、时间戳与可模糊检索的历史偏好/经验是否应分存，需要按实际系统验证。视频未提供可核查的机器人真实部署数据；不可将通用 Agent 架构直接当作机器人端性能结论。

## 后续核验清单

1. 打开短链接的最终目标，确定仓库拥有者、源码目录、许可与运行命令。
2. 完整阅读官方视频/字幕及与 Fashion Mind 演示对应的主要文档。
3. 若有可运行代码，阅读真实入口、数据库 schema、记忆读写模块，再补来源归档、Mermaid 项目流程图与源码运行时序图。
4. 核对项目是否已被本库独立收录；如适合，再建立唯一实体节点并完整运行健康度与 CI 检查。

## 外部来源

- [Google Cloud Tech 官方视频入口](https://www.youtube.com/watch?v=8C4bh59FtsI)
- [用户提供的代码入口](https://g.dev/ai/fashion-mind)
- [视频标题、描述及时间索引（第三方镜像，仅作定位）](https://zolotube.com/watch?v=8C4bh59FtsI)
