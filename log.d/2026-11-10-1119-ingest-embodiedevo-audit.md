## [2026-11-10] ingest | EmbodiedEvo 原文与源码深读复核

- **意图：** 不重复创建既有 EmbodiedEvo 节点；按仓库最新规范登记 arXiv v2 全文通读及附录 A–F，修正 Playground / Test 只读边界，核对实际代码入口与仿真调用。
- **一手来源：** https://arxiv.org/html/2610.18867v2 ; https://lab.einsia.ai/embodiedevo/ ; https://github.com/Einsia/EmbodiedEvo
- **关键补充：** 区分 full 2,100 与配对 210 episode；+41.3%（相对无经验）与 -53.0%（相对原始历史）的不同成本口径；跨任务与未成功 Playground 的正面与局限案例。
- **开源边界：** 源码可访问，但项目级许可证未选定；仿真资源/GPU/模型服务另配。未本地复现 RoboDojo-Sim。
- **落地节点：** [EmbodiedEvo](../wiki/entities/paper-embodied-evo-coding-agent.md)。
