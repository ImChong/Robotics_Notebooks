# [2026-11-08] ingest | Recova 既有节点原文复核与真机编排补全

- **去重：** `wiki/entities/paper-recova.md` 已存在且 `project_id: recova`；本次不新建重复实体。
- **官方原文：** https://arxiv.org/html/2610.01178v1 （论文 §1–5、附录 A–E、Tables 1–5、Figures 1–10）；https://research.nvidia.com/labs/gear/recova/
- **更新节点：** `wiki/entities/paper-recova.md`，补 Appendix A Algorithm 1 对应的 Mermaid 真机调度时序（**论文伪代码，不是源码时序**）、四任务逐项 20-trial 成功率、异步 VLM 判定与过期回调防护、训练动作 offset/absolute 语义、技能计数原文差异。
- **更新原文归档：** `sources/papers/recova_arxiv_2610_01178.md`，逐条记录对照章节、实验统计和未开放源码边界。
- **数据核验：** Base 23.8% → DAgger 77.5% → 恢复 87.5%；摸牌任务四轮人工接管 7/8 → 0/7；不能将后者推广到所有真机任务。
- **论文内部统计差异：** §4.2 报告 25 recovery skills；附录 D 图 8–9 展示 24 项。本次不推测缺少的一项。
- **健康度目标：** 沿用原节点 `type/project_id/tags/status/summary/original/sources` 和九节阅读骨架，`original` 指向标有「通读：已完成」的现有来源；按仓库三项静态检查预期 3/3，需 Actions 最终复核。
- **发布边界：** 没有找到官方独立完整源码/权重/数据下载入口，不虚构源码调用或复现结果；不更新自动生成的 `catalog.md`、`log.md` 或站点导出物。
