## [2026-11-07] ingest | sources/papers/robodoctor_arxiv_2610_35604.md — RoboDoctor 原文通读、保留抓握的局部修正与重复节点合并

- 意图：根据 <https://robodoctor-rsi.github.io/> 对照论文 Figure 4(b) 的瓶口/杯口偏移，一手通读后更新 RoboDoctor 的**单一论文/项目实体**，补 Mermaid 项目流程与已核实代码时序。
- 原文：<https://arxiv.org/html/2610.35604v1>，2026-10-26 v1；代码 <https://github.com/THUSI-Lab/RoboDoctor>（核对 main `e3ba0ea`）。
- 实质：保留稳定抓握，Doctor 横向一步修正，WAM 后续倾倒；局部对齐与任务成功分别核验。RoboDojo-Sim 54 项，Warmup 49.74→Refinement 57.10；但跨回合经验增益未经受控隔离。
- 开源：正式 Python 运行时、入口和测试已公开；模型权重与 RoboDojo 仿真环境独立获取。删除早期重复 wiki 节点并登记旧页面别名；未复制网页/论文原文，未触及 catalog.md/log.md。
