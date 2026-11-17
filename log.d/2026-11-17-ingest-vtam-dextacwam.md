## [2026-11-17] ingest | VTAM v2 与 DexTacWAM 视触觉 WAM 更新

- 意图：依据两篇 arXiv 论文与作者官方实现重新核对触觉世界建模、部署推理与开源边界，更新已有节点而不重复建页。
- 论文：VTAM arXiv:2603.23481v2（2026-11-05）更新了标题、UniVTAC 八任务、Hanoi 跨 Xense 传感器与延迟消融；DexTacWAM arXiv:2609.24976v1 保留已有完整原文和源码归档。
- 开放核查：VTAM 训练/推理代码与 xArm+GelSight 数据已发布，UniVTAC 仿真配置及任务 checkpoint 仍未开放；DexTacWAM 三阶段实现与 Stage 1 权重/数据发布，任务 Stage 2/3 checkpoint 不提供。
- 知识结构：升级 VTAM 唯一论文实体，补项目流程与真实源码运行时序；DexTacWAM 与 VTAM 双向对照；接入 WAM / 接触操作 Stage 阅读链。
- 风险边界：区分 VTAM 三任务成功率（90%）与 DexTacWAM 六任务综合评分（70.6），不跨任务排名；两仓均存在非商业许可组件。
