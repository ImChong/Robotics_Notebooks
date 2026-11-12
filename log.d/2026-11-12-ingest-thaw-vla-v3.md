## [2026-11-12] ingest | THAW-VLA | 论文 v3 复核与实体健康度补全

- 原因：同一论文已经入库，不创建重复实体；将全文通读状态、项目身份和源码核对补全至既有 THAW-VLA 节点。
- 核查：arXiv:2609.24682v3、作者项目页、GitHub README 与预计算 / 蒸馏 / 评测源码、Hugging Face 模型列表。
- 结论：Cosmos3-Nano 特征离线缓存，训练使用余弦对齐损失且部署去掉辅助头；权重可公开浏览，但 README 仍写 private。
- 产物：wiki/entities/paper-thaw-vla.md，及论文/项目页/源码三份 sources 归档；保留 Mermaid 主流程图并修订源码运行时序图。
