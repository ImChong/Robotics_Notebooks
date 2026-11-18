## [2026-11-18] ingest | sources/papers/vino_arxiv_2611_07176.md — VINO 官方资料增量复核及跨域迁移评测

- 对照作者论文 HTML v1 的 §3–5 与 Appendix A.2/A.5，补录 3DMatch/3DLoMatch 冻结特征配准、NYUv2 单视角 RGB-D 与训练资源口径；明确配准后端与机器人部署边界。
- 核实 VINO 作者项目页、官方 GitHub：2026-11-18 仍为 README 占位，代码与权重待发布；复用唯一的论文节点，不重复创建。
- 将现有 VINO 节点接入模型架构纵深 Stage 2，与 Volt 和二维预训练骨干作区分。
