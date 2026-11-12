## [2026-11-12] ingest | Eureka ICLR 2024 官方论文与源码复核

- 在已有 [Eureka 论文实体](../wiki/entities/paper-eureka-llm-reward-design.md) 上增补基于源码的 Mermaid 运行时序图，没有新增重复项目节点。
- 复核 [arXiv 2310.12931v2](https://arxiv.org/html/2310.12931v2) 与 [官方项目页](https://eureka-research.github.io/)，新增 [官方仓库源码档案](../sources/repos/eureka-research-eureka.md) 和 [项目页归档](../sources/sites/eureka-research.md)。
- 源码快照 eureka-research/Eureka@9eee428；核实 MIT 许可、`eureka.py` → `train.py` → `rl_games` 的训练/反思路径；发现 README 和 config.yaml 的 sample/iteration 默认值不一致。
- 论文原文通读结果沿用 2026-11-08 的详细归档；本次源码检查只读与本流程相关的文件，未运行训练或视频。
