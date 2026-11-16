# Awesome Agentic Robots — 《Towards Agentic Robots》官方动态文献索引

- **名称：** Awesome Agentic Robots
- **类型：** repo / literature-index / agentic-robotics / survey-companion
- **官方仓库：** <https://github.com/chonghanqin/Awesome-Agentic-Robots>
- **对应论文：** [Towards Agentic Robots（arXiv:2610.20442v1）](../papers/towards_agentic_robots_arxiv_2610_20442.md)
- **官方说明：** README 自称该综述的 companion repository，组织 Model / Data / Environment / Harness 四功能域与 17 个非互斥类别。
- **开源状态（2026-11-16 核对）：** README、文献数据与索引生成/校验工具公开；LICENSE 为 MIT。**非机器人学习方法代码**，没有本论文独立的训练、推理、ROS 或真机控制流水线。
- **数据入口：** [data/papers.yaml](https://github.com/chonghanqin/Awesome-Agentic-Robots/blob/main/data/papers.yaml)、[data/additional_papers.yaml](https://github.com/chonghanqin/Awesome-Agentic-Robots/blob/main/data/additional_papers.yaml)、[data/publications.yaml](https://github.com/chonghanqin/Awesome-Agentic-Robots/blob/main/data/publications.yaml)、[data/tables.yaml](https://github.com/chonghanqin/Awesome-Agentic-Robots/blob/main/data/tables.yaml)。
- **文献维护工具：** [scripts/validate.py](https://github.com/chonghanqin/Awesome-Agentic-Robots/blob/main/scripts/validate.py)、[scripts/generate_readme.py](https://github.com/chonghanqin/Awesome-Agentic-Robots/blob/main/scripts/generate_readme.py)、[scripts/sync_from_overleaf.py](https://github.com/chonghanqin/Awesome-Agentic-Robots/blob/main/scripts/sync_from_overleaf.py)。
- **核对范围：** 官方 README 的 Overview / Taxonomy / Papers 及仓库根目录、data/、scripts/ 的实际清单；2026-11-16。
- **通读：** 未完成
- **一句话说明：** 用于持续按四域十七类检索 agentic-robotics 研究；**不能把当前 main 的新增论文倒推为 2026-10-14 的 v1 论文引用**。

## 使用规则

1. **固定的论文证据：** 引述定义、Table 1、Figure 1/2/7、四项 Outlook 时，请使用 [arXiv 固定 v1](https://arxiv.org/html/2610.20442v1)。
2. **持续更新的索引：** 追踪后续工作用仓库 README 和数据清单，记录查询日期、论文首次公开版本与链接；需要量化/方法结论时再回各篇一手论文。
3. **代码边界：** 这里的 Python 文件负责文献清单同步/校验/生成 README，**不应生成或推断机器人运行时序图**。

## 对 wiki 的映射

- [唯一的综述论文实体节点](../../wiki/entities/paper-towards-agentic-robots-survey.md)：固化 v1 定义与系统分工，提供动态检索资源入口。

## 独立节点补全进度

- [307 条固定快照分批台账](../../docs/checklists/awesome-agentic-robots-node-coverage.md)：记录候选与验收状态，详情链接统一回各自的论文实体。
- [PAFF 独立详情](../../wiki/entities/paper-paff-2212-07398.md)：第一批完成原文与相关源码入口核查。
