## [2026-11-16] ingest | sources/papers/opennwm_arxiv_2611_01844.md — 核对 OpenNWM arXiv v1 公式与实验、源码和 Go2 部署边界

- 更新已存在的论文实体 wiki/entities/paper-opennwm.md（不重复建页）：UNLA 物理监督、CDiT、真实动作 CEM、v1 Table 1–5 与 NavAnywhere Table 6。
- 更新来源归档（paper/repo/site），核查根 run.py、LAM/NWM 训练评测入口、Hugging Face 模型与数据。
- 明确 Office-Go2 ATE 2.89 / RPE 0.65 是离线评测；Go2 EDU + Jetson 仅有部署计划，尚无验证的闭环节点。
- 将节点挂接至导航纵深路线 Stage 5。
