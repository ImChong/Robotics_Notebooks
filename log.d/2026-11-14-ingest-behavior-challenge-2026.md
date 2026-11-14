## [2026-11-14] ingest | 2026 BEHAVIOR Challenge 赛制、评测协议与源码核对

- **意图：** 按 CoRL 2026 官方一手资料补全已有 2026 BEHAVIOR Challenge 来源归档，增加独立赛事节点，不重复建设 BEHAVIOR-1K 基准或 HiPhys-RSI 冠军论文。
- **归档：** sources/sites/behavior-challenge.md 与 sources/repos/behavior-1k-challenge.md。
- **wiki：** wiki/entities/behavior-challenge-2026.md；评测选型 hub 添加内部入口。
- **开源结论：** 赛事环境与基线开放、示范数据公开；冠军方案并不包含在赛事代码库。官网要求 v3.9.3，但本次源码接口只成功读到 main，版本对齐与端到端复现待办。
