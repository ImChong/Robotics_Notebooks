## [2026-11-14] ingest | AHEADance 官方项目页复核与复现边界

- 意图：用户再次提出 AHEADance 入库；按 arXiv:2611.13370 在库内查重，复用现有唯一论文实体，不重复创建节点。
- 来源：arXiv v1 完整论文和 UCL Humanoid Robotics Lab 官方项目主页（https://ihcr.github.io/AHEADance/）。
- 结论：ListenAhead 与 Commit Forcing 的既有论文分析可保留；补录官方网页真机视频、麦克风演示、MuJoCo 回放和云端 Live 的开放范围。
- 开源：项目页仍为 “Code soon”，未发布可复现的 AHEADance 研究训练/推理源码或评测入口，源码运行时序图不适用。
- 更新：`sources/sites/aheadance_project.md`、`sources/papers/aheadance_arxiv_2611_13370.md` 与 `wiki/entities/paper-aheadance.md`；流程图补全机器人测量状态仅反馈 SONIC 跟踪层。
