# 当下的智能体机器人在哪些地方成功、在哪些地方失败？ | Where Do Today’s Agentic Robots Succeed and Fail?

- **原文：** <https://www.embodied-first-exam.ai/blogs/failure-analysis/>
- **发布者：** Embodied First Exam 团队；[Manling Li 的发布帖](https://x.com/ManlingLi_/status/2125300304648470767)由原文直接链接，帖附 [X 长文](https://x.com/i/article/2124944949011709952)。
- **日期：** 页面标注 2026-11；页脚标注结果更新至 2026-11-24，数据说明注明核心榜单对应 2026-11-20 快照。
- **相关材料：** [arXiv v1 论文归档](../papers/embodied-first-exam-2611-12177.md) · [官方项目](../sites/embodied-first-exam-ai.md) · [官方代码](../repos/embodied-first-exam.md) · [运行回放](https://www.embodied-first-exam.ai/runs/) · [公开轨迹](https://huggingface.co/datasets/MLL-Lab/embodied-exam-data)。
- **核对范围：** 2026-11-26 读取官方英文博客全文（正文、图表文字、算例、局限与数据说明），并与 arXiv v1 正文 §2–4、附录 G–J 复核口径。X 直连受限，按原帖 ID 通过可读镜像核验发布帖与所附长文各节文字；定量结论以项目博客为准。未下载全套轨迹或复跑评测。
- **通读：** 已完成

## 官方博客分层摘记

| 章节 | 复核要点 |
|---|---|
| 哪些任务未解 | 10 个模型与 harness 组合各试一次；481 个 Standard 测试中 GPT-6.2 Astra + Codex 解出 279，112 个无解；112 个中 104 个有 Privileged 配对版本，其中 90 个至少有一个组合解出。54 个时序配对任务中，Standard / Realtime 全员未解分别为 15 / 22，两个最强 Realtime 组合各解出 22 个。 |
| 接口权限 | 旧 391 任务复跑，默认模式为 341 个 Standard 与 50 个 Realtime。对 GPT-6 Astra，博客报默认接口 29%、仅物体状态 45%、仅提高重置预算并支持存档 53%、两个都开 68%；博客对照 Privileged 为 79%。这和 arXiv v1 报告的 28% / 77% 基线及 Privileged 数字有差别，不跨来源混算。 |
| 失败阶段 | 允许从进度图、轨迹与日志区分失败位置和失败现象；九个已标注组合不含后加入的 GPT-6.2 Astra。Standard 中同一测试的失败阶段集中度 57%，比前置阶段加权的随机基线 35% 高；相同阶段不代表相同根因。按已到达阶段统计，材料 / 机身运动在 Standard 分别为 73% / 55%，在 Privileged 为 32% / 21%；不可当作成功率互补或跨新套件排序。 |
| 控制方式 | 作者随机抽样 2,359 次，Standard 61% 直接控制、Realtime 95% 写控制程序，均为样本描述；足球 GPU 版本同时提供训练库，不分离两者贡献。 |
| 成本 | 十个组合在 Standard 与 Privileged 内失败尝试的成本中位数均高于成功尝试；例如 GPT-6.2 Astra Standard 按标价估算为 5.75 美元 vs 2.76 美元，不是提高预算的因果效应。 |
| harness、提示和重复 | 同模型对比要固定 harness 和工具。RoboDojo 提示消融固定 54 个场景、时限和判据，Fable 从 34/54 到 44/54；47/54 个场景有回初始位姿要求。GPT-6.1 Sol 在旧 456 题、固定 high effort 三次均值 30.8%，样本标准差 2.5 个百分点，不能外推到所有模型。 |
| 持续学习 | Embodied Olympics 区分赛季保留工作区和每场清空工作区；515 个 episodic agent-match 对中约 26% 改程序、4% 改后多拿分，观察到对手策略不等于形成可保留改进。 |
| 局限 | 大部分结果来自仿真，真机 buzz-wire 只是接口演示；公开任务可能被预训练见过。人工抽查参与失败标注方法开发，自动标签不能替代独立 grader。 |

## 对 Wiki 的映射

- [Embodied First Exam 主节点](../../wiki/entities/embodied-first-exam.md)
