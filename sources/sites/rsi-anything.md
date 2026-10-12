# RSI-Anything 官方项目页

> 来源归档（项目页与发布说明）

- **标题：** RSI-Anything: Unlocking Recursive Self-Improvement with World Models
- **项目页：** <https://phai-labs.com/news/rsi-anything/>
- **发布：** 2026-10-12
- **论文：** [arXiv:2610.13912](../papers/rsi_anything_arxiv_2610_13912.md)
- **代码：** <https://github.com/Gen-Verse/RSI-Anything>（Apache-2.0；[仓库归档](../repos/rsi-anything.md)）
- **关联项目：** [JEPA-Anything](./jepa-anything.md)
- **一句话说明：** PhAI Labs 将 RSI-Anything 描述为建立在 JEPA-Anything 预测式世界模型基础上的通用递归改进框架。

## 项目页所述结果

- CodeContests 消融：静态候选验证 26.3%，预测选择 29.2%，再做世界模型适配 31.2%；报告采用 ReasonFlux-Coder-7B、200 道题、每题 3 个种子和近似在线测试调用预算。
- MiniWoB++：成功率由 51% 到 99%，平均动作数减少 39.4%。OSWorld 报告相对初始值 +18.4%、相对 RLAnything +5.5%（相对增幅）。
- LabHorizon F200 L2 五个版本：21.80 到 46.01%。反应优化 40 次实验平均最佳产率 96.15%，对照 Bayesian Optimization 为 82.16%。
- 项目页还报告一个免疫学方向假设，但明确仍待验证。

## 读数边界

这些数字对应项目方报告的任务与协议。项目页说明仍需多轮联合评估来确认策略和世界模型互相促进；实验室规划分数不等于真实仪器执行验证；反应结果受已报告筛选范围与实验预算限制。

## 对 wiki 的映射

- [RSI-Anything 实体页](../../wiki/entities/paper-rsi-anything.md)
- [PhAI Labs 机构页](../../wiki/entities/phai-labs.md)
