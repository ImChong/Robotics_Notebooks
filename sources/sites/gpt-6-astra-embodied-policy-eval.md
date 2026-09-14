# GPT 6 Astra as an Embodied Policy（技术报告站点）

> 来源归档

- **标题：** GPT 6 Astra as an Embodied Policy
- **类型：** site（双语技术报告 + 轨迹视频库）
- **URL：** <https://anonymous-report-421.github.io/public-website/?view=1>
- **英文入口：** <https://anonymous-report-421.github.io/public-website/?lang=en&view=1>
- **作者：** Yu-Mool Shu、Lipxin Zheng
- **代码：** <https://github.com/anonymous-report-421/eval-of-gpt-6-astra-as-policy>
- **站点源码：** <https://github.com/anonymous-report-421/public-website>
- **入库日期：** 2026-09-14
- **一句话说明：** 独立技术报告站：在 RoboDojo 十任务双臂仿真与 RoboLab 十任务子集上，对比 **GPT 6 Astra Direct** 与 **π0.5 + GPT 6 Astra** 混合闭环，附 100 条评测视频与 43 条精选片段画廊。

## 开源核查（步骤 2.5，2026-09-14）

| 资源 | 状态 | 说明 |
|------|------|------|
| 评测与报告代码 | **已开源** | [eval-of-gpt-6-astra-as-policy](https://github.com/anonymous-report-421/eval-of-gpt-6-astra-as-policy)（MIT）：`hybrid_rollout/robodojo` 仿真集成、`robolab` 集成、`report_site` 双语报告构建、`public_results/` 种子与分数元数据 |
| 静态报告站 | **已开源** | [public-website](https://github.com/anonymous-report-421/public-website)：预构建 `report_web/` + 画廊媒体 |
| 模型权重 | **未分发** | 需自备 `gpt-6-astra` / `xhigh` 授权；π0.5 使用 RoboDojo 发布的任务微调 checkpoint（上游 OpenPI/JAX） |
| 仿真资产 | **未分发** | RoboDojo 依赖 Isaac Sim 5.1 + 官方资产；RoboLab 依赖各自上游环境 |

## 页面结构（维护索引）

| 区块 | 内容要点 |
|------|----------|
| 摘要 | RoboDojo 十任务：混合 **48%** SR / **62.60** Score（仅 **14.4%** 步由 GPT 修正）；Direct **26%** / **37.81**；π0.5 公开参照 **15.67%** / **24.43** |
| 方法与设置 | 混合：π0.5 生成 50×14 关节候选 → GPT 审核沿用 1–15 步或 EEF 修正 1–5 步；Direct：纯 EEF 双臂输出 |
| RoboDojo 结果 | 50 对齐实例（10 任务 × 5）；官方榜模型在同子集重加权对照 |
| RoboLab 结果 | 同十任务语义抓放子集：Direct **98%**、混合 **92%**、π0.5 **36%**（各 50 episode） |
| 视频画廊 | 目标对齐、非抓取操作、反馈恢复、失败模式等定性片段 |
| 参考文献 | RoboDojo、π0.5、社区 Astra 实践索引等 |

## 对 wiki 的映射

- 主实体：[GPT 6 Astra 具身策略评测](../../wiki/entities/paper-gpt-6-astra-embodied-policy.md)
- 基准交叉：[RoboDojo](../../wiki/entities/robodojo.md)
- 学生策略：[π0.5](../../wiki/entities/paper-pi05-open-world-vla.md)
- 仓库归档：[eval-of-gpt-6-astra-as-policy](../repos/eval-of-gpt-6-astra-as-policy.md)
- 论文摘录：[gpt_6_astra_embodied_policy_2026.md](../papers/gpt_6_astra_embodied_policy_2026.md)
