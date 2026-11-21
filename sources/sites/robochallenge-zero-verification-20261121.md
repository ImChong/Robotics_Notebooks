# RoboChallenge Zero：新真机评测榜单的一手资料核验记录（待完成）

- **资料类型：** 官方平台 / 新评测榜单线索
- **归档日期：** 2026-11-21
- **核对范围：** 核查 RoboChallenge 官方网站首页与 Challenges 页面可检索正文、官方 GitHub 组织页公开仓库清单，并对照 Table30 V2 的官方论文入口；当前资料能证实 RoboChallenge 平台与 Table30 V2，**不能证实名为 Zero 的新榜单及其独立评测规则**。
- **通读：** 未完成

## 已找到的一手资料

1. [RoboChallenge 官方首页](https://robochallenge.ai/)：官方介绍真实机器人、标准化任务和公开排名；首页可检索内容展示 Table30 / Table30 V2，而非 Zero 的完整规则。
2. [RoboChallenge Challenges](https://robochallenge.ai/challenges)：目前可检索内容列出 Table30 V2、Table30 V2 Simulation 与 Table30；不可据此断言 Zero 不存在，只能说未取得可验证的独立条目。
3. [RoboChallenge 官方 GitHub 组织](https://github.com/RoboChallenge)：官方公开 RoboChallengeInference、Community、openpi 等仓库；截至本次核验，公开列表未提供 Zero 独立实现仓库。
4. [Table30 V2 原始论文（CVPR 2026 Workshops）](https://openaccess.thecvf.com/content/CVPR2026W/GigaBrainChallenge/html/Ma_Table30_V2_Evaluating_Generalized_Models_by_Real_Robots_at_Scale_CVPRW_2026_paper.html)：讨论 Table30 V2 的多任务、零样本和分布外泛化评测；**不等于** Zero 新榜单的公告。

## 线索及证据边界

- 用户提供线索：**“2026-11-20 前后 RoboChallenge 发布名为 Zero 的真机评测榜单，计划每月换题”。**
- **尚未独立验证**：发布日、正式产品名、月度更题频率、是否公开隐藏任务、参赛入口、评测本体、计分规则、参测模型、榜单成绩。
- **不得混淆**：Table30 V2 论文中的「zero-shot 评测能力」不是“RoboChallenge Zero”独立榜单已经上线的证据。
- 以上线索不应作为已核实事实导出至正式 wiki 实体或绘制推测的技术流程图；不能据此宣称原文通读完成或健康度 3/3。

## 下一步核验清单

1. 获取 RoboChallenge 官方 Zero 公告原文或独立榜单页面，核对日期与原始措辞。
2. 通读规则文档：任务公布与轮换机制、是否允许微调、测试执行协议、成功判据、复现实验、排名与结果公开范围。
3. 核对是否开放 Zero 专用源码、基线、数据、权重；区分已发布与计划发布。
4. 与现有 Table30 / Table30 V2 去重，建立独立且有一手证据的实体页，并补 Mermaid 主流程与源码运行时序图（有实际可运行源码时）。
5. 运行仓库健康度脚本和全量 CI，确认变更页面真实 3/3 后，将草稿 PR 转正。

> 本记录只用于保存核验进度，**不代表 Zero 项目已完成正式 ingest**。
