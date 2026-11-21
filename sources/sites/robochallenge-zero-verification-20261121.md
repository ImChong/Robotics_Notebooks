# RoboChallenge Zero：新真机评测榜单的一手资料核验记录（待完成）

- **资料类型：** 官方平台 / 新评测榜单线索
- **归档日期：** 2026-11-21
- **增量核查：** 2026-11-21，复查官方 Challenges、官方原始论文及年度报告；以“RoboChallenge Zero / monthly / 每月换题 / 2026-11-20”等关键词检索英文、中文和官方站点。用户补充了 Zero 独立详情页及技术报告直链。
- **核对范围：** 已核查可检索的 RoboChallenge 官方入口、Table30 V2 公开论文与官方年度报告相关章节，以及 GitHub 公开资源。现已明确 **Zero 官方详情页与专用技术报告的 URL**，但尝试直接读取详情页与 PDF 时均无法取得正文（网页抓取 cache miss；独立下载失败），因此仍不能声称完成了 Zero 报告通读。
- **通读：** 未完成

## Zero 官方一手入口（已定位链接，正文待读）

1. [RoboChallenge Zero 官方独立详情页](https://robochallenge.ai/benchmark_detail/zero)：用户提供的 `/benchmark_detail/zero`；已尝试打开，未获取到正文。**不以链接本身推断规则**。
2. [RoboChallenge Zero 官方技术报告 PDF](https://robochallenge.ai/docs/robochallenge_techreport_ZERO.pdf)：用户提供的 `/docs/robochallenge_techreport_ZERO.pdf`；已尝试直接抓取、变体 URL 及下载，但未能取得 PDF 字节或可引用正文，因此 **并未通读该报告**。

## 已找到的其他一手资料（仅可支撑旧版平台事实）

1. [RoboChallenge 官方首页](https://robochallenge.ai/)：公开真机评测平台、Table30 和 Table30 V2 排名入口；已检索的可读页面未给出 Zero 新榜单的完整规则。
2. [RoboChallenge 官方 Challenges](https://robochallenge.ai/challenges)：当前可检索条目有 Table30 V2、Table30 V2 Simulation 和 Table30；这不等于证明未上线别的渠道。
3. [RoboChallenge 官方 GitHub 组织](https://github.com/RoboChallenge)：能找到评测基础设施与既有任务资源；未找到可直接识别的新 Zero 独立源码/评测协议。
4. [Table30 V2 原始论文，CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/GigaBrainChallenge/html/Ma_Table30_V2_Evaluating_Generalized_Models_by_Real_Robots_at_Scale_CVPRW_2026_paper.html)：官方摘要明确提出 **multi-task、zero-shot、out-of-domain** 评测；它描述的是 Table30 V2，而不是在 2026-11-20 发布的独立 Zero。
5. [RoboChallenge 官方年度报告（2025 Q4–2026 Q1）](https://robochallenge.ai/2025%20RoboChallenge%20%E5%B9%B4%E5%BA%A6%E6%8A%A5%E5%91%8A.pdf)：报告将“提升 Zero-Shot 能力评估覆盖”列为社区反馈与后续改进方向，**不能推导成 2026-11-20 发布的新榜单或每月换题承诺**。

## 已核实结论与证据边界

- **已证实：** RoboChallenge 原有 Table30 真机榜、升级版 Table30 V2 及后者规划的 zero-shot / OOD 测试，均有官方独立材料支撑。
- **已定位：** 用户提供的 Zero 独立详情页与专用技术报告官方直链（见上），此前「尚未找到 Zero 独立入口」的表述不再准确。
- **尚未核实正文：** 2026-11-20 的发布时间、每月换题机制、提交条件、评测与打分规则。目前**没有成功读取 Zero 技术报告原文**，更不能把独立 PDF 的存在视为规则已通读。
- **不可混淆：** Table30 V2 中的 **zero-shot 测试设置**、报告中的 **Zero-Shot 规划**、独立命名为 **Zero** 的新产品/榜单是不同层面的说法，不能相互替代。
- **不做推测：** 参赛模型、硬件、任务种类、是否免微调、成绩、评分规则和真实发布时间均留待一手资料核对。

## 下一步核验清单

1. 成功下载并**通读已定位的 Zero 官方技术报告 PDF**，获取独立详情页正文与发文时间，逐项标注对应页码。
2. 通读规则原文，核对“每月换题”的准确措辞、轮换周期与版本、训练集和测试集可见性、提交入口及评分指标。
3. 查证是否有 Zero 专用评测代码、基线、数据、模型权重；区分实际可访问、部分开放和预告。
4. 与既有 Table30 / V2 **先去重**，正式建立独立实体及基于原文的 Mermaid 主流程；存在可运行源码才做源码运行时序图。
5. 执行 `python3 scripts/report_health.py --changed` 实测新增节点 **3/3** 和仓库完整预检后再将 PR 转正。

> **当前状态：** 已收到 Zero 官方原文 URL，但无法取得 PDF 内容；故保留 `通读：未完成`，不创建臆测的 wiki 节点，不宣称健康度 3/3。
