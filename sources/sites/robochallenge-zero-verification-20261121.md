# RoboChallenge Zero：官方技术报告全文核验（2026-11-21）

- **资料类型：** 官方评测平台 / 官方技术报告入口
- **归档日期：** 2026-11-21
- **核对范围：** 用户提供 `robochallenge_techreport_ZERO.pdf`，已阅读 PDF 全文 11 页（摘要、§1–§4、表 1–2、图 1–10、参考文献）；并复查 RoboChallenge 官方公开 Challenges/首页及 Table30 V2 的上下文。**Zero 独立官网详情页目前无法经公开抓取工具取得正文**，故其网页发布公告和代码入口仍待确认。
- **通读：** 已完成

## 一手入口

1. [RoboChallenge Zero 官方技术报告 PDF](https://robochallenge.ai/docs/robochallenge_techreport_ZERO.pdf) — 本次原文全文读取依据；[逐页归档与实验记录](../papers/robochallenge_zero_technical_report_2026.md)。
2. [RoboChallenge Zero 官方独立详情页](https://robochallenge.ai/benchmark_detail/zero) — 已确定 URL；网页正文尚不可直接取得。
3. [RoboChallenge 官方 Challenges](https://robochallenge.ai/challenges) — 可检索到 Table30 V2 / Simulation / Table30，不能据此推定 Zero 未发布。
4. [Table30 V2 一手报告](https://robochallenge.ai/robochallenge_techreport_V2.pdf) — 对照旧任务套件，与 Zero 独立报告分开解读。
5. [RoboChallenge 官方 GitHub 组织](https://github.com/RoboChallenge) — Zero 专属可运行代码尚未核实。

## 已从报告核实（按页码）

| 核对项 | PDF 一手依据 | 结论 |
|---|---|---|
| 任务刷新 | §1、§2.3（pp.1,3） | 月度新任务 + 旧任务物体、布局、指令变体；同系统版本跨月重测 |
| 严格零样本 | §2.1（p.2） | 无任务专属训练示范；原则上不进行任务定制 prompt、在线更新或人工指导 |
| 系统范围 | §2.2（p.2） | 支持原生 VLA、VLM/LLM Agent、集成系统；首批数据为 11 个 Agent |
| 真机协议 | §2.2（p.2） | W1 双臂与 ARX5 单臂、RGB-D、Agent 调末端位姿/夹爪工具、IK 执行 |
| 指标 | §2.4（p.3） | 完整成功率（主）、0–50 进度分、耗时、调用次数 |
| 首轮结果 | §3、表 2、图 1–5（pp.3–6） | 六任务、330 次测试；148 次成功（44.85%）；T4 89.09%，T3 0% |
| 局限 | §3.4–3.5（pp.7–10） | 接触密集/柔性物体/连续精确轨迹弱；标定/避碰与模型推理误差尚未隔离 |

## 证据边界及待核项

- **不能把报告的 11 个 Agent 结果说成原生 VLA 榜单结果**：报告明确原生 VLA 提交仍开放。
- **不能用月度跨期分数直接排强弱**：报告限定同一 release 内比较。
- **发布日期**：2026-11-20 的首次发布信息尚无 PDF 正文证据；不能猜。
- **源码与数据开放**：报告未列 Zero 独立可运行代码路径，网页无法核实；不声称已开源，也不声称一定闭源。
- 已将全文事实写入 [RoboChallenge Zero 独立 wiki 实体](../../wiki/entities/paper-robochallenge-zero.md)。此来源归档“通读：已完成”只证明用户所供 PDF 已通读，**不代表官网详情页已读取或仓库健康度已验证**。
