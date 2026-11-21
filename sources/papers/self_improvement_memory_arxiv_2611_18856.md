# Self-Improvement Memory Enables Continual Recursive Self-Improvement / 自改进记忆支持持续递归自我改进

> 论文一手资料精读归档；记录事实、实验与边界，不复制原论文全文。

- **作者：** Xin Yu、Fei Xia、Jiaqi Pan、Youbao Tang、Chengzhi Liu、Adams Yu、Sicong Jiang、Yanfei Chen、Lingzhou Xue、Ting Yu
- **机构：** Google Cloud AI、Google DeepMind、Google Cloud AI Research、The Pennsylvania State University（按作者论文署名）
- **类型：** arXiv 预印本，cs.LG / cs.AI，未宣称已经同行评审
- **版本：** arXiv:2611.18856v1，2026-11-13
- **论文：** https://arxiv.org/abs/2611.18856
- **HTML 全文：** https://arxiv.org/html/2611.18856v1
- **PDF：** https://arxiv.org/pdf/2611.18856v1
- **DOI：** https://doi.org/10.48550/arXiv.2611.18856
- **作者项目页：** https://lucasxinyu.github.io/sim/
- **代码：** 官方项目页截至 2026-11-21 未列 SIM 可运行实现仓库；不可用其他 RSI 仓库替代
- **核对范围：** arXiv v1 正文 §1–§5（问题定义、算法、四域实验、局限和伦理）；附录 B（评测口径）、C（独立运行与预算）、D（SIM 结构与 proposer/selector 使用）、E（Tables 5–7 与消融）、F（实例与跨域归纳）；作者项目页开放状态及数据解释。
- **通读：** 已完成

## 一手资料要点

1. **问题：** 在固定 policy model / benchmark 上优化好的 harness，换模型或数据分布时可能退化；仅继承最后的 harness 代码，无法回答「为什么改、预测能解决什么、测得什么」。研究的是可兼容接口前提下的 Continual Recursive Self-Improvement（CRSI）。
2. **优化对象：** harness 是围绕被冻结权重的策略模型配置的 prompt、工具与执行流程。SIM 是给 **harness proposer 和 selector** 的改进过程记忆，不是任务执行策略的用户记忆，也不是权重级在线训练。
3. **记录格式：** 每个根因条目包含多次尝试，存储改动及理由、评测前预测、评测后修复/未修复项、域/benchmark/model/round 及评分依据。论文附录 D 描述单 JSON 文件中的 `next_id` / `carried_from` / `aliases` / `problems[].attempts[]`。
4. **过程：** proposer 读整本 SIM 决定复用旧根因或提出新根因；selector 也读 SIM 比较候选。选定的改动即便未修好预测任务，也记录已观测结果；新域/模型下延续旧记录并追加新尝试。held-out 评分只用于报告，不进入候选选择/记忆更新。
5. **对照：** Cold Start = 初始 harness；Harness Transfer = 仅继承演化 harness；Problem-Only = harness + 根因；Full SIM = harness + 完整改进经验。三种迁移条件从同一源演化 harness 出发，目标任务、策略、轮数、候选预算一致；每阶段 10 轮，每轮 5 个候选。
6. **四域：** coding（Terminal-Bench 2.1→4）、text classification、agentic workspace（Harvey LAB→JobBench/GDPval/APEX-Agents）、engineering design（EngDesign→Frontier-Eng）；跨模型对照是 Claude Opus 5.5→Gemini 3.8 Flash，任务/拆分固定。
7. **跨模型 Workspace（Table 6）：** 未演化种子在 Gemini 端为 73.89；Opus 演化 harness 直接迁移起始 68.41，下降 **5.48** 个百分点。相同起点继续 10 轮，Harness Transfer **73.42**、Problem-Only **79.22**、Full SIM **84.75**；Full SIM 比无记忆 **+11.33** 分。此结果不是十次独立 Workspace 试验。
8. **其他跨模型（Table 6）：** Coding / AHE 最终 18.18（仅 harness）vs 21.82（SIM）；Text classification / FewShotAll 80.20 vs 83.80；Engineering / ReAct 42.61 vs 50.43。Full SIM 在全部四域最终领先。
9. **跨数据集（Table 5）：** 六个任务设置 Full SIM 全部最终领先仅继承 harness，分差 2.43–4.35；仅保留根因可能正收益或负收益。
10. **独立编程配对（Table 3）：** Terminus-2 在 TB2.1 独立演化 10 次，各自将相同源 harness 分配为有 SIM/无 SIM 的 TB4 十轮继续优化。平均起点 18.36；最终无 SIM **19.33**、有 SIM **21.70**；按未四舍五入结果的配对均值 **+2.36** 分，9/10 组受益，1 组为 -0.61；95% CI [1.05,3.68]，配对 t 检验 p=0.003。
11. **跨模型代际（Table 1）：** 4 Argon 上 CRSI **94.8** 低于 Fresh RSI **95.6**，故结果不等于每次继承必胜。
12. **案例（Appendix F）：** 从 TB2.1 到 TB4 时，已有完成判定门控仍有价值，但要补上新任务的「可行与最优」判定；SIM 留下源诊断和目标追加的结果，支持保留动机而重写实现。

## 局限与证据边界

- 十组独立重复仅覆盖 coding；其他领域每条件单次完整 RSI 运行，五次 task rollout 不等于五个独立优化种子。
- 只评估 **harness 接口兼容** 的迁移；当前实现完整读取 SIM、无检索/截断，长期扩展的 token/时间成本没有受控比较。固定轮数与候选数并不等于固定 token 或墙钟预算。
- Workspace 使用 LLM judge，存在评分偏差风险；改进可能迎合评分器，未证明全部修改单独的因果效应。
- 无公开完整代码；无机器人真机、Sim2Real 或在线力控实验，不可将 AI 工作台实验结论直接外推至机器人自主自改进。

## 对 wiki 的映射

- [独立论文详情：SIM / CRSI](../../wiki/entities/paper-self-improvement-memory-crsi.md)
- [RSI 纵深路线 Stage 4](../../roadmap/depth-rsi.md)：harness 迁移、改进经验与跨模型验证
- [递归自我改进](../../wiki/concepts/recursive-self-improvement.md) · [Agent 记忆](../../wiki/concepts/agent-memory-and-dreaming.md)

## 参考来源

- [arXiv v1 HTML](https://arxiv.org/html/2611.18856v1)
- [作者项目页](https://lucasxinyu.github.io/sim/)
