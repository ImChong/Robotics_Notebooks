# SO-Arena：评测 System One 模型的能力谱系与决策可靠性 / Benchmarking the Capability Spectrum and Reliability of System One Models

> 一手论文阅读归档。只总结方法、结果和复现边界；不转载基准测试样本或隐藏答案。

- **类型：** paper / system-one / decision-model / benchmark / reliability
- **英文标题：** SO-Arena: Benchmarking the Capability Spectrum and Reliability of System One Models
- **中文标题：** SO-Arena：评测 System One 模型的能力谱系与决策可靠性
- **作者：** Lichen Li, Ziping Dong, Sicheng Feng, Qi Li, Xinchao Wang（前两位共同一作）
- **arXiv：** <https://arxiv.org/abs/2611.15777>
- **HTML 原文：** <https://arxiv.org/html/2611.15777v1>
- **PDF：** <https://arxiv.org/pdf/2611.15777v1>
- **版本：** arXiv:2611.15777v1，2026-11-11；32 页、10 图、7 表
- **项目主页：** <https://lichenlillc.github.io/SO-Arena/>
- **代码：** <https://github.com/holdrain/SO-Arena>
- **数据集：** <https://huggingface.co/datasets/SO-Arena/SO-Arena>（用户给出的 https://huggingface.co/SO-Arena 是组织主页）
- **代码归档：** [SO-Arena 复现入口](../repos/so-arena.md)
- **项目页归档：** [SO-Arena 官方主页](../sites/so-arena.md)
- **wiki 映射：** [SO-Arena 独立论文节点](../../wiki/entities/paper-so-arena.md)
- **入库与核对日期：** 2026-11-19
- **核对范围：** arXiv HTML v1 的 §1–6、表 1–3、图 1–8、附录 A（数据构造、历史依赖验证、BFCL/ToolSandbox/Web 评分）、附录 B（模型设置与拒答）、C（延迟测量）、D（域差异及可靠性分析）；交叉核对 GitHub README、evaluator/README、MODEL_ADAPTER、jev/so_arena.py、evaluator/jevbench.py、reliability 目录的 README 和实现。
- **通读：** 已完成
- **一句话说明：** 用独立的能力分数 C 与决策可靠性分数 R 评测快速结构化决策模型，且加入交互式任务验证。

## 论文精读要点

### 任务和数据（§2–3、Appendix A）

- System One 模型对状态 s、问题 q、候选集合 C 输出概率，覆盖 choice、noul（yes/no）与 score（有序等级）。
- 2,919 自然任务 / 11 领域 / 5 能力家族：语义理解、知识推理、安全政策遵循、证据评估、Agent 任务完成。
- 2,300 静态、282 工具/API 模拟器 episode、337 文本 Web（100 单步 + 237 参考路线）；九域设难度等级，数学额外有 30 个 Very Hard。
- 标签按模型可见信息人工核验；终态验证包括 CALVIN、MiniGrid、EAI-VirtualHome 等任务，后者构造了**终态一致但过程条件不同**的反事实历史。

### 指标（§3.3）

- 能力榜 C：默认高难面板共 1,232 任务，每域先平均 credit，11 域再等权平均；不能将所有任务直接整体平均。
- **S（Option-order stability）**：原题与 12 次重排，按选项内容比较 argmax，扣除随机一致率；测稳定而非是否正确。
- **N（Null discrimination）**：一份保留正确选项、一份删除正确选项，按模型 null 概率得到 ROC AUC，N = 2×AUC−1。
- **R（Reliability）**：S、N 均为正时 R = √(S×N)，否则 R=0；与 C 并列展示，不能相互替代。

### 实验观察（§4、Table 1–2、Figure 5–8）

- 评估 21 个模型，其中 18 个 System One 有完整可靠性主指标。能力领先者仅列可靠性第七；JevAny-27B 可靠性领先，但能力排名不领先。
- 语义判断部分 System One 可接近 System Two；安全、路由、工具/Web 场景 backbone 更关键；数学、科学和代码理解的多步推导更难，显式 thinking 对数学、科学改善较大。
- 作者的 H100、batch=1 的单模型计时显示开源 System One 平均单题低于 0.3 秒；这不是其他硬件或机器人实时控制保证。
- 可靠性中缺答案识别差异显著，不能用能力高分推断其稳健性。
- 所有数据与排名是原文**作者报告**，并非独立实验结果。

### 局限和使用边界

1. 领域等权不代表具体生产流量分布；S 不等于校准，N 不等于开放任务全面拒答。
2. 工具和 Web 环境是受候选集合、参考路线和模拟器预算约束的测试；不代表任意真实互联网或机器人的开放闭环。
3. SO-Arena 主体针对通用 System One 决策模型，虽然借用了 CALVIN 等机器人任务，但**不是机器人 VLA/运控模型 benchmark**。
4. HF 数据集展示的行数不代表完整 2,919 自然任务的可复现运行流程；须依据 data-lock.json 与官方启动器。
5. 部分测试样本来自受非商业许可约束的第三方数据，部分条目用受保护压缩包；不得把训练集与测试集混用，也不得把评测题目导入训练语料。

## 对 wiki 的映射

- [SO-Arena](../../wiki/entities/paper-so-arena.md) — 独立论文节点、技术图、工程入口和决策可靠性。
- [AI Agent 评测](../../wiki/concepts/ai-agent-evaluation.md) — 在终态、成功率之外补 S/N/R 可靠性维度。
- [Jev](../../wiki/entities/typesafe-jev.md) — 有限候选概率接口是模型/服务，SO-Arena 是 benchmark。
