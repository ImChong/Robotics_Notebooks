# EmbodiedRSI: Active Continual Robot Learning Through Hypothesis-Guided Co-Evolution（基于假设引导协同进化的主动持续机器人学习）

> 作者论文的一手阅读归档；仅提炼可追溯事实和工程含义，不转载论文全文。

- **类型：** 论文 / 具身递归自我改进 / VLA Harness / 主动试验与记忆强化学习
- **论文：** https://arxiv.org/abs/2610.10498
- **全文 HTML：** https://arxiv.org/html/2610.10498
- **PDF：** https://arxiv.org/pdf/2610.10498
- **版本：** arXiv:2610.10498v1（2026-10-07，公开 PDF 12 页）
- **作者：** Python Song、Zhixuan Liang、Kelsey Fu、Mengdi Wang、Junfeng Yang、Shilong Liu
- **机构：** 哥伦比亚大学（Columbia University）、普林斯顿大学（Princeton University）
- **项目主页：** https://geeksongs.github.io/agentic_robotics/
- **知识节点：** [EmbodiedRSI](../../wiki/entities/paper-embodiedrsi.md)
- **核验日期：** 2026-11-08
- **核对范围：** 已通读公开 v1 论文的 12 页与 arXiv HTML：摘要、§1 引言、§2 相关工作、§3.1 问题、§3.2 Fast System、§3.3 Slow System、§4.1–4.4 三组评测与消融（Tables 1–5 / Figures 1–7）、§5 结论及参考文献；并核对作者站点结果和发布范围。**公开 v1 只含这些正文与参考文献，没有其多次引用的附录 A–I；附录算法、超参数、backbone 细节无法全文核验，不能据正文推断其细节。**
- **通读：** 已完成

## 研究问题与关键贡献

预训练 VLA 在任务/物体变化时失败；直接追加遥操作数据进行模型微调昂贵。作者将可修改层限定在 **frozen VLA 外面的 agentic harness**，提出三处协作改进：

1. **Hypothesis Graph + VOI 实验选择（§3.2、式 4–6）**：节点存 code 或 skill 假设；有向 `refines` 存修订历史。相同初始条件下进行 matched trials，用支持/反驳计数更新 Beta 后验；选实验时比较预期不确定性下降与执行成本。
2. **Code-Skill Co-Evolution（§3.2、式 7）**：同一初始状态比较 `code-only`、`skill-only`、`code+skill`，联合增益 `w_ij = R(C_i,S_j) - max(R(C_i),R(S_j))`。正联合增益在图上保留 `works-with` 关系。
3. **分层记忆 + Reward-Grounded Memory Learning（§3.3、式 8–10）**：M1=物理轨迹与工具/动作/回报；M2=假设图及物理证据；M3=抽象规则与策略。记忆 Manager 在 Retain / Merge / Abstract / Compress / Forget / Skip 中选动作，冻结 operator 执行；用 PPO 优化后续任务改善、假设不确定性降低与下次试验成本的组合奖励。

### 公式与模型边界

- `VOI_k(t) = G_k(t) / C(t)`，`G` 是候选试验期望不确定度下降，`C` 聚合耗时、仿真步、GPU、reset 开销。**不是**由单次成功率决定试验是否值得做。
- 后验均值 `c_i = (n_sup,i + alpha)/(n_sup,i+n_ref,i+alpha+beta)`；论文明示用 matched success-rate difference 给支持/反驳证据，而不是未经控制的自由比较。
- 记忆奖励 `r_mem = λ1(R_next-R_now) + λ2(H_now-H_next) - λ3*C_next`；论文使用 PPO 训练 **memory manager 的选择策略**，而非在线 PPO 训练整个机器人 VLA。
- 冻结 `Qwen3-Embedding-0.6B` 编码检索内容，冻结 `Qwen3.5-122B-A10B` operator 执行记忆变更。训练的是 Manager 的 context / skill / scoring 模块（§3.3）。
- 全文明确 frozen robot foundation model 不改权重；§5 将用轨迹改进底层模型列为未来工作。

## 主要实验结果（作者报告）

| 设置 | EmbodiedRSI | 对照与含义 |
|---|---:|---|
| RoboCasa365 Overall（Table 1） | 77.0% | Harness VLA 63.6%；注意是 Overall |
| RoboCasa365 Composite-Unseen（Table 1） | 71.3% | Harness VLA 40.1%；差 +31.2 个百分点 |
| LIBERO-Pro Overall（Table 2） | 86.8% | Harness VLA 72.1%；+14.7 个百分点 |
| 真机 SO-101 五任务合计（Table 3） | 71.3% | 同一已微调 frozen SmolVLA baseline 46.0%，各任务每组 30 次 |
| RoboCasa365 @50 episodes（Table 4） | 72.9% | Random Exploration 27.1%；学习 AUC 0.635 对 0.309 |
| Composite-Unseen: full memory（Table 5） | 71.3% | retrieval-only 61.7%；PPO task reward only 65.2% |

### 实验读法与限制

- 仿真每任务 **10 seeds × 每 seed 10 次试验**；evolution 与 evaluation seeds 不重叠（§4.1）。
- 真机零样本迁移是 **在仿真中演化 harness，再直接部署到实体 SO-101**，不是「真机现场自主训练进化」。
- 真机 SmolVLA 在评测前每任务使用 **50 条人工遥操作示范**，并在 **4×H200 上微调 20,000 steps**；只是在后续对比及 harness 迁移时冻结（§4.3）。
- 真机五项分别为 Oreo→红碗 70.0→86.7；算术筹码 10.0→90.0；蛋糕堆叠 63.3→76.7；**眼镜桥抓取 60.0→56.7（退化）**；折毛巾 26.7→46.7。不能宣称所有任务提升。
- 联合进化优于单独技能进化：Composite-Unseen 71.3% vs 55.6%（§4.4）。论文未给出实现源码、所有附录设定及独立复现，不能据报告数值判定可直接复现。

## 原文未公开部分及核验范围

- PDF 共 12 页，结尾为 References，尽管正文多处引用 **Appendix A–I**（算法 1、超参数、消融细节、backbone 设定等）；公开正文无所引附录。这是**源论文发布的材料缺口**，非「已读完附录」。
- 截至 2026-11-08 官方项目页无训练/评测/机器人部署代码入口。作者公开的 `Geeksongs/agentic_robotics` 是**网站源码**，不是 EmbodiedRSI 算法实现；不把非作者仓库 `Einsia/EmbodiedRSI` 混为对应论文代码。
- 作者站点 Abstract 段将 EmbodiedRSI 误写成 “EmbodiedEvo”，与原论文标题和站点其他区块不一致；实体身份以 arXiv:2610.10498v1 为准。

## 对知识库的映射

- [EmbodiedRSI：论文实体](../../wiki/entities/paper-embodiedrsi.md)：VOI、假设图、联合进化、记忆奖励与性能边界。
- [递归自我改进 RSI](../../wiki/concepts/recursive-self-improvement.md)：属于 frozen-policy harness-level RSI。
- [VLA](../../wiki/methods/vla.md)：底层权重固定与外层 agent 调度分工。
- [强化学习](../../wiki/methods/reinforcement-learning.md)：PPO 用于 memory manager，勿误读为在线 VLA 更新。

## 原始入口

- [arXiv v1 全文](https://arxiv.org/html/2610.10498)
- [作者项目主页](https://geeksongs.github.io/agentic_robotics/)
- [官方项目网站源码（仅网页）](https://github.com/Geeksongs/agentic_robotics)
