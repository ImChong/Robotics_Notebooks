# Causal Reward World Models: Zero-shot Reward Design for Automated Skill Generation

- **作者：** Yang Yang, Yuchuang Tong, Zhengtao Zhang, Xu Ding, Ning Yang, Yifan Zhang, Haipeng Li, Kehu Yang, Miao Xin（通信作者）
- **机构：** 中国科学院自动化研究所（CASIA）；合肥工业大学智能制造技术研究院（HFUT）与安徽大学电气工程与自动化学院；中国矿业大学（北京）人工智能学院（致谢中提及 CASBOT 提供实验平台支持）
- **版本 / 载体：** arXiv:2606.23280v1 [cs.RO]（2026-06-22 提交，22 页，18 张图）
- **原文：** https://arxiv.org/html/2606.23280
- **项目页 / 代码：** https://yy12136.github.io/CRWM （宣称将开源或未列链接：项目页标注 `Code (coming soon)` 与 `HuggingFace (coming soon)`，论文第 I 节标注 `Codes will be available after review at https://yy12136.github.io/CRWM`，截至 2026-11-27 未确认公开源码仓库）
- **核对范围：** 通读 arXiv HTML 全文（Abstract、I–VII 节、Appendix A–C 与 Supplementary Material S-A–S-F，含 Table I–IV、Table S-I–S-II、Alg. 1 及全部公式与图注）及项目页全文（`project_page.txt`）
- **通读：** 已完成

## 原文核对笔记

- **核心问题：** 现有基于大语言模型（LLM）的自动奖励设计（ARD）依赖相关关系语义推理与在线试错迭代（Evolutionary Search Iterations, ESI），不仅在连续控制任务上推理成本高昂（每轮需多块 GPU 训练评估），还容易引入与任务语义相关但因果虚假（spurious correlation）的奖励项，导致**优化崩溃（Optimization Collapse）**（如 `ShadowHandLiftUnderarm` 中引入高频噪声项 `velocity`）与**规范博弈（Specification Gaming）**（如 `ShadowHandScissors` 中仅刷 `scissors_open` 奖励）。
- **方法架构（三阶段）：**
  1. **因果基础模型结构先验提取（IV-A）：** 在多任务交互中对 $k \in [3, 8]$ 个归一化物理奖励分量权重 $\bm{\omega}^{(n)}$ 做 $do$-扰动，同步采集宏观介入结果 $\mathcal{D}_{macro} = \{(\bm{\omega}^{(n)}, F^{(n)})\}_{n=1}^N$ 与微观结构变量轨迹 $\mathcal{D}_{micro} = \{(s_0^{(n)}, \dots, s_T^{(n)})\}_{n=1}^N$（丢弃原始观测 $o_t$，零填充对齐至共享 $d$ 维原子奖励分量池）。使用预训练因果基础模型 **LimiX** 从 $\mathcal{D}_{macro}$ 提取粗糙先验矩阵 $\langle M_{prior}, M_{conf} \rangle$。
  2. **联合优化与 CRWM 蒸馏（IV-B）：** 在 $\mathcal{D}_{micro}$ 上联合优化目标 $\mathcal{L} = \mathbb{E}[\mathcal{L}_{MSE}(s_t, M_{inst}(s_{t-1})s_{t-1})] + \lambda h(M_{inv}) + \mathcal{L}_{soft}$。其中**显式机制解耦（EMD）**将瞬时拓扑分解为 $M_{inst}(s_{t-1}) = M_{inv} \odot (\mathbf{1}_{d \times d} + M_{trans}(s_{t-1}; \theta))$，用 Hadamard 门控限制 MLP 瞬态调制项在骨架 $M_{inv}$ 内吸收接触/碰撞等短时物理交互；**置信度感知软融合** $\mathcal{L}_{soft} = \gamma \|M_{conf} \odot (M_{inv} - M_{prior})\|_F^2$ 自适应校正先验幻觉；结合 DAGMA 无环正则与增广拉格朗日法（ALM）蒸馏出任务无关的因果骨架 $M_{inv}^*$（即 CRWM）。
  3. **因果自动奖励设计 Causal-ARD（IV-C）：** 对 $\{|M_{inv}^{*,i,j}|\}$ 做一维 2-means 聚类得到自适应阈值 $\tau = (\mu_{noise} + \mu_{signal}) / 2$，将活跃因果边序列化为三元组 `(source, target, weight)` 写入系统提示词，引导 LLM 对不连通至 `score` 的虚假分量执行系数归零剪枝，单次生成可执行 Python 奖励函数（$\text{ESI}=0$）。
- **关键实验数据与消融：**
  - **Dexterity 20 任务（14 训练 / 6 未见测试，DeepSeek-V3）：** 未见任务零样本平均成功率 **0.85**（`ESI=0`），其中 `CatchOver2Underarm` 达 **0.92**、`LiftUnderarm` 达 **0.87**、`Kettle` 达 **0.88**、`CatchAbreast` 达 **0.65**；相比 Eureka（需 4 轮 ESI）与 URDP（需 1–4 轮 ESI，每轮约 1 小时 / 8×A800 GPU）免除部署期进化搜索。
  - **消融实验（V-E, Table S-II）：** 平均 SR 从 0.85 分别降至 `w.o. Prior`（0.36）、`w.o. EMD`（0.35）、`w.o. Soft-Fusion`（0.38）、`w. LLM-Prior`（0.09）；`w.o. DAG` 在 Kettle/CatchAbreast/CatchOver2Underarm 分别从 0.88/0.65/0.92 降至 0.81/0.61/0.82。跨 LLM（DeepSeek-V3 / Qwen-2.5 / Llama3）与跨因果基础模型（LimiX / AVICI）保持稳健。
  - **跨本体与跨环境（V-F, ManiSkill2）：** 由 20 个 ShadowHand 任务蒸馏的 CRWM 零样本迁移至单臂操作任务，SR 分别达到 `PickCube` **0.64**、`OpenCabinetDoor` / `OpenCabinetDrawer` **0.89**、`TurnFaucet` **0.80**。
  - **真机离线轨迹对齐（V-G, CASBOT W1）：** 在 3 个任务各 25 条成功/25 条失败遥操作轨迹（各 625 对）上，成对可区分准确率（AUC）达 `PickCube` **99.6%**、`LiftUnderarm` **95.3%**、`Scissors` **94.5%**（均值 **≈96.5%**），显著高于 `ZS-LLM` 的 **≈53.9%**。
  - **失效模式统计（VI, Table II）：** 20 个任务上 Eureka 出现 21.2% 优化崩溃与 10.4% 规范博弈；URDP 出现 12.6% 优化崩溃与 4.7% 规范博弈。
  - **预训练开销（App. B-D, Table IV）与局限（App. C）：** 14 个训练任务、约 140 个奖励规格、$N=30$ 权重配置共需 4200 次 RL 训练（每次 3000 迭代，8×A800 GPU 约 50 批次、每批约 1 小时）；零样本泛化受限于全局原子奖励分量池覆盖范围（暂不支持流体、柔性体等未入池物理原语）；尚无双灵巧手真机闭环验证。
- 对 wiki 的映射：[Causal Reward World Models 实体页](../../wiki/entities/paper-agentic-2606-23280.md)
