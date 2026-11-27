# MEMENTO: Memory-Guided Memetic Code-as-Policy Evolution

- **作者：** Alkis Sygkounas, Victor Aregbede, Amy Loutfi, Andreas Persson
- **机构：** Center for Applied Autonomous Sensor Systems (AASS), Örebro University, Sweden；Linköping University, Sweden
- **版本 / 载体：** arXiv:2607.22832v1 [cs.LG]（2026-07-24 提交）
- **原文：** https://arxiv.org/abs/2607.22832（全文核对使用 arXiv HTML 版 https://arxiv.org/html/2607.22832）
- **项目页 / 代码：** https://github.com/sygkounas/MEMENTO（已开源：含核心演化搜索代码 `MEMENTO/`、提示词模板 `MEMENTO/prompts/`、配置 `MEMENTO/cfg/`、最优演化策略 `best_policies/`、物理 Franka 部署代码 `hanoi/` 及演示视频；GitHub 仓库元数据未标明开源协议）
- **核对范围：** 通读 arXiv:2607.22832v1 全文（Abstract、Sections 1–7、Appendices A–C、Algorithms 1–2、Tables 1–11、Figures 1–9 题注）及官方 GitHub 仓库主页 `https://github.com/sygkounas/MEMENTO`（README.md 与文件清单）。
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[MEMENTO 实体页](../../wiki/entities/paper-agentic-2607-22832.md)

1. **研究问题与核心动机**
   - 长时域具身控制任务需要连续执行多个强依赖动作后才能观测到最终任务成功（$\sigma(\tau)\in\{0,1\}$），存在长时域信用分配难题。
   - **Code-as-Policy** 将策略表示为可执行控制程序 $p\in\mathcal{P}$，使决策逻辑可检查、可修改；但仅依赖稀疏二值成功信号不足以定位程序哪一部分需要修改，而已有基于 LLM 的演化方法（如 Eureka、REvolve）主要在独立生成的候选变体间选择，缺乏**针对已接受候选的顺序局部精炼（local search）**以及**失败修改记忆（rejected-proposal memory）**。
   - MEMENTO 提出**记忆引导的单精英模因框架（memory-guided single-elite memetic framework）**：先演化出一个固定评估器 $E^\star$，再在其提供的标量适应度 $F(p)$ 与结构化反馈指标 $\mathbf{M}(p)$ 指导下执行三分支模因策略演化。

2. **两阶段方法架构（Algorithms 1–2）**
   - **阶段一：评估器演化（Evaluator Evolution，Section 3.1 / Algorithm 1）**
     - 对初始策略 $p_0$ 执行 $N_{\mathrm{roll}}=10$ 回合得到固定轨迹集 $\mathcal{T}_0$ 与经验成功率 $S_0$。
     - 维护大小 $n=8$ 的评估器程序池 $\mathcal{E}_i$，共演化 $T_{\mathrm{eval}}=5$ 代。每代将所有评估器作用于 $\mathcal{T}_0$，输出 $(F_i^{(j)}(p_0), \mathbf{M}_i^{(j)}(p_0))$，并与 $S_0$ 一起送入 **LLM-as-judge 多数投票**（$k=10$ 次独立判断）选出 $E_i^\star$，再用评估器宏变异算子 $q_{\mathrm{eval}}^{\mathrm{macro}}$ 生成下一代池。最终冻结 $E^\star = E_{T_{\mathrm{eval}}-1}^\star$。
   - **阶段二：固定评估器下的单精英模因策略演化（Section 3.2 / Algorithm 2）**
     - 维护单精英状态 $\mathcal{C}_g = (p_g, F(p_g), \mathbf{M}(p_g), S(p_g))$，每代预算 $B = 24$ 个候选策略（$G=5$ 代，每候选评估 $N_{\mathrm{roll}}=10$ 回合，模型统一采用 GPT-5.2）：
       1. **爬山局部搜索分支（Hill-climb，$K_{\mathrm{hc}}=10$）**：从 $\mathcal{C}_g$ 出发顺序进行微变异（micro-mutation）；仅当 $F(\mathcal{C}^{\mathrm{new}}) \ge F(\mathcal{C}^{\mathrm{hc}})$ 时接受更新，否则将该候选的修改摘要 $\mathrm{history}(p)$ 追加到**拒绝修改记忆 $\mathcal{H}$**，用于阻止后续重复尝试已失败的局部修改方向。
       2. **宏变异分支（Macro-mutation，$K_{\mathrm{macro}}=10$）**：以 $\mathcal{C}_g$ 为父本独立采样较大结构改写并择优得到 $\mathcal{C}_g^{\mathrm{macro}}$。
       3. **交叉分支（Crossover，$K_{\mathrm{cross}}=4$）**：以 $\mathcal{C}_g^{\mathrm{hc}}$ 与 $\mathcal{C}_g^{\mathrm{macro}}$ 为双亲进行策略重组，取最高适应度候选 $\mathcal{C}_g^{\mathrm{cross}}$。
       4. **精英选择**：在 $\{\mathcal{C}_g^{\mathrm{hc}}, \mathcal{C}_g^{\mathrm{macro}}, \mathcal{C}_g^{\mathrm{cross}}\}$ 中选出适应度最高者作为下一代精英 $\mathcal{C}_{g+1}$。

3. **两大实验域与接口设计（Section 4 & Appendices A–B）**
   - **Robosuite Franka Tower-of-Hanoi（全观测长时域操作）**：4 个不同尺寸立方体（half-edge $0.022, 0.025, 0.028, 0.031\text{ m}$）、3 个带孔容器（hole center 偏移 $h_{\mathrm{local}}=[0.055,0,0]^\top$），需从 A 移至 C（最少 $2^4-1=15$ 步合法抓放），最长 12,000 步；策略类 `FrankaPolicy` 输出 7 维连续末端增量与夹爪动作，训练启用**重度域随机化（Table 3）**（滑动摩擦 $0.3\text{--}1.8$、位置噪声标准差 $0.002\text{--}0.025$、位置偏置 $\pm 0.015$、质量缩放 $0.7\text{--}1.5$ 等）。
   - **AI2-THOR 家居交互（部分可观测长时域交互）**：任务 `apple_in_microwave_on_bread_in_fridge`，最大 250 步，仅接收第一人称可见物体摘要及当前合法交互对象 ID（`obs["action_space"][a_t]`）。训练用 `FloorPlan1`–`FloorPlan3`，跨场景泛化测试用未见厨房 `FloorPlan4`–`FloorPlan29`。

4. **主实验、泛化与消融结果（Section 5, Tables 1–11, Appendix C）**
   - **主演化对比（3 个随机种子，第 5 代，Section 5.1）**：
     - Robosuite：MEMENTO 达到适应度 **$0.99 \pm 0.02$**、成功率 **$0.97 \pm 0.06$**；REvolve（适应度替换为主评估器 $E^\star$）为 $0.72 \pm 0.08$ / $0.53 \pm 0.12$；Eureka 为 $0.45 \pm 0.04$ / $0.00 \pm 0.00$。
     - AI2-THOR：MEMENTO 达到适应度 **$0.98 \pm 0.00$**、成功率 **$1.00 \pm 0.00$**；REvolve 为 $0.67 \pm 0.05$ / $0.40 \pm 0.10$；Eureka 为 $0.00 \pm 0.00$ 成功率。
     - 评估器对齐相关性（54 组配对值）：Robosuite Pearson $r=0.922$、Spearman $\rho=0.755$；AI2-THOR Pearson $r=0.871$、Spearman $\rho=0.801$。
   - **泛化与 RL 基线对比（Table 1）**：
     - Robosuite Held-out 物体几何（Asym. 非对称带臂堆叠体 / Cyl. 圆柱盘）：MEMENTO 保持 **$0.87 \pm 0.23$**（Asym.）与 **$0.63 \pm 0.17$**（Cyl.）；REvolve 跌至 $0.23 \pm 0.15$ / $0.00 \pm 0.00$；Eureka、SAC+HER sparse（50M 步）与 SAC+HER dense（50M 步）在 Cube 上均为 $0.00 \pm 0.00$。
     - AI2-THOR Held-out 未见厨房（`FloorPlan4`–`FloorPlan29`）：MEMENTO 达到 **$0.78 \pm 0.08$**，REvolve 仅 $0.08 \pm 0.04$，Eureka 为 $0.00 \pm 0.00$。
   - **消融实验（Section 5.3）**：
     - Zero-shot（24 候选无演化）：Robosuite $0.37\pm 0.12$ 适应度 / $0.00$ 成功率；AI2-THOR $0.22\pm 0.10$ / $0.00$。
     - 未演化评估器（No-evaluator-evolution）：Robosuite $0.42\pm 0.05$ / $0.00$；AI2-THOR $0.33\pm 0.05$ / $0.00$。
     - 算子分支消融：Hill-only（$K_{\mathrm{hc}}=24$）适应度 Robosuite $0.50\pm 0.04$、AI2-THOR $0.46\pm 0.04$（均 $0.00$ 成功）；Without-Crossover（$K_{\mathrm{hc}}=12, K_{\mathrm{macro}}=12$）为 $0.55\pm 0.02$ / $0.51\pm 0.04$（均 $0.00$ 成功）。
     - 搜索场景多样性（仅用单厨房演化后测试 29 个厨房）：成功率跌至 $0.09 \pm 0.02$（$2/29, 3/29, 3/29$）。

5. **真机 Sim-to-Real 部署（Section 5.4 & Appendix A.4）**
   - 将 Robosuite 最优演化策略 $p_{\mathrm{best}}$（保留递归 15 步 Hanoi 调度、EMA 滤波、抓握校验、居中释放与放置后校验逻辑）固定部署到物理 Franka 机械臂。
   - 使用 RGB-D + SAM3 语义分割实时估计立方体位姿（容器位置标定一次固定），并引入仅输出执行反馈指标的精简真机评估器 $E_R$ 执行 $N_{\mathrm{cal}}=3$ 步 LLM 辅助执行常数校准（`grasp_depth_offset`: $0.002 \to 0.045\text{ m}$ 等 4 项常数，Table 4）。
   - 在 $N_{\mathrm{real}}=10$ 次物理试验中取得 **$0.90$（9/10）** 成功率，唯一失败源自感知误差。
