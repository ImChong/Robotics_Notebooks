# RDA: Reward Design Agent for Reinforcement Learning

- **作者：** Hojoon Lee (Meta, KAIST)、Ajay Subramanian (Meta, NYU)、Ben Abbatematteo (Meta)、Vijay Veerabadran (Meta)、Pedro Matias (Meta)、Karl Ridgeway (Meta)、Nitin Kamra (Meta)
- **机构：** Meta、KAIST、NYU（通讯作者邮箱：`joonleesky@kaist.ac.kr`、`nitinkamra@meta.com`）
- **版本 / 载体：** arXiv:2606.01672v1 (`[cs.LG]`, 2026-06-01)，Accepted to RLC'26（Reinforcement Learning Conference 2026）
- **原文：** [arXiv HTML (2606.01672v1)](https://arxiv.org/html/2606.01672)、[arXiv 摘要页](https://arxiv.org/abs/2606.01672)
- **项目页 / 代码：** [项目主页](https://nitinkamra1992.github.io/reward-design-agent) — **部分开源**（项目页公开了 12 个 ManiSkill 任务与 4 个 HumanoidBench 任务的演示视频及最佳生成奖励函数文本片段 `rewards/rda/*.txt`，网站采用 NeRFies/Eureka 模板；截至 2026-11-27，原文与项目页均未提供可运行的 RDA 框架官方代码仓库。文中参考文献引用的 `https://github.com/krafton-ai/Prompt2Policy` 为并发工作 Chung et al., 2026 的仓库，非本论文源码）
- **核对范围：** 全文通读 arXiv HTML v1 正文 §1–§6（Introduction、Related Work、Preliminary、RDA、Experiments、Conclusion）与附录 §7–§12（完整 5 类 Prompt 模板、PlugCharger 与 HumanoidBench Package 完整演化案例、表 1 超参数与算力消耗、GPT-4.1 对齐率评估细则、Human Sparse/Dense 与 Eureka 基线实现细节），以及项目页全部文本与任务列表（未本地运行或复现训练）。
- **通读：** 已完成

## 原文核对笔记

### 1. 问题形式化与核心框架（§3–§4）

- **问题定义（Definition 3.1, RDP）**：将 Singh et al. (2009) 的奖励设计问题拓展到指令条件设定 $P = \langle \mathcal{I}, \mathcal{E}, \mathcal{R}, \mathcal{L}, \mathcal{S} \rangle$，目标是寻找可执行奖励函数 $R^\star \in \arg\max_{R \in \mathcal{R}} S(L(E, R), I)$，使强化学习算法 $L$ 训练出的策略在行为上对齐自然语言指令 $I$。
- **两阶段流水线**：
  1. **初始化（§4.1）**：VLM 读取自然语言指令 $I$、含观测定义与成功判定条件的环境源码 $E$ 及初始环境图像，先分解出 $J$ 个有序、可测量的子任务 $T^1 = [t_1^1, \dots, t_J^1]$（简单任务约 5 个，长程复杂任务约 10 个）；随后生成 $N$ 个结构化的可执行 Python 奖励候选 $\{R_1^1, \dots, R_N^1\}$（各分量经 `exp`/`tanh` 归一化至 $[0,1]$ 后按显式权重加权求和，并附加终止成功奖励与 `self.reward_info` 日志字典），经编译校验后进入演化循环。
  2. **演化搜索（§4.2，共 $M=5$ 轮）**：
     - **策略训练与回放**：对每个候选奖励 $R_n^i$ 用 SAC + SimbaV2 训练策略 $\pi_n^i$（可从上一轮最优 checkpoint 热启动），采样 $K=3$ 条轨迹 $\{\tau_{n,k}^i\}_{k=1}^K$ 并记录渲染帧与逐步 `reward_*` 数值日志。
     - **视觉轨迹分析**：VLM 按子任务顺序逐一评估（3 档评分 $s_{n,k,j}^i \in \{0.0, 0.5, 1.0\}$），同时引用图像帧行为描述与关键时间步的 `reward_*` 数值输出诊断理由 $\rho_{n,k,j}^i$。
     - **汇总与选优**：计算平均子任务视觉得分 $\bar{s}_n^i = \mathbb{E}_{k,j}[s_{n,k,j}^i]$ 与各子任务失败模式汇总 $p_{n,j}^i$，选出最优候选 $n_i^\star = \arg\max_n \bar{s}_n^i$。
     - **双层反思（Subtask & Reward Reflection）**：先基于诊断修正引发非预期行为的最关键子任务（保持总数 $J$ 不变），再联合 $(I, E, T^{i+1}, R_{n_i^\star}^i, G_{n_i^\star}^i)$ 修改奖励塑形系数、权重或引入门控项（如 `bilateral_contact`），生成下一代候选池。

### 2. 实验设置、主结果与消融（§5、§9–§12）

- **基准与训练预算（§5.1、Table 1）**：
  - **ManiSkill3（12 个桌面操作任务）**：`LiftPegUpright`、`PegInsertionSide`、`PickCube`、`PlaceSphere`、`PlugCharger`、`PokeCube`、`PullCube`、`PullCubeTool`、`PushCube`、`PushT`、`RollBall`、`StackCube`；每轮 $N=4$ 候选，50M 环境步（250K 更新步，2000 并行环境，§5.1 正文写 2048），$\gamma=0.95$，单任务约 20–40 A100 GPU 小时。
  - **HumanoidBench（4 个全身操作任务）**：`h1hand-basketball`、`h1hand-door`、`h1hand-package`、`h1hand-push`；每轮 $N=8$ 候选，10M 环境步（625K 更新步，16 个 MuJoCo 环境，最大步长 500），$\gamma=0.98$，单任务约 240–360 A100 GPU 小时。
  - **VLM 配置**：智能体主干使用 GPT-5 (`2025-08-07`, medium reasoning, 每次查询 20 张图)；对齐率自动评测使用 GPT-4.1 (`2025-04-14`, 每次查询 50 张图，每策略 5 段视频 × 4 次独立打分，1–5 李克特量表归一化至 $[0,1]$，3 个随机种子共 60 次评测/任务) 以避免生成器-评测器同源偏差。
- **核心数值结论（§5.2–§5.4）**：
  - **ManiSkill**：RDA 对齐率 **0.95** / 成功率 **0.90**；Eureka（同配 GPT-5 与预算）**0.87** / **0.87**；Human Dense **0.68** / **0.74**；Human Sparse **0.28** / **0.17**。仅 RDA 成功解决高难度插拔任务 `PlugCharger`。
  - **HumanoidBench**：RDA 对齐率 **0.70** / 成功率 **0.42**；Eureka 对齐率 **0.47**（RDA 相对提升 49%），成功率略高于 RDA；Human Dense **0.25** / **0.08**。案例 `Package Delivery` 中，RDA 第 1–2 轮即达 1.0 任务成功但对齐率仅 0.30（躯干顶推），经双层反思引入双手接触门控后第 4–5 轮对齐率升至 0.68（Eureka 为 0.28，Human Dense 为 0.08）。
  - **消融（PlugCharger + Package 3 种子均值）**：Eureka 基线（0.31 对齐 / 0.50 成功）→ `+Ckpt`（0.35 / 0.55）→ `+Traj Analysis`（0.82 / 0.83）→ 完整 RDA（0.84 / 1.0）。候选数 $N$ 从 1 → 2 → 8 时，对齐率从 0.46 → 0.58 → 0.84，成功率从 0.20 → 0.51 → 1.0。

### 3. 局限性与知识映射（§6）

- 三大核心局限：（1）多候选演化导致 RL 重训算力开销高；（2）VLM 上下文长度、推理成本及细粒度视觉判断能力限制（因而采用单轨迹独立评分而非成对比较）；（3）前置子任务奖励可能抑制后续子任务探索（如 `Humanoid Basketball` 中接球奖励抑制投篮探索）。
- 对 wiki 的映射：[RDA 独立实体页](../../wiki/entities/paper-agentic-2606-01672.md)
