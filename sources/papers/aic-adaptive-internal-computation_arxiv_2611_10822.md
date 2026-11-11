# Learning When to Think: Adaptive Internal Computation for RL（arXiv:2611.10822）

> 来源归档（paper）

- **标题：** Learning When to Think: Adaptive Internal Computation for Reinforcement Learning
- **类型：** paper
- **作者 / 机构：** Rohit Kumar Salla, Simon Stepputtis（Virginia Tech）
- **会议：** NeurIPS 2026（论文首页脚注与项目页均标 40th NeurIPS / Accepted）
- **arXiv：** <https://arxiv.org/abs/2611.10822>
- **PDF：** <https://arxiv.org/pdf/2611.10822>（读取版本 v1，2026-11-07）
- **项目页：** <https://rohitsalla.github.io/aic-neurips2026/>
- **代码：** 未开源。项目页 Paper / Code 两个入口均显示 "coming soon"（2026-11-11 访问）；论文附录 O 写 "the code release includes their seed-level records"，但未给仓库地址。
- **入库日期：** 2026-11-11
- **一句话说明：** 迭代式策略（共享残差块反复精炼隐表示）把「停在第几步」与「动作」作为同一个联合决策，用 PPO 同时训练；在 12 个仿真任务上平均执行 3.3 步（上限 8），相对固定 8 步策略在 4 个控制 / 机器人任务上单次决策延迟降低 31–44%，但在 FourRooms、Puzzle 等任务回报低于固定深度基线。
- **核对范围：** 读了 arXiv v1 PDF（pdftotext 文本）正文第 1–6 节全文、参考文献，以及附录 A（方法细节）、B（定理 1 的设定 B.1 与序贯回报引理 B.5）、C（C.1、C.2、C.4 及反例）、D（任务）、E、F（析因）、G（停止基线）、H（深度统计，表 10）、I（超参与敏感性，表 11–12）、J（特权参考，表 13）、L–N（截断方法、关键状态、任务级分析）、O（局限）、P（记忆任务）、Q（训练成本与延迟）、R（迁移细节）、S（协议）与表 14–21 全部数值。未逐步核对的部分：附录 A.3–A.11 的推导细节、B.2–B.4 与 C.3、C.5–C.7 的证明步骤，只读了命题陈述；图 1–5 只读图注与对应数据表，未看图像；项目页 HTML 静态读过全文文字。未运行任何代码。
- **通读：** 已完成

## 开源状态

- **未开源**（步骤 2.5，2026-11-11）：项目页 Code 入口为 "coming soon"，论文未给仓库；因此 wiki 页不写 `code:` 字段。

## 核心摘录

1. **方法（§3，式 1–3）：** 编码器 f_enc 得 z0，共享残差块 z_d = z_{d-1} + MLP(z_{d-1})，d=1…d_max（主配置 d_max=8，d_z=256）；每个深度有动作头 π_d、价值头 V_d 与停止概率 h_d（d<d_max，上限处强制停止）。联合分布 p(a,d|s)=q(d|s)·π_d(a|z_d)，PPO 比率同时含停止概率与动作概率；损失 L = L_PPO + (λV/d_max)·Σ_d E(V_d − 目标)² + λc·E[C(s)]，C(s)=(1/d_max)·Σ_d S_d(s)（期望深度惩罚）。评测时取首个 h_d>0.5 的深度（§3，附录 A.12）。
2. **理论（§3.2，定理 1）：** 若仅部分状态需要更深精炼，则理想的状态相关分配器在同一期望深度下优于所有状态无关的停止分布（含随机化）；这是理想分配器（知道难易划分）的存在性结论，不保证学习得到（§3.2、项目页同样声明）。
3. **主结果（§4.1，表 1，B=8）：** 相对按预算调参的固定深度基线，9 个任务更高（7 个 p<.05，Welch t 检验，5 个种子），DistShift 持平，FourRooms（−.0216）与 Puzzle（−.0047）更低；对「任一预算 B′≤8 的最强固定基线」，7 个任务领先（3 个 p<.05）、1 个持平、4 个落后（RareEvent −.0420、AntMaze low-het −.0140、FourRooms、Puzzle）。12 任务平均深度 3.32（表 10）。
4. **延迟（附录 Q，表 19）：** A100 上中位单决策延迟，HalfCheetah 0.86→0.48 ms（−44%）、AntMaze −40%、PickCube −38%、StackCube −31%；GPT-2/RTE 因冻结编码器占主导，49% 更少精炼步只换来 −7.8% 延迟；PickCube 训练耗时 5.9 h→7.2 h（1.22×）。
5. **POGP 对比（§4.1，表 2）：** 四个 MuJoCo 任务（仅评估每回合前 100 步，附录 D.2），AIC 每动作迭代数少 22–28%，回报与 POGP 在 3–5 以内（<0.3 标准差）。
6. **归因（§4.2，表 4–5）：** 数据源互换实验显示增益需「AIC 数据 + AIC 学习器」同时具备；停止 × 监督析因显示 PickCube、AntMaze 以自适应停止为主，RareEvent 以价值监督为主；论文自承该析因中自适应配置同时多执行 1.8–3.9 步，未做等计算量对照（附录 F）。
7. **迁移（§5，表 7）：** 挂到 DiffusionQL、DPPO、D3P、DSAC-D、IDQL 五类扩散策略（每次精炼=一次去噪步），八任务中六个任务回报提升，FourRooms 五类全部下降。
8. **局限（§5、附录 O）：** 仅小型精炼模块与仿真；放大到大模型与真机是未来工作；FourRooms 在所测损失权重网格内始终低于平均深度匹配的固定基线；λc、λV 只在 PickCube 上选定并共享。

## 对 wiki 的映射

- [paper-aic-adaptive-internal-computation](../../wiki/entities/paper-aic-adaptive-internal-computation.md)
