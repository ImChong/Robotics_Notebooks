# Action Chunk Scheduling for Batched Robot Policy Serving

- **作者：** Rohan Bansal*, David He*, Nadun Ranawaka Arachchige, Zhenyang Chen, Soobum Kim, Kexin Rong†, Danfei Xu†
- **机构：** Georgia Institute of Technology（ARM Lab）
- **版本 / 载体：** arXiv:2608.00337v1 [cs.RO]（2026-07-31 提交，含正文 §1–§7 与 Appendix A.1–A.9、Table 1–5、Figure 1–10）
- **原文：** [arXiv HTML v1](https://arxiv.org/html/2608.00337) / [arXiv Abs](https://arxiv.org/abs/2608.00337) / [项目页](https://gatech-rl2.github.io/actionchunkscheduling)
- **项目页 / 代码：** 已开源（官方仓库 [`GaTech-RL2/armory`](https://github.com/GaTech-RL2/armory)，Apache-2.0 协议，包含服务端调度引擎、轻量机器人客户端、OpenPI 与 GR00T 适配器及 LIBERO 评测工具）
- **核对范围：** arXiv:2608.00337v1 全文（Abstract、§1–§7、Appendix A.1–A.9、Table 1–5、Figure 1–10 文字与数值）及官方项目页 [`actionchunkscheduling`](https://gatech-rl2.github.io/actionchunkscheduling)
- **通读：** 已完成
- **入库日期：** 2026-11-27

## 原文核对笔记

- **核心问题与闭环视角（§1–§2）：** 远端 GPU 通过批处理（batching）同时向多台机器人提供 VLA 策略服务，必须在 GPU 吞吐与机器人闭环响应性之间权衡。与常规 ML/LLM 服务不同，机器人执行动作分块 $A_t = [a_i, a_{i+H-1}]$（控制频率 $f_c$，执行前缀区间 $[H_{\min}, H_{\max}]$），端到端延迟分解为 $d = d_{\text{obs}} + d_{\text{infer}} + d_{\text{action}}$。若等待过久导致动作队列耗尽，机器人进入**饥饿（starvation）**并停顿抖动；若 $H_{\max}$ 过大导致动作过时，高动态任务性能急剧退化。
- **MDP 建模（§3.1）：** 将多机器人批处理调度建模为马尔可夫决策过程。对每个机器人 $j \in \{1,\dots,N\}$，服务端软件镜像跟踪状态 $s_k = (t_k, i_j, \hat{i}_j, \mathcal{Q}_j)$，其中 $i_j$ 为最新已执行动作索引，$\hat{i}_j$ 为服务端最新收到的观测索引，$\mathcal{Q}_j$ 存储已生成分块 $c = (i_{\text{start}}, h_j, t_{\text{arr}})$。每步选取非空批次 $B_k$，于 $t_{k+1} = t_k + \tilde{d}_{\text{infer}}(|B_k|)$ 完成推理并入队新分块，单步奖励为加权非饥饿执行时间 $R(s_k, B_k) = \frac{1}{f_c}\sum_{j=1}^{N} w_j \Delta i_j(s_k, B_k)$。
- **Lookahead 调度算法（§3.2 & Appendix A.1）：** 对比 Round Robin（RR）、Earliest Deadline First（EDF）与 Lookahead（LA）。Lookahead 利用 GPU 空闲前的时间裕度（`has_slack()`）持续在软件镜像上推演长度为 $L$ 的候选调度序列 $S = (s_0, B_0), \dots, (s_{L-1}, B_{L-1})$，按每单位 GPU 推理时间的累计奖励 $\text{Score}(S) = \sum_{\ell=0}^{L-1} \frac{R(s_\ell, B_\ell)}{d_{\text{infer}}(|B_\ell|)}$ 选出最优序列并下发首个批次 $B_0$（主实验采用 $L=1$，并通过快速层级权重 $w_j \in \{1, 3, 5\}$ 记作 `LA@w_j` 调控异构优先级）。
- **Armory 系统实现（§3.3 & Appendix A.9）：** 采用推式（push-based）WebSocket 通信与多进程共享内存解耦架构（Frontend、Scheduler、Engine 独立进程），启动时测量各 batch size 的 $d_{\text{infer}}(b)$ 并用 EMA 持续估计 $d_{\text{obs}}, d_{\text{action}}$；Engine 消费批次前还会剔除重复请求或未执行满 $H_{\min}$ 步的无效请求。
- **关键实验与消融（§4 & Appendix A.2–A.8）：**
  - **GPU 与模型推理剖面（Table 1–2）：** 单块 L40S 上 $\pi_{0.5}$ 在 $b=1..5$ 的延迟为 $73.0 / 109.8 / 142.4 / 177.5 / 211.1\text{ ms}$（H100 上为 $42.3..87.2\text{ ms}$）；GR00T N1.7 在 L40S 上更具内存受限特征，延迟为 $71.8 / 76.5 / 83.5 / 91.8 / 100.9\text{ ms}$。
  - **LIBERO-10 仿真（Table 4，$N=2..10, b=3$）：** 在 `all-fast` 同构场景下 RR/EDF/LA 总体表现接近（$N=10$ 时总吞吐 29.23–31.02 successes/min）；在 `one-fast` 与 `half-fast` 异构场景中，Lookahead 可通过 $w_j$ 显式调节快/慢层级吞吐配比（`half-fast` 下 `LA@3` 在保持慢层接近基线的同时提升快层服务）。
  - **10 台 AgileX PiPER 真机（Table 5，$f_c=30\text{ Hz}, H=20$）：** 静态分拣（slow，$H_{\max}=20$）与动态转盘抓取（fast，$H_{\max}=10$）混跑。`one-fast` 下 `LA@5` 将快层吞吐从 EDF 的 3.67（RR 2.00）提升至 **8.33** legos/min（约 2 倍于 EDF、4 倍于 RR），同时慢层达 10.04（EDF 8.81），系统总吞吐由 83.00 升至 **98.67** legos/min（约提升 18%）。`half-fast` 下 `LA@5` 将快层从 3.13 提至 6.13，同时以慢层从 9.40 降至 6.53 为代价实现可控权衡。
  - **批大小、步长与网络消融（A.4–A.6）：** 批上限设为 10 时 EDF/RR 均值批大小达 4.7 导致过批与饥饿攀升，Lookahead 自发选出更紧凑批次（均值 3.3、中位数 3）；规划深度 $L \in \{1,2,3\}$ 增大反而略增饥饿率；在 $m=50\text{ ms}$ 对数正态抖动与 50–100 ms 单向延迟下系统仍保持稳健。
- **核查边界：** 原文 §7 指出当前引擎假设单 GPU 静态批处理，尚未结合连续批处理（continuous batching）、PagedAttention 或预填充-解码分离；真机评测受限于 10 臂环境搭建成本，每组仅跑 3 个随机种子、每次 1 分钟窗口，且采用朴素异步执行（Appendix A.8 观察到 Lookahead 更倾向从索引 0 开始执行分块，从而在无意中降低了中段切换抖动）。
- **对 wiki 的映射：** [Action Chunk Scheduling for Batched Robot Policy Serving](../../wiki/entities/paper-agentic-2608-00337.md)
