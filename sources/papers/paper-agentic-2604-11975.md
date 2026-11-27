# M2HRI: An LLM-Driven Multimodal Multi-Agent Framework for Personalized Human-Robot Interaction

- **作者：** Shaid Hasan, Breenice Lee, Sujan Sarker, Tariq Iqbal
- **机构：** University of Virginia (Collaborative Robotics Lab)
- **版本 / 载体：** arXiv:2604.11975v2 [cs.RO]（2026-04-13 提交 v1，2026-06-21 修订 v2）
- **原文：** [arXiv HTML v2](https://arxiv.org/html/2604.11975) / [arXiv Abs](https://arxiv.org/abs/2604.11975)
- **项目页 / 代码：** 已开源（[项目页](https://project-m2hri.github.io/) 提供论文与代码入口；官方 GitHub 仓库 [project-m2hri/m2hri](https://github.com/project-m2hri/m2hri) 公开了服务端、客户端、人格、记忆与动作控制代码；详见 [仓库档案](../repos/paper-agentic-2604-11975.md)）
- **核对范围：** arXiv:2604.11975v2 全文（Abstract、§I–§VII、Eq. 1–7、Fig. 1–4 说明及 References）、项目页全文，以及官方仓库 `project-m2hri/m2hri` 的 `README.md`、`ma_server.py`、`ma_clients.py` 与 `modules/` 核心源码
- **通读：** 已完成
- **入库日期：** 2026-11-27

## 原文核对笔记

- **核心动机与问题定义（§I–§II）：** 现有面向社交场景的多机器人人机交互（Multi-Agent HRI）系统常把机器人视作功能可互换的同质执行体，忽略了“个体身份（人格 + 长期交互历史）”对用户感知的影响，以及个体差异给轮次协调带来的新约束。论文提出三个研究问题：RQ1 机器人人格如何塑造用户对单体的感知；RQ2 智能体专属长期记忆如何影响个性化与情境意识；RQ3 情境化多智能体协调如何影响对话与交互质量。
- **M2HRI 形式化架构（§III-A，Eq. 1–5）：** 每个机器人智能体建模为 $a_i = \langle \phi_i, \Lambda_i, \Gamma_i \rangle$，其中身份画像 $\phi_i = \langle M_i, B_i \rangle$、认知循环 $\Lambda_i = \langle P_i, R_i, E_i \rangle$、动作能力集为 $\Gamma_i$：
  1. **感知 $P_i$（§III-A1，Eq. 2）：** 接收多模态输入 $\mathbf{x}_t = \langle x_t^{\text{speech}}, x_t^{\text{vision}} \rangle$，由 VLM 生成融合场景上下文与非语言线索的语义文本观测 $o_t = P_i(x_t^{\text{speech}}, x_t^{\text{vision}})$。
  2. **记忆 $M_{a_i}$（§III-A2）：** 分为工作记忆 $c_t = \{o_{t-k}, \dots, o_t\}$（会话内滑动窗口、每次会话重置不落盘）、语义记忆（用户/对象/环境事实）与情景记忆（带时间锚点的偏好与历史交互）。检索与写入由基于 LLM 的记忆控制器执行：$m_t = R_{\text{mem}}(o_t, M_{a_i})$、$M_{a_i} \leftarrow M_{a_i} \cup S_{\text{mem}}(o_t, M_{a_i})$。
  3. **人格 $B_{a_i}$（§III-A3，Eq. 3）：** 基于五因素模型（FFM / Big Five）构建五维离散向量 $\mathbf{B}_{a_i} = (O, C, E, A, N) \in \{1,2,3,4,5\}^5$（$3$ 为中性），经确定性映射 $f_{\text{trait}}$ 转为语言修饰词写入系统提示。
  4. **交互规划器 $R_i$ 与执行器 $E_i$（§III-A4–5，Eq. 4–5）：** 规划器输出参数化动作原语序列 $\pi_t = \{(a_1, \theta_1), \dots, (a_k, \theta_k)\}$（$a_j \in \mathcal{A}$），执行器将其映射为受 $\Gamma_i$ 约束的具身行为 $u_t = E(\pi_t)$（如 `gesture -> speak -> nod`）。
- **情境化多智能体协调（§III-B，Eq. 6–7）：** 保持各智能体分散式感知-规划循环的同时引入中心化协调器：汇聚 $\mathcal{Z}_t = \{z_t^{(i)}\}_{i=1}^N$（$z_t^{(i)} = (o_t^{(i)}, c_t, B_{a_i})$），由基于 LLM/VLM 的协调函数计算参与适配度评分 $r_{i,t} = f_{\text{coord}}(\mathcal{Z}_t)_i \in [0, 1]$，筛选 $S_t = \{i \mid r_{i,t} \ge \tau\}$ 后按序生成并执行响应。
- **系统实现（§III-C）：** 在两台配备板载 RGB 相机、麦克风与扬声器的 NAO 人形机器人上部署。外部 LLM 工作站运行 `gpt-4o-mini`（AutoGen `AssistantAgent` 工具调用规划与视觉感知）与 `gpt-4o`（轮次打分与记忆代理），向量检索采用 ChromaDB + `text-embedding-3-large` 并由 LangGraph 记忆代理调度，控制指令通过 ZMQ 序列化下发至 Python 2.7 / NaoQi 客户端执行。
- **受控用户研究与关键数据（§IV–§V，$n=105$）：**
  - 通过 Prolific 招募 105 名被试（年龄 18–77，$M=39.3, SD=13.7$；男 66 / 女 38 / 非二元 1），分 7 个实验组（每组 15 人，5 个被试间人格组 + 1 个被试内记忆配对组 + 1 个被试内协调配对组），采用约 10 分钟预录视频观察范式（IRB 批准）。
  - **RQ1 人格（1 vs. 5 对比，其余维度设为 3）：** 区分度相对中性值 3.0 在 O（$M=4.73, p<.001, d=3.79$）、N（$M=4.80, p<.001, d=4.35$）、A（$M=4.47, p<.001, d=2.29$）、E（$M=4.47, p<.001, d=2.84$）均显著，但尽责性 C（$M=3.53, p=.120, d=0.43$）未达显著；一致性 5 个维度全部显著（$p<.001$，N 最高 $M=4.87, d=5.31$）；参与度仅 O（$M=4.40, p<.001$）、N（$M=4.33, p<.001$）、A（$M=4.13, p=.003$）显著高于中性，E（$M=3.40$）与 C（$M=3.33$）未显著偏离中性；特质识别准确率相对随机基准 $33.3\%$ 均显著（$p \le .0085$）：O/E $93.3\%$、A/N $86.7\%$、C $66.7\%$。
  - **RQ2 记忆（$n=15$ 配对检验）：** 回忆准确率 $t(14)=6.52, p<.001, d=1.68$；偏好意识 $t(14)=12.65, p<.001, d=3.27$；交互自然度 $t(14)=3.09, p=.008, d=0.80$。
  - **RQ3 协调（$n=15$ 配对检验）：** 对话流畅度 $t(14)=4.08, p=.001, d=1.05$；响应恰当性 $t(14)=4.00, p=.001, d=1.03$；避免插话重叠 $t(14)=7.89, p<.001, d=2.04$。
- **核查边界：** 用户研究基于双 NAO 机器人预录视频而非真人现场多轮自由部署；论文未报告实体系统端到端推理延迟、并发 API 成本或长期多日跨会话稳定性指标。
- **对 wiki 的映射：** [M2HRI: An LLM-Driven Multimodal Multi-Agent Framework for Personalized Human-Robot Interaction](../../wiki/entities/paper-agentic-2604-11975.md)
