# RoboRouter: Training-Free Policy Routing for Robotic Manipulation

- **作者：** Yiteng Chen, Zhe Cao, Hongjia Ren, Chenjie Yang, Wenbo Li, Shiyi Wang, Yemin Wang, Li Zhang, Yanming Shao, Zhenjun Zhao, Huiping Zhuang, Qingyao Wu
- **机构：** South China University of Technology, Nanjing University, University of New South Wales, Southwest Jiaotong University, Xiamen University, The Hong Kong Polytechnic University, ShanghaiTech University, University of Zaragoza
- **版本 / 载体：** `arXiv:2603.07892v4 [cs.RO]`（2026-06-24 更新，v1 提交于 2026-03-09）
- **原文：** [arXiv HTML](https://arxiv.org/html/2603.07892) / [arXiv Abstract](https://arxiv.org/abs/2603.07892)
- **项目页 / 代码：** 宣称将开源或未列链接（第 1 节提及 "We will release our framework to facilitate future research"，截至 2026-11-27 核查 `arXiv:2603.07892v4` 原文及检索缓存未列独立项目页或 GitHub 代码仓库链接）
- **核对范围：** 已通读 `arXiv:2603.07892v4` HTML 全文（Abstract、§1–§5、Appendix A–C、Table 1–6、Figure 1–8 图注及参考文献）；未发现官方开源代码仓库
- **通读：** 已完成

## 原文核对笔记

- **核心问题与动机：** 现有机器人操作策略（VLA、扩散/视觉-动作 VA 策略、基于代码组合的策略）各自只在特定任务与数据分布上表现突出，遇到域外（OOD）视觉或几何变化时容易退化，而训练更大规模的单体通才模型数据与算力代价高昂。RoboRouter 将范式从“训练单一全能策略”转向“在异构现成策略池中按任务实例进行免训练路由”。
- **形式化定义：** 给定视觉观测 $o_t$、自然语言指令 $l$ 与候选策略池 $\mathcal{P}=\{\pi^{(m)}\}_{m=1}^M$，系统构建多模态任务表征 $z=g(o_t,l)$，基于历史经验评估各策略得分 $S_\pi(z)$，通过 $\pi^\star(o_t,l)=\arg\max_{\pi\in\mathcal{P}} S_\pi(g(o_t,l))$ 选择单一最合适策略执行动作 $a_t\mid o_t,l\sim\pi^\star$。
- **四智能体闭环架构：**
  - **Retriever：** 先用视觉基础模型（如 SAM 2）从图像提取任务物体等元数据，再用 `Ops-MM-embedding-v1-2B` 将指令、图像与元数据编码为多模态嵌入存入 `Qdrant`；检索时按余弦相似度取 top-$k$ 历史记录，并用 `Qwen3-VL-2B-Instruct`（基于 yes/no token 概率的单次前向批量打分）重排，缓解长上下文“lost in the middle”位置偏置。
  - **Router：** 基于 LLM/VLM（主实验用 GPT-4o），结合重排后的相似任务历史记录与结构化 **Router Context**，预测并调用最优策略执行当前回合。
  - **Evaluator：** 基于 VLM 的单次执行分析器；将视频采样为关键帧（默认首帧+尾帧+2 个均匀中间帧），配合任务定制 Python 工具（仿真中读取物体位姿等状态；真机中用 DINOv2 + SAM + k-means 提取 3D 语义关键点计算距离与接触）生成元数据，再以 VQA 压缩输出成功摘要或具体失败模式诊断。
  - **Recorder：** 将 Evaluator 结果写入双层记忆——向 `Qdrant` 数据库增量添加带多模态索引的执行记录，并按条目化结构增量更新 Router Context（唯一 ID 原位修改 + 嵌入去重合并，避免全量重写导致细节丢失）。
- **三条核心工作流：** (1) **推理管线**：构建 $z$ → 检索与重排 → Router 选策略 → 直接执行；(2) **新策略接入（Onboarding）**：单任务策略仅在对应训练任务评估，多任务策略按 $z$ 聚类并在簇代表任务评估（经验上每任务约 10 次试跑即可通过 Evaluator 刻画能力边界）；(3) **在线反馈（Online Feedback）**：在并行线程异步运行，不增加在线推理额外时延，支持研究人员介入修正 VQA 摘要或指定次优策略对照采样（Human-in-the-loop）。
- **关键实验数据（核对自 Table 1–6）：**
  - **RoboTwin 2.0 仿真（20 个简单模式双臂任务，每任务 100 次）：** RoboRouter 平均成功率 **79.85%**（Table 1；Table 5 消融主行与正文记为 79.9% / 79.90%），高于所有单体基线（DP3 76.45%、$\pi_0$ 69.90%、RDT 60.35%、ACT 55.90%、Code as Policies 54.45%、DP 51.05%）；路由准确率 **96.32%**，路由时延 **4.79 s**，平均每回合耗时 **38 s**。
  - **真机实验（UR5e 6-DoF + 步进电机二指夹爪 + RealSense D435i，5 个任务各 20 次）：** RoboRouter 平均成功率 **47%**，较最高单体基线 $\pi_0$（34%）提升 13 个百分点（RDT 33%、ACT 26%、DP 18%）；路由准确率 **98.63%**，路由时延 **4.54 s**，平均每回合耗时 **26 s**。
  - **关键消融（Table 5–6）：** 去掉 Evaluator 降幅最大（仿真 79.90% → 69.80%，真机 47% → 37%），说明基于关键帧+规则工具的细粒度归因远胜单纯统计成功率；去掉多模态检索退化为纯文本 LightRAG 降至 78.40% / 43%；去掉 Python 工具或 VQA 分别降至 77.70% / 41% 与 76.35% / 42%。
- 对 wiki 的映射：[RoboRouter: Training-Free Policy Routing for Robotic Manipulation](../../wiki/entities/paper-agentic-2603-07892.md)
