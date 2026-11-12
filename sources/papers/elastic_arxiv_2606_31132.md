# ELASTIC（arXiv:2606.31132）

> 来源归档（paper）

- **标题：** ELASTIC: Efficiently Learning to Adaptively Scale Test-Time Compute for Generative Control Policies
- **类型：** paper
- **作者：** Andrew Zou Li, Gokul Swamy, Yonatan Bisk, Andrea Bajcsy（Carnegie Mellon University）
- **arXiv：** <https://arxiv.org/abs/2606.31132>（v1，2026-06-30，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2606.31132>
- **代码：** 论文正文未给出代码仓库链接
- **入库日期：** 2026-11-12
- **一句话说明：** 把生成式控制策略的测试时算力分配建成元 MDP：用 SAC 训练元策略，在冻结的基策略去噪循环里逐状态决定各样本的去噪步长（顺序轴）与是否剪枝（并行轴），奖励 = 验证器价值 - 算力惩罚。
- **核对范围：** 通读 arXiv v1 PDF 全文：摘要、第 1-6 节（引言、相关工作、预备知识、方法 4.1-4.3、实验 5.1-5.3、局限）与附录 7.1-7.6（算法、补充结果、元策略训练细节、延迟测量、验证器训练、超参）；参考文献只扫读未逐条核对；图 2 / 图 5 看了渲染图（分辨率低，散点数值只能估读），其余图以图注与正文为准。未读：无补充材料，未找到项目页与代码，故没有核对任何代码实现。
- **通读：** 已完成

## 核心摘录

1. **问题（§1、§3）：** 生成式控制策略（GCP，扩散策略与基于流的 VLA）有两条测试时算力轴：顺序（更多去噪步，步长 Δτ 越小 L = ⌈T/Δτ⌉ 越大）与并行（采 N 个样本 + 验证器 Best-of-N）；最优分配依赖任务、状态与策略。
2. **方法（§4.1）：** 元 MDP M_meta = (S, A, T, r)。状态 s_k = (物理状态 x, N 个部分去噪动作 u_k, 各自去噪时间 τ_k, 活跃掩码 m_k)；动作 a = 各样本本步前进的 Δτ，Δτ = 0 表示永久停止（剪枝）该样本；全部样本去噪完或被剪后，由验证器 Q_ϕ 选最高分动作执行。
3. **奖励（§4.2，式 1）：** r(ξ) = Q_task(x,u) - α·L(ξ) - β·(P(ξ) - 1)；L 为最慢样本的去噪长度，P 为时间平均活跃样本数；用环境级 Q 函数作去噪轨迹末端价值以缓解信用分配。
4. **训练（§4.3、附录 7.3、7.6）：** SAC，注意力处理样本集（置换等变的 actor、置换不变的集中式 critic），对基策略观测嵌入做 FiLM；两阶段混合 RL（离线 critic 预训练 5000 步，再在线）；每 4 个环境步用反事实子采样扩充回放。
5. **扩散策略仿真（§5.1、图 2）：** PushT、PushT Obstacle、Robomimic Square PH/MH、Can Paired、Can Reverse，每任务 200 个固定初始条件；三组 (α,β)：(0.03,0.1)、(0.1,0.03)、(0.1,0.1)；与算力匹配的 Fixed、Sequential 对比，ELASTIC 在匹配算力下成功率更高，增益在 Can Paired 与 Square MH 最大。文中以散点图呈现，正文未给成功率表。
6. **π0.5 VLA（§5.3）：** LIBERO-10 用 π0.5-LIBERO 检查点 zero-shot，每任务 20 次；真机为 Franka + π0.5-DROID，单一「把紫色杯子放进篮子」任务，20 次，基策略成功率 45%；验证器为微调的 V-GPS。真机上 ELASTIC 与 BoN（N=10）的成功率相当，推理延迟低 34%（图 5）。
7. **LIBERO-10 结果（§5.3、图 11）：** 基策略已很强，各方法成功率相近；BoN 延迟 +36.8%，Sequential（18 步）+40.3%；ELASTIC 平均 (2.6 步, 6.9 样本)，成功率增益与 Sequential 相当，延迟比基策略低 6.1%。
8. **作者自述局限（§6）：** 学得的验证器 Q_ϕ 是主要瓶颈；元策略按任务训练，对多任务 VLA 代价高；跨环境泛化留作未来工作。

## 对 wiki 的映射

- [paper-elastic-test-time-compute-gcp](../../wiki/entities/paper-elastic-test-time-compute-gcp.md)
