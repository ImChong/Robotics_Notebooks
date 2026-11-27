# Find Something You Can't Do: Agentic Real-World Reinforcement Learning for Self-Improving VLA Models

- **作者：** Yuan Fang, Zechu Li, Haolei Tong, Puze Liu, Georgia Chalvatzaki
- **机构：** TU Darmstadt, Hessian.AI, University of Augsburg, Tongji University, Shanghai Research Institute for Intelligent Autonomous Systems
- **版本 / 载体：** arXiv:2609.32069v2 [cs.RO]（2026-10-01 修订版；v1 提交于 2026-09-25）
- **原文：** https://arxiv.org/abs/2609.32069 / https://arxiv.org/html/2609.32069
- **项目页 / 代码：** 未确认开源代码（论文声明项目页 `http://FIND.github.io`，实测返回非本项目占位页 `Yuguo's place`；论文仅在 §IV-A 引用第三方数据格式库 `https://github.com/huggingface/lerobot`，未公布 FIND 官方代码仓库）
- **核对范围：** 全文通读 arXiv HTML 版（Abstract、§I Introduction、§II Related Works、§III Methods 及公式 (1)–(18)、§IV Experiments 及 Table I–III、§V Conclusion / Limitations / Future Work、References [1]–[41]），并核对项目页与候选仓库上下文
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [Find Something You Can't Do (FIND)](../../wiki/entities/paper-agentic-2609-32069.md)
- **核心问题与思路（§I, §III-A）：** 现有 VLA（如 $\pi_{0.5}$）部署后通常保持参数冻结，无法从真机失败中持续改进；传统真机 RL 受限于“手动环境重置”与“人工/定制化成功奖励标注”。FIND 将持久工作空间中的自主练习重构为**场景条件、性能感知的任务选择问题**（公式 (1)–(6)）：上一轮执行后的物理场景 $z_{k+1}$ 直接作为下一轮起点，无需将物体复位到固定初始状态。
- **Agentic 闭环三大模块（§III-B）：**
  1. **场景条件任务筛选（Scene-conditioned task selection）：** VLM 根据当前观测推断任务相关状态 $z_k$，从预定义任务库 $\mathcal{T}$ 中过滤出当前可执行的子集 $\mathcal{F}(z_k)$。
  2. **性能感知课程（Performance-aware curriculum）：** 为每个任务 $\tau_i$ 维护最近 $K=20$ 次二值结果的滑动窗口 $\mathcal{H}_i$（以基础策略评估 rollout 初始化），计算近期成功率 $\hat{p}_i$ 与采样权重 $w_i=\max(\epsilon, 1-\hat{p}_i)$，仅在可行集 $\mathcal{F}(z_k)$ 内归一化采样（公式 (7)–(10)）。
  3. **成对图像自评估（Paired-image self-evaluation）：** 在执行前与执行后分别以同一场景视角采集观测，由 VLM 结合指令判定二值成败 $r_k \in \{0,1\}$（公式 (11)），同时作为 RL 稀疏奖励和课程历史更新信号。
- **残差离线策略 RL 与执行门控（§III-C）：**
  - 冻结 $\pi_{0.5}$ 提供基础动作 $\mathbf{a}^{\mathrm{base}}_t$，轻量残差策略 $\pi_\phi$ 叠加有界修正 $\mathbf{a}^{\mathrm{comb}}_t = \operatorname{clip}(\mathbf{a}^{\mathrm{base}}_t + \alpha\Delta\mathbf{a}_t)$（零初始化）。
  - 语言条件分支采用冻结 Sentence-BERT + 可训练投影 $z_l = g_{\theta_l}(f_{\mathrm{SBERT}}(l))$ 拼接到本体状态 $\tilde{s}_t=[s_t;z_l]$，投影层仅由 Critic 目标优化并在 Actor 更新时截断梯度。
  - 采用 10 个 Q 头的 REDQ 集成 Critic；执行门控计算残差优势 $A_{\mathrm{res}}(x_t)=\bar{Q}(x_t,\mathbf{a}^{\mathrm{comb}}_t)-\bar{Q}(x_t,\mathbf{a}^{\mathrm{base}}_t)$，仅当 $A_{\mathrm{res}}(x_t) > \delta$ 时输出 $\mathbf{a}^{\mathrm{comb}}_t$，否则回退到 $\mathbf{a}^{\mathrm{base}}_t$。
  - 工程上采用双线程异步采集与学习：有序 FIFO 队列传递转移数据，最近参数队列同步权重，采集端超前学习端最多 2000 个环境步即暂停等待。
- **实验验证与关键数值（§IV, Table I–III）：**
  - **平台与数据：** 7-DoF Franka Emika Panda + 平行夹爪，双 ZED X Mini（第三人称 + 腕部左目 RGB，缩放至 $224\times 224$），RTX 4090 + i9-14900K + 64 GB RAM；Meta Quest 遥操作采集并存为 LeRobot 格式。8 个任务构成 4 组可逆任务对（入碗/出碗、堆叠/拆叠、开抽屉/关抽屉、挂杯/取杯），每任务 50 条离线演示（Table I）。
  - **长程自治统计（Table II）：** 6 小时代表性运行完成 456 个回合，426 回合无需人工干预，30 次人工场景恢复（21 次物体掉落至不利姿态、7 次物体难以分离、2 次超出可达空间），人工干预率 6.58%，最长连续无干预运行 48 分钟，在线训练全程无人工奖励标注。
  - **VLM 评测（Table III）：** 在 100 场景任务筛选与 100 对图像成败评估上，GPT-5.6 Terra 达到 96% 任务筛选准确率（$4.1\pm 1.2$ s）与 100% 成败评估准确率（$2.6\pm 1.2$ s），优于同测的 GPT-5.6 Luna、Claude Fable 5、Gemini 3.1 Pro Preview 和 Kimi K3。
  - **整体提升与消融（Fig. 3）：** 5 个随机种子下，每 20k 步独立人工评估平均成功率（8 任务 $\times$ 20 回合）从微调基础 VLA 的 55% 提升至 71.9%；在线 VLM 滑动成功率达到 $(70.4\pm 4.6)\%$。移除性能感知课程或 Critic 门控均明显削弱自改进幅度。
- **边界与局限（§IV-B, §V）：** 依赖预定义任务库且假设基础 VLA 已具备可尝试的初始能力；任务设计依赖可逆或互斥兼容的任务对；人工干预主要源于基础 VLA 恢复覆盖不足的极端失败构型。
