# Learning and Transferring Closed-Loop Robot Software

- **作者：** So Kuroki, Kento Kawaharazuka, Yujin Tang
- **机构：** Sakana AI；The University of Tokyo
- **版本 / 载体：** arXiv:2609.19906v3 [cs.RO]（2026-09-17 提交 v1，2026-10-30 修订 v3；arXiv HTML 全文）
- **原文：** https://arxiv.org/html/2609.19906 （摘要页：https://arxiv.org/abs/2609.19906 ）
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，arXiv v3 原文及当前检索材料未提供独立项目页或官方 GitHub 仓库链接）
- **核对范围：** 已通读 arXiv:2609.19906v3 HTML 全文（§I Introduction、§II Related Work、§III Method A–C、§IV Experiments A–D、Table I–III、§V Physical-Robot Validation、Table IV、§VI Limitations and Future Work、§VII Conclusion 与参考文献表）
- **通读：** 已完成

## 原文核对笔记

- **核心问题：** 闭环机器人策略代码包含观测处理、内部状态管理 $h_t$ 与条件分支逻辑。利用编码智能体在源任务上通过仿真执行反馈优化得到的完整闭环实现，究竟会因为过拟合源任务而阻碍迁移，还是能作为可复用执行经验促进新任务策略获取？
- **方法框架（§III）：**
  1. **策略接口（Eq. 1）：** $o_t = \mathcal{O}_\omega(x_t)$，$(a_t, h_{t+1}) = P(o_t, h_t; q)$，冻结后的策略在测试时直接执行，不再调用编码模型。
  2. **源任务学习与归档（§III-B，Eq. 2）：** 编码智能体 $\mathcal{A}_B$（实验用 `gpt-6-astra`，高推理强度）基于任务指令、成功演示 $\mathcal{D}_s$ 和接口说明 $\mathcal{I}_\omega$ 生成初始代码 $P_s^{(0)}$；在训练集上按轨迹观测/动作/成败/报错反馈迭代优化共 $K$ 代（候选按训练集成功率排序作为修订起点，平局优先早代），通过验证集聚合成功率选出 $P_s^\star$ 存入软件档案库 $\mathcal{M} = \{P_s^\star \mid s \in \mathcal{S}\}$。验证集仅向智能体暴露汇总成功次数与评测总数，不暴露单条轨迹或图像。
  3. **跨任务迁移（§III-C，Eq. 3）：** 对新任务 $q$，智能体获得 $\mathcal{M}$ 的只读访问权，结合目标指令、演示与接口生成初始策略 $\widetilde{P}_q^{(0)} = \mathcal{A}_B(q, \mathcal{D}_q, \mathcal{I}_\omega, \mathcal{M})$，再根据目标任务训练集执行反馈进一步修订得到 $\widetilde{P}_q$（无固定源-目标映射，由智能体自主选择参考实现与修改方式）。
- **主要仿真实验（RoboCasa / PandaOmron，§IV）：**
  - **Table I（4 个源任务：Faucet、Drawer、Pick → Sink、Microwave；每任务 10 演示/10 训练/10 验证/30 测试，5 代优化）：** 验证筛选后的策略代码平均成功率 **64.2%**（77/120），相对初始生成代码的 **28.3%**（34/120）提升 **35.8 个百分点**；也高于在线调用同模型的 `GPT-6 Astra online`（**51.7%**，且在线模式平均每回合需 **52.6 次** 模型调用）与同 10 条演示训练的 `BC-Transformer-10`（**9.2%**）。Panel B 列出 100 条演示/RGB+state 条件下的公开多任务策略作为参考对比（GR00T N1.5 **62.5%**、$\pi_{0.5}$ **58.3%**、$\pi_0$ **37.5%**、Diffusion Policy **20.0%**）。
  - **Table II（观测消融；每任务 5 训练/5 验证/10 测试，3 代优化）：** Full 配置平均 **65.0%**；仅 RGB 降至 **22.5%**（-42.5 pp）；单独移除 `GT-Geom` 降至 **35.0%**（-30.0 pp），降幅最大；单独移除 `Metric 3D` 为 **67.5%**（+2.5 pp）、移除 `Contact` 为 **70.0%**（+5.0 pp），但单轮运行不足以证明移除信息普遍有益。
  - **Table III（9 个目标任务跨任务迁移，Near 4 / Intermediate 2 / Far 3，3 次独立运行，每任务 5 演示 + 5 训练单轮改进 + 10 共享测试，单会话限额 30 分钟和 80 API 调用）：** 九任务宏平均成功率方面，`No reference` 为 **45.2 ± 10.3%**（122/270），`Initial4` 为 **41.5 ± 10.5%**（112/270），`Best4` 为 **57.0 ± 7.8%**（154/270）。`Best4` 在全部 3 次运行上均优于 `Initial4`，平均提升 **15.6 个百分点**。静态分析显示 4 个源策略 Python 非测试代码的 `if`/`elif` 语句总数由 Initial4 的 **622** 增至 Best4 的 **741**（+19.1%）。
- **真机验证（Unitree G1，§V，Table IV）：**
  - 29 自由度全身关节 + 固定五指手，20° 俯视相机，使用 **Claude Opus 5.5** 在 RoboCasa 中无演示训练 10 条案例（最多 8 代，10/10 早停），部署到 G1 的 **SONIC** 全身控制器之上（策略 50 Hz 输出行走与头/手目标位姿），感知端采用深度点云 + **JEV-1.14** 图像目标区域识别（不使用仿真真值物体位姿或接触标签），部署前微调目标距离等少数运行时参数。
  - 真机测试（未评测跨任务档案迁移，仅验证学习环节）：Close drawer **4/5**、Close refrigerator **3/5**、Navigation **4/5**。
- **对 wiki 的映射：** [Learning and Transferring Closed-Loop Robot Software](../../wiki/entities/paper-agentic-2609-19906.md)
