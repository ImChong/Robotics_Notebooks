# VLA-Corrector: Stage-Aware Observable State Understanding for Prompt-Based Closed-Loop Recovery of Vision-Language-Action Policies

- **作者：** Chang Song, Bin Qian, Yan Feng, Zhijie Song
- **机构：** 原文 HTML v1（arXiv:2609.06508v1）未标注作者所属机构
- **版本 / 载体：** arXiv:2609.06508v1 [cs.RO]（2026-09-06 提交）
- **原文：** [arXiv HTML v1](https://arxiv.org/html/2609.06508) / [arXiv Abs](https://arxiv.org/abs/2609.06508)
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，arXiv:2609.06508v1 正文及附录链接未列出 VLA-Corrector 独立项目页、GitHub 代码库或 Learned Verifier 权重入口；基座策略使用 OpenPI 公开检查点 `gs://openpi-assets/checkpoints/pi05_libero`）
- **核对范围：** arXiv:2609.06508v1 全文（Abstract、Introduction、Related Work、Problem Formulation、Method、Experimental Setup、Results、Discussion、Conclusion，含 Eq. 1–7、Table 1–4 及 Fig. 1–5 文字说明）
- **通读：** 已完成
- **入库日期：** 2026-11-27

## 原文核对笔记

- **核心问题（Execution-Time Opacity）：** 动作分块（action chunking）型 VLA 策略在交互过程中易因动作噪声、目标物体位移、抓取偏差或目标区域偏移而偏离正常轨迹，但仅靠回合结束时的二值任务成败信号无法诊断并及时纠正中间阶段失败。论文提出在不修改冻结 VLA 参数、且部署期不使用特权仿真状态（privileged state）的前提下，通过可观测历史进行阶段感知失败验证并触发闭环提示恢复（Prompt Recovery）。
- **形式化定义与基座策略（Problem Formulation & Method）：**
  - 可观测历史 $h_t = \{(I_i^a, I_i^w, s_i, u_i)\}_{i=0}^t$，包含第三人称（agent-view）与腕部（wrist-view）双视角 RGB 图像、8 维本体感知状态 $s_t \in \mathbb{R}^8$ 及实际执行的 7 维动作 $u_t \in \mathbb{R}^7$（严格因果序）。
  - 基座策略为冻结的 OpenPI `pi05_libero`（`gs://openpi-assets/checkpoints/pi05_libero`），渲染 $256 \times 256$ 后缩放到 $224 \times 224$ 推理，输出 7 维动作块；常规 `replan_steps = 5`（上限 280 步），恢复阶段 `recovery_replan_steps = 3`（单次 Prompt Recovery 最多 100 个动作）。
- **双验证器设计与阶段语义：**
  - **Privileged Rule-Based Verifier（教师/上限对照）：** 跟踪状态机 $\texttt{APPROACH} \rightarrow \texttt{ALIGN} \rightarrow \texttt{GRASP} \rightarrow \texttt{MOVE} \rightarrow \texttt{PLACE} \rightarrow \texttt{SUCCESS}$，借助仿真器真实坐标、夹爪状态与 BDDL 谓词生成 `(stage, context)` 触发与离线监督标签（Eq. 7：距失败事件 $\le 10$ 步标为正样本 $f_t=1$，$>20$ 步标为负样本 $f_t=0$，中间过渡窗口 $[11, 20]$ 剔除）。
  - **Learned Verifier（纯可观测部署模块）：** 以 8 步历史窗口（每 2 步采样，$112 \times 112$ 图像）为输入；视觉分支为无参数空间统计特征提取器（每视角 $3 \times 4 \times 4$ 自适应平均池化拼接 RGB 均值与标准差，共 54 维，双视角拼接投影至 64 维），15 维 $[s_t; u_t]$ 投影至 32 维，经隐层 64 维的 GRU 融合后分别由二分类风险头 $p_t$ 和 5 阶段分类头（不含终态 `SUCCESS`）输出。
  - **在线触发门控与 Prompt Recovery：** 风险阈值 $\tau = 0.40$、连续 4 帧确认（$k=4$）、10 步冷却期，每 2 个环境步查询一次；阶段置信度低于 $0.9625$ 时回退至通用恢复提示 `UNKNOWN_FAILURE`。触发后丢弃待执行动作块前缀，构造 $q_t^{\mathrm{rec}} = q \oplus r_t$（保留原始指令 $q$ 并追加 `"Immediate recovery: "` 与阶段指令），由同一冻结 $\pi_{0.5}$ 生成恢复动作后再恢复原始 $q$。
- **关键实验数据（Table 1–4 & Real-World）：**
  - **训练配置（Table 2）：** 120 条 episode-disjoint 轨迹（60% 训练 / 20% 验证 / 20% 测试），30 epochs，batch size 32，Adam 类优化器学习率 $10^{-3}$、权重衰减 $10^{-4}$，CPU 训练；离线检测阈值 $0.03294$ 区别于在线门控阈值 $0.40$。
  - **仿真基准（Table 3 & 汇总）：** 标准 LIBERO 复合扰动下，No Assistance 为 $70.9\%$（Table 3 Combined 列印为 $70.7\%$）、Rule + Prompt 为 $82.6\%$（$+11.7\text{ pp}$）、Learned Binary + Prompt 为 $80.2\%$（$+9.3\text{ pp}$）；五种工况平均成功率分别为 $81.14\%$、$88.12\%$（$+6.98\text{ pp}$）、$86.88\%$（$+5.74\text{ pp}$）。LIBERO Plus 难度 4 上三者分别为 $73.7\% / 79.7\% / 79.0\%$，难度 5（Object & 10）为 $66.7\% / 73.6\% / 72.6\%$。
  - **检测与消融（Table 4）：** Temporal + Fusion 达 $93.09\%$ Accuracy、$61.14\%$ Precision、$94.85\%$ Recall、$74.35\%$ Positive F1、AUROC $0.902$、AUPRC $0.7123$，推理延迟仅 $0.457\text{ ms}$（p50）/ $0.589\text{ ms}$（p95）；在线门控触发精确率 $79.2\%$，恢复成功率 $75.0\%$（Rule + Prompt 为 $77.0\%$）。去掉时序（Single frame）使 Positive F1 降 $15.02\text{ pp}$，仅用 state-action 降 $15.80\text{ pp}$，仅用 agent-view 降 $10.79\text{ pp}$，仅用双视角视觉降 $6.32\text{ pp}$。
  - **真机零样本迁移：** 在 Pick and Open、Place、Pour Water 三项任务（每任务 40 次独立试验，执行中途人为移动目标物体）上，无真机微调或重训下平均成功率从冻结 `pi05_libero` 基线的 $82.5\%$ 提升至 $88.3\%$（$+5.8\text{ pp}$）。
- **核查边界：** 原文未公布作者单位，也未发布项目代码仓库与 Learned Verifier 训练数据集/权重；真机分任务具体百分比仅在 Fig. 5 柱状图中展示，HTML 文本未给出分任务数值明细。
- **对 wiki 的映射：** [VLA-Corrector: Stage-Aware Observable State Understanding for Prompt-Based Closed-Loop Recovery of Vision-Language-Action Policies](../../wiki/entities/paper-agentic-2609-06508.md)
