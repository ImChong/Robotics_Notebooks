# Compiling and Benchmarking Task-State Horizons for Embodied Agents

- **作者：** Meiqi Wang, Shichao Li
- **机构：** Tsinghua University, Beijing University of Posts and Telecommunications
- **版本 / 载体：** arXiv:2608.08036v1 [cs.RO]（2026-08-08 提交，32 页含附录，HTML 全文）
- **原文：** [arXiv HTML 全文](https://arxiv.org/html/2608.08036)、[arXiv 摘要页](https://arxiv.org/abs/2608.08036)
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，arXiv:2608.08036v1 正文与附录未提供独立项目页或官方源码仓库链接）
- **核对范围：** 已通读 `paper_text.txt` 全文（3,183 行），包括 Abstract、Introduction、Related Work、Problem Formulation（Definition 1–2，Eq. 1–6）、RoboGraph（Eq. 7–10）、RoboGraphBench（Table 1–2）、Experiments（RQ1–RQ3，15 个模型评测与视觉后端扩展）、Conclusion、References 及 Appendix A–E（Table 3–9，Figure 10–13 交互载荷与 RoboTwin 2.0 轨迹日志）
- **通读：** 已完成

## 原文核对笔记

1. **核心问题与任务状态视界（TSH）**：现有长程具身基准多以动作序列长度或子目标数量衡量难度，但相同动作序列在存在遮挡、潜藏对象或执行期环境变化时，对世界状态跟踪的要求截然不同。论文在部分可观测决策过程下形式化定义 **Task-State Horizon (TSH)** 为三元组 $H = (H_{\mathrm{maintain}}, H_{\mathrm{explore}}, H_{\mathrm{update}})$，并在最短符号 Oracle 解 $\tau^*$ 上计算，分别刻画状态保持跨度、视图展开探索步数与外生事件导致的失效事实数量。
2. **RoboGraph 编译器**：将任务族规约与初始潜藏语义场景图 $G_0$ 编译为可执行实例 $\mathcal{I} = (G_0, g, \mathcal{A}, O, \Lambda)$。观测算子 $O(G_t, h_t)$ 根据当前房间边界与阻塞边 $E_t^{\mathrm{block}}$ 计算可见子图（离开视野的事实不自动保留），事件调度 $\Lambda = \{\langle \psi_\lambda, \delta_\lambda, \kappa_\lambda \rangle\}$ 基于语义进度触发图重写（非固定时间步），支持符号后端与仿真后端（RoboTwin 2.0、RoboCasa）。
3. **RoboGraphBench 规模与干预设计**：包含 7 个任务族（Table 3 统计为 57 个桌面场景与 27 个室内场景，共 84 个源场景），每个源场景衍生 1 个基线与 6 个互斥干预条件（$I_1$ 动作失败、$I_2$ 状态回退、$I_3$ 已完成子目标回滚、$I_4$ 错误重定位、$I_5$ 新增遮挡、$I_6$ 新增对象子目标），总计 588 个回合（399 桌面 + 189 室内）。图谱平均含 $18.4 \pm 3.3$ 个节点、$21.8 \pm 6.1$ 条边、$6.1 \pm 1.8$ 个初始遮挡节点与 $14.3 \pm 4.1$ 个子目标；Oracle 动作长度均值 $21.7 \pm 5.4$，有序动作距离均值 $24.3 \pm 5.6$。
4. **评测协议与指标**：统一闭环 Harness（`recent_history` 保留最近 16 步，步数上限 $T_{\max}=100$，严格 JSON 输出含 `reason`、`recovery`、`action`）。除 SR 与 SSAL 外，提出 5 个状态管理指标（AER、RAR、SEG、ADR、IRR）及 2 个失败恢复指标（F1@RD、Acc@RG）。
5. **主要发现（8,820 条轨迹，15 个模型）**：8 个闭源模型与 7 个开源模型差距显著；GPT-5.5 在语义闭环中桌面 SR 达 78.4%、室内 SR 达 82.5%，Claude Opus 4.7 室内 SR 达 80.4%，Gemini 3.1 Pro 桌面 SR 为 62.7%；开源模型中 Qwen3-VL-32B 桌面 SR 仅 9.3%，Cosmos-Reason1-7B 与 RoboBrain2.0-7B 为 0.0%。新增遮挡（$I_5$）导致最大 SR 下降（桌面 -10.4 pp、室内 -4.4 pp），动作失败（$I_1$）导致最长恢复延迟（桌面 35.8 步、室内 37.1 步）。在 RoboTwin 2.0 三视角 RGB 视觉后端（desk organization 与 cosmetics storage，8 个闭源模型）中，ADR 与语义后端呈显著正相关（$\rho = 0.75$），但视觉接地使绝对成功率大幅降低。

对 wiki 的映射：[Compiling and Benchmarking Task-State Horizons for Embodied Agents](../../wiki/entities/paper-agentic-2608-08036.md)
