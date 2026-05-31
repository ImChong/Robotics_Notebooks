# τ₀-WM 项目页（AGIBOT Finch）

> 来源归档

- **标题：** τ₀-World Model: A Unified Video-Action World Model for Robotic Manipulation
- **类型：** site
- **URL：** <https://finch.agibot.com/research/tau0-wm>
- **组织：** AGIBOT Research / Finch
- **入库日期：** 2026-05-31
- **一句话说明：** 官方研究页：5B 统一视频–动作世界模型，异构约 2.73 万小时数据、共享 VAM 表征、动作条件仿真器与测试时 propose–evaluate–revise 闭环；附技术报告 PDF 与 Hugging Face 权重入口。

## 页面要点（策展）

1. **问题动机：** 操纵需要同时 **预测动作后果** 与 **输出可执行控制**；机器人示教有动作但场景窄，自我中心人视频有交互动力学但无机器人动作空间。
2. **统一 formulation：** 异构数据源只监督其 **合法信号**（视频-only 样本不监督动作；机器人样本同时监督视频与动作；失败/rollout 可监督任务进度）。
3. **VAM 核心：** 共享 **视频扩散骨干**（多视角观测 + 语言 + 机器人状态）联合预测 **未来视觉 latent** 与 **连续 action chunk**；动作支路经 **逐层 cross-attention** 接入视频中间表征。
4. **双接口：** **策略**（采样 action chunk）+ **动作条件视频仿真器**（给定候选动作预测多视角未来与 **稠密任务进度轨迹**）。
5. **测试时计算：** 多候选 action → **Re-denoising Consistency Score** 排序；不佳时再 **仿真未来 → 选优 rollout → 二次动作预测**（proposal–evaluation–revision）。
6. **数据规模（页面口径）：** 约 **27,300 小时** — 真机遥操作 **17,800h**（双臂多视角）、UMI **6,500h**、自我中心人交互 **3,000h**（双臂多视角）。
7. **真机结果：** 预训练未见的四类任务上 **平均成功率最佳**，精细对齐任务（如 Faucet）上相对其他方法更稳。

## 对 wiki 的映射

- [τ₀-World Model（τ0-WM）](../../wiki/entities/tau0-world-model.md) — 方法级归纳
- [sources/papers/tau0_wm_tech_report.md](../papers/tau0_wm_tech_report.md) — 技术报告 PDF
- [sources/repos/sii_research_tau_0_wm.md](../repos/sii_research_tau_0_wm.md) — 代码与部署说明
