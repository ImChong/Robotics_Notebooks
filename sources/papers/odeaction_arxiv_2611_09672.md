# ODEAction — Continuous Action Evolution（论文来源归档）

- **英文：** ODEAction: From Predicting Chunks to Evolving Actions
- **中文：** 从预测动作块到连续演化动作
- **作者：** Qishen Wang 等
- **论文：** https://arxiv.org/abs/2611.09672
- **全文：** https://arxiv.org/html/2611.09672v1
- **版本：** arXiv v1，2026-11-06（技术报告，含附录）
- **代码：** 论文及 arXiv 页未提供可确认的官方 ODEAction 实现仓，不能将 StarVLA 或 PI0.5 基线仓冒充其源码
- **核对范围：** 阅读 HTML 方法 §2、实验 §3、结论与补充实现、评测协议、指标、统计分析、RoboTwin/真机逐任务结果；未重跑仿真或真机实验。
- **通读：** 已完成

## 核实机制

1. **条件 ODE：** 视觉、语言、机器人本体感知输入构成条件 c；学习 d a/dτ = Fθ(a,τ|c)，从当前动作初值积分以递推得到后续动作，而非分别拼接固定动作块。
   - 对 wiki 的映射：[ODEAction 实体](../../wiki/entities/paper-odeaction.md)「核心原理」。
2. **示范诱导矢量场：** 插值示范轨迹 γ(τ)，在轨迹邻域采样偏离动作；目标场为 γ点导数 + k(γ(τ)-a)，兼具前进和回归轨迹两个项。通过直接监督拟合 Fθ；推理可用 Euler 或更高阶积分。
   - 对 wiki 的映射：实体「流程总览」。
3. **EBench：** 同一基础策略对照，动作块 baseline **15.28%** → ODEAction **24.72%**，**+9.44 个百分点**，相对提升 **61.8%**；不是“成功率达到 61.8%”。
   - 对 wiki 的映射：实体「评测口径」。
4. **LIBERO 与 RoboTwin：** LIBERO 平均 96.85%→98.25%；RoboTwin 50 项 Clean 85.7%→89.4%、Random 87.3%→89.8%（StarVLA-OFT 配置）。真实机器人采用 AgileX Cobot Magic 1.0；不可跨配置直接相减。
   - 对 wiki 的映射：实体「实验与评测」。
5. **消融：** 无动作邻域采样成功率 47.4%、无修正项 56.2%，完整方法 84.4%（论文消融设置）；强调连续递归的偏差传播风险。
   - 对 wiki 的映射：实体「局限与风险」。
