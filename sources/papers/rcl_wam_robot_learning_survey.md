# World-Action Models for Robot Learning and Control: A Survey（RCL / MBZUAI）

- **类型**：论文（survey）
- **收录日期**：2026-09-16（arXiv 落地更新 2026-09-25）
- **项目页**：<https://rcl-robotics.github.io/Awesome-World-Action-Models/>
- **配套策展**：[Awesome World-Action Models 仓库](../repos/awesome-world-action-models-rcl.md) · [静态站点](../sites/awesome-world-action-models-rcl.md)
- **arXiv**：[2609.16074](https://arxiv.org/abs/2609.16074)（2026-09-25 发布；此前站点标注 Coming soon）
- **中文导读**：[具身智能之心 WAM 训练策略（2026-09-25）](../blogs/wechat_embodied_heart_rcl_wam_survey_2026-09-25.md)
- **作者**：Zuxing Lu, Hongjia Zhai, Guanzhi Wang, et al.（MBZUAI 通讯：Xingxing Zuo）

## 一句话

面向机器人学习与控制的 **World-Action Models（WAM）** 综述：在部分可观测与物理约束下，把 **未来世界预测** 与 **可执行动作生成** 放在统一学习/推理过程中，并以 **2×2 架构 taxonomy**、训练数据金字塔、应用域与评测协议系统整理现有工作。

## 为什么值得保留

- 给出 **control utility** 导向的 WAM 判据：动作接地、时空一致、闭环改进、实时预算 — 避免只用视觉保真度评价。
- **架构与接口解耦**：One Model / Dual-system 与 Joint prediction / IDM 构成独立两轴，比单纯 Cascaded/Joint 更细。
- **机器人导向**：覆盖操纵、导航、自动驾驶；强调数据集、基准、指标与神经仿真闭环等工程议题。

## 核心摘录（面向 wiki 编译）

### 统一视图

\((\widehat{\mathbf{O}}, \widehat{\mathbf{A}}) = f_{\mathrm{WAM}}(h_t, \ell)\)

历史观测 \(h_t\) 与语言 \(\ell\) 条件下，联合或分步产生未来观测块 \(\mathbf{O}\) 与动作块 \(\mathbf{A}\)。

### 与 VLA / 世界模型 / MBRL 的边界

- **VLA**：\(p(a \mid o, l)\) — 反应式映射。
- **World model**：\(p(o' \mid o, a)\) — 预测演化，策略可外接。
- **Action-conditioned video**：动作条件视频生成，未必构成闭环策略。
- **WAM**：预测后果 **inform** 动作生成；与 reactive VLA、纯世界模型、经典 MBRL 分解均有明确对照节。

### 架构 taxonomy（2×2）

| | Joint prediction | IDM |
|--|------------------|-----|
| **One Model** | Q1：共享骨干联合出未来与动作 | Q2：共享骨干，先规划未来再 IDM |
| **Dual-system** | Q3：世界/动作专家分离，联合预测 | Q4：双专家 + plan-then-act |

注：Joint training  alone 不决定 One Model；部分 IDM 推理时不显式滚完整未来。

### 训练与数据

- **两阶段**：大规模 action-free 视频预训练 → 策略微调 / 数据增强 / RL 后训练。
- **数据金字塔**：互联网/自我中心视频提供世界先验；具身轨迹提供动作接地。
- **多模态传感**：RGB、深度、触觉等暴露互补动作相关状态变量。

### 应用与评测

- **应用域**：操纵、导航、自动驾驶 — 预测角色分表征学习、推理期 look-ahead、合成轨迹改进三类。
- **评测**：数据集、基准、指标与协议专节；强调世界侧与策略侧联合评价。

### 开放挑战

动作对齐、世界–动作因子分解、空间/多视角一致、长程记忆、神经仿真闭环策略学习、高效推理。

## 对 wiki 的映射

- 策展实体：[Awesome World-Action Models（RCL）](../../wiki/entities/awesome-world-action-models-rcl.md)
- 概念补强：[World Action Models（WAM）](../../wiki/concepts/world-action-models.md) — 补 RCL 2×2 taxonomy 与 control utility
- 对照综述：[world_action_models_survey_2605.md](world_action_models_survey_2605.md)（OpenMOSS · arXiv:2605.12090）
