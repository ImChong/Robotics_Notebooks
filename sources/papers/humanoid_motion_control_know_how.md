# humanoid_motion_control_know_how

> 来源归档（ingest）

- **标题：** 人形机器人运动控制 Know-How（飞书公开文档）
- **类型：** course-map / long-form guide
- **来源：** 飞书 <https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87>
- **入库日期：** 2026-04-18
- **最后更新：** 2026-07-14（全文再 ingest）
- **一句话说明：** RoboParty 万字长文：趋势、双学习路线、建模+求解、Sim2Real、技术框架五条思考、Model-based 七段链与 Learning-based 全方法族；每法「原理 / 伪代码 / 局限性」。

## 全文归档

- **完整 Markdown（2260 行）：** [feishu_humanoid_motion_control_know_how_full_2026-07-14.md](../raw/feishu_humanoid_motion_control_know_how_full_2026-07-14.md)（用户导出全文，2026-07-14）
- **Jina 部分摘录：** [feishu_humanoid_motion_control_know_how_2026-07-14.md](../raw/feishu_humanoid_motion_control_know_how_2026-07-14.md)
- **目录树备份：** [know-how.md](../notes/know-how.md)

## 文档结构（一级章节）

| 章节 | 行号约 | Wiki 父节点 |
|------|--------|-------------|
| 运动控制发展趋势 | §1 | [humanoid-motion-control-trends](../../wiki/overview/humanoid-motion-control-trends.md) |
| 学习路线（传统 / RL） | §2 | [depth-classical-control](../../roadmap/depth-classical-control.md)、[depth-rl-locomotion](../../roadmap/depth-rl-locomotion.md) |
| 控制问题解决思路 | §3 | [modeling-and-solving](../../wiki/concepts/modeling-and-solving-for-control.md) 等 |
| 技术框架路线展望 | §4 | [framework-outlook](../../wiki/overview/humanoid-motion-control-framework-outlook.md) |
| 传统运动控制（Model Base） | §5 | [model-based-stack](../../wiki/overview/humanoid-model-based-control-stack.md) |
| 深度强化学习（Learning Base） | §6 | [rl-methods](../../wiki/overview/humanoid-rl-motion-control-methods.md) |

## Model-based 子章节（含伪代码块）

OCP → LIP+ZMP → SLIP+VMC → WBD+WBC/TSID（Pinocchio+CasADi 示例）→ SRBD+Convex MPC+WBC → CD+NMPC+WBC → 状态估计

## Learning-based 子章节（含伪代码块）

RL 理论 → Teacher-Student+DAgger → DreamWaQ（CENet，速度估计最关键）→ PIE（多头估计器）→ Attention 落足 → Retarget（PHC/GMR/OmniRetarget 对比）→ DeepMimic → AMP → BFM（TS 多动作 / SONIC / BFM-Zero FB）

## 作者核心判断（摘录）

- 具身智能 = 硬件 + 数据 + 算法范式；benchmark 仍缺统一标准。
- 动捕重映射仅**运动学可行**，需 RL/TO 补**动力学可行**。
- 人形 = 高维非线性 + 动力学突变 + 低静态稳定裕度 + 浮动基。
- Physical AI 需要更多 model 直觉；DreamWaQ/Attention 落足带有传统建模直觉。
- Retarget 质量链：**本体 > 重定向轨迹 > RL 算法**。

## 对 wiki 的映射

- **图谱索引：** [humanoid-motion-control-know-how-technology-map.md](../../wiki/overview/humanoid-motion-control-know-how-technology-map.md)
- **Query 摘要：** [humanoid-motion-control-know-how.md](../../wiki/queries/humanoid-motion-control-know-how.md)

## 当前提炼状态

- [x] 全文 2260 行归档（2026-07-14）
- [x] 全主题独立 wiki 节点
- [x] 技术框架路线展望独立 overview
- [x] 各方法页补充全文「局限性 / 工程判断」要点
