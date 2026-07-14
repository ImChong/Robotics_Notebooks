# 人形机器人运动控制 Know-How（飞书公开文档抓取摘录）

- **原始 URL：** <https://roboparty.feishu.cn/wiki/GvUxwKVeNiGa7kku6vEcvqfKn87>
- **抓取日期：** 2026-07-14
- **抓取工具：** Agent Reach v1.5.0 → Jina Reader（`curl https://r.jina.ai/<URL>`）
- **说明：** 飞书长文仅部分正文可通过 Jina 公开读取；完整目录树见 [sources/notes/know-how.md](../notes/know-how.md) 与 [resources/knowhow/人形机器人运动控制Know-How.md](../../resources/knowhow/人形机器人运动控制Know-How.md)。

## 文档定位（作者自述摘要）

RoboParty 飞书公开文档《人形机器人运动控制 Know-How》是一份**课程地图 / 方法总纲**，从传统运动控制（MPC）到运控大模型（BFM）梳理小脑 Know-How。每个方法按 **「方法原理 → 基本代码 → 方法局限性」** 呈现；示例代码偏伪代码，完整复现见 [RoboParty GitHub](https://github.com/Roboparty)。

## 宏观判断（正文可见段落）

- 具身智能是**硬件、数据、算法范式**三类问题的叠加（引用 CMU 石冠亚观点）。
- 2025 末 BFM 范式带来 data scale 可能，但数据质量/采集/筛选与 benchmark 仍无统一标准。
- RL 是腿足成功的关键起点，但 Physical AI 可能需要**不同于 LLM 的算法范式**（如监督 vs 无监督 BFM 路线分歧）。
- 文档同时给出 **Model-based** 与 **Learning-based** 两条学习路线，强调从项目实践入门而非纯理论堆叠。

## 传统运动控制学习路线（摘录要点）

1. 机器人学入门：《机器人建模和控制》、ETH《Robot Dynamics Lecture Notes》；正逆运动学/动力学、浮动基模型。
2. 人形建模：URDF → Pinocchio 正逆运动学/动力学示例。
3. 仿真：MuJoCo + Pinocchio 运动学/动力学分析。
4. 优化：LQR vs QP、MPC 特点、OSQP/qpOASES 等求解器；手写 QP 对比 Python/C++。
5. WBC：全身动力学 QP，MuJoCo 中实现站立/踏步/蹲起。
6. 四足开源代码（如 cheetah software）理解 MPC+WBC 工程实现。

## 完整主题树（与 wiki 节点映射）

见 [wiki/overview/humanoid-motion-control-know-how-technology-map.md](../../wiki/overview/humanoid-motion-control-know-how-technology-map.md)。

## 对 wiki 的映射

- 技术地图父节点：[humanoid-motion-control-know-how-technology-map.md](../../wiki/overview/humanoid-motion-control-know-how-technology-map.md)
- 结构化摘要（2026-04 query）：[humanoid-motion-control-know-how.md](../../wiki/queries/humanoid-motion-control-know-how.md)
- 源归档：[humanoid_motion_control_know_how.md](../papers/humanoid_motion_control_know_how.md)
