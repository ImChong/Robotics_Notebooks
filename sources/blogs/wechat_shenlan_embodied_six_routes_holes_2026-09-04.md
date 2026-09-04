# 拆完这六条技术路线，总算知道具身智能的“窟窿”都分别在哪了！

> 来源归档（blog / 微信公众号）

- **标题：** 拆完这六条技术路线，总算知道具身智能的“窟窿”都分别在哪了！
- **类型：** blog
- **作者：** 深蓝具身智能（编辑｜阿豹；审编｜具身君）
- **原始链接：** https://mp.weixin.qq.com/s/k7CR03ZHaSQRMVvutpSnCg
- **发表日期：** 2026-09-04（create_time 1788491496 UTC）
- **入库日期：** 2026-09-04
- **抓取方式：** 移动端 UA 直拉 `mp.weixin.qq.com` HTML 后去标签（Jina Reader 返回验证页）
- **原始抓取落盘：** [`sources/raw/wechat_shenlan_embodied_six_routes_holes_2026-09-04.md`](../raw/wechat_shenlan_embodied_six_routes_holes_2026-09-04.md)
- **一句话说明：** 把产业叙事里并置的六条路线（模块化 / LLM+技能 / IL / RL / VLA / 世界模型）拆成「现状—卡点—趋势」；主张 2026 前沿重新分层，而不是端到端消灭模块。文内案例**映射到已有 wiki 节点**，不重复造页。

## 核心摘录（归纳，非全文）

作者用 Figure Helix 02 已能收拾房间、却仍发 **Index（截至 2026-08-25 自称 1600 万条人视频）** 并签约 Nscale（文称最高 10 万张 GPU、意向超 400 亿元人民币）开场：能力演示与数据/算力饥渴并存，说明卡点已从「完全不会」换成更深的窟窿。

文内声明：六条**不在同一分类轴**——模块化是工程框架，IL/RL 是学习范式，LLM+技能 / VLA / WM 是模型家族；并置只因行业把它们当主流流派，未来取决于如何缝合。

| 路线 | 文内卡点（一句话） | 文内趋势 | 本库节点（复用，不新建） |
|------|-------------------|----------|--------------------------|
| 模块化（感知→规划→控制） | 开放世界规则写不完；TAMP 组合爆炸 | 学习替换最难手写部分（可行性、affordance） | [MPC](../../wiki/methods/model-predictive-control.md)、[WBC](../../wiki/concepts/whole-body-control.md)、[ScheduleStream / TAMP](../../wiki/entities/schedulestream.md) |
| LLM+技能 | 计划聪明 ≠ 底层技能可靠；失败难恢复 | 高层变成编排层：ER → VLA / Planner / 工具 / 他机 | [Gemini Robotics](../../wiki/entities/gemini-robotics.md)、[LLM 控制接口](../../wiki/concepts/llm-robotics-control-interfaces.md) |
| 模仿学习 | 分布偏移 + 扩散推理延迟；长 chunk 钝于变化 | ACT / DP 被吸进 VLA 当动作解码底座 | [Action chunking](../../wiki/methods/action-chunking.md)、[Diffusion Policy](../../wiki/methods/diffusion-policy.md) |
| 强化学习 | 真机试错贵；仿真 ≠ 现实 | 从零练会 → 基础模型后训练（文举 RL Token） | [RL](../../wiki/methods/reinforcement-learning.md)、[Sim2Real](../../wiki/concepts/sim2real.md)、[SmoothRL](../../wiki/entities/paper-smoothrl.md) |
| VLA | 数据、实时、记忆、最后一毫米 | 单体大模型 → 记忆 + 触觉 + RL + 高速层的复合系统 | [VLA](../../wiki/methods/vla.md)、[Figure AI](../../wiki/entities/figure-ai.md)、[OpenVLA](../../wiki/entities/paper-openvla.md) |
| 世界模型 | 画面合理 ≠ 动作对齐的物理正确 | 评测从「像不像」转向「能否改善决策」 | [生成式世界模型](../../wiki/methods/generative-world-models.md)、[WAM](../../wiki/concepts/world-action-models.md) |

收束：Helix 02 的 System 2 / 1（200 Hz）/ 0（1 kHz）、Gemini Robotics ER 2 的脑–手分离、π 系外挂 memory / WM subgoal / 在线 RL，都指向**按时间尺度划分的学习模块**，而不是「端到端还是模块化」二选一。

数字与合同金额均为公众号转述，**以 Figure / Google / PI 官方为准**；本库不把 Nscale 意向金额写成已核实事实。

## 对 wiki 的映射

- 写回：[六条路线的窟窿](../../wiki/queries/embodied-six-routes-holes.md)（**非**论文详情替代）
- 交叉：[五大范式](../../wiki/comparisons/robot-learning-five-paradigms-taxonomy.md)、[五大模型族选型闭环](../../wiki/queries/embodied-fm-taxonomy-loop.md)、[SmoothRL](../../wiki/entities/paper-smoothrl.md)
- **本次未**为文内 Helix / Index / RL Token / Cosmos 3 新建实体：已有页只回链；缺页标「待升格」。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 六条路线对照既有 wiki（复用 / 待升格，**0 重复造页**）
