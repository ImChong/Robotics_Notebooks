# 机器人数据不是一个池子：示范、失败和接管分别教会模型什么

> 来源归档（blog / 微信公众号）

- **标题：** 机器人数据不是一个池子：示范、失败和接管分别教会模型什么
- **类型：** blog（监督信号 / 数据工程）
- **作者：** 具身智能前沿（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s?__biz=Mzg5OTY3ODkzNg==&mid=2247494519&idx=1&sn=7970dffc18701098a5fbd545d328649c
- **发表日期：** 2026-09-14
- **入库日期：** 2026-09-19
- **抓取方式：** wechat-article-for-ai（Camoufox）
- **原始抓取落盘：** [`sources/raw/wechat_jushen_qianyan_robot_data_supervision_types_2026-09-14.md`](../raw/wechat_jushen_qianyan_robot_data_supervision_types_2026-09-14.md)
- **一句话说明：** 示范/rollout+结果/接管纠正/Ego RGB/Ego+轨迹/偏好 各回答不同训练问题；混池会丢标签语义。

## 对 wiki 的映射

| 主题 | wiki |
|------|------|
| 概念页 | [robot-data-supervision-signal-types](../../wiki/concepts/robot-data-supervision-signal-types.md) |
| 飞轮闭环（#3） | [embodied-data-flywheel-minimal-closed-loop](../../wiki/concepts/embodied-data-flywheel-minimal-closed-loop.md) |
| BC / IL | [behavior-cloning](../../wiki/methods/behavior-cloning.md)、[imitation-learning](../../wiki/methods/imitation-learning.md) |
| EgoMimic / R3M | 论文链接见概念页 |

## 核心摘录（MVP）

- **分流表：** 成功示范→BC 动作；rollout+结果→价值/筛选；接管→恢复；Ego RGB→表征；Ego+3D 手→对齐后 IL。
- **示范边界：** 覆盖专家状态分布；部署偏离需结果与纠正信号。
- **Ego 分层：** RGB ≠ 示范；需轨迹+坐标+对齐工程才可动作监督。
- **RECAP 读法：** rollout 与纠正不同角色，非把失败当专家标签。
- **元数据：** 动作可用性、结果口径、介入、版本、训练去向五类字段。
