# Karpo Decision：Ten Appetites, One Menu 官方交互体验归档

- **标题：** Ten Appetites, One Menu: a Karpo Decision showcase
- **类型：** site / 官方在线决策体验
- **发布方：** MachinePulse Pte. Ltd.
- **体验页：** https://karpo.ai/models/decision/ten-appetites
- **模型：** https://huggingface.co/MachinePulseAI/karpo-decision-31B
- **核查版本：** 2026-11-21 可见网页
- **核对范围：** 通读官方页面 Meet the guests、Start together、Serve all ten、Handle a sold-out dish、How a round is run、计时与约束说明；未实际进入交互应用逐轮运行，未复测响应延迟。
- **通读：** 已完成
- **开放状态：** 官方交互页面公开可访问；体验页并非训练代码或复现数据集。
- **沉淀到 wiki：** [Karpo Decision 家族](../../wiki/entities/karpo-decision.md)

## 官方演示机制

1. 同样的 **10 位虚构顾客** 分别出现在两个餐厅房间，各有预算、过敏和口味偏好；菜单有 **8 道菜**。
2. 页面发给 Karpo Decision 的是**单个状态（完整菜单 + 顾客）与每位顾客一个 choice 问题**；所有菜品仍列在候选内，包括售罄或不合适的选项。
3. 模型对每个顾客的每个候选返回概率，展示分配结果；错误选择也保留并标记。先检查**硬约束**，再检查偏好覆盖。
4. 第二轮允许标记某道菜售罄，只需要重新决定受影响顾客的菜，其他订单保持不变。
5. 两个计时器都在浏览器测量；Karpo 从启动至整个响应返回、**含网络耗时**；随后的上菜动画**不计入**。不能把网页交互计时直接当作 GPU 模型纯前向延迟。

## 范围与局限

- 顾客、菜单、营养资料均为虚构；展示的是**有限候选的多问题批量决策**，不是机器人连续控制、真实餐厅安全推荐或真实用户偏好学习。
- 页面提及约束与偏好分别计分，但未公开该演示所用训练数据/额外运行代码；因此不推断底层推理服务或客户数据。
- 本归档只覆盖网页可读说明，不声称人工互动通过。

