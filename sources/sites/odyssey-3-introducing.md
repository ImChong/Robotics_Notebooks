# Introducing Odyssey-3: A General-Purpose Physical Intelligence

> 来源归档（ingest · 官方首发技术说明；非论文）

- **类型：** 官方研究发布 / project announcement
- **URL：** <https://odyssey.systems/introducing-odyssey-3>
- **发布时间：** 2026-09-15
- **作者：** Oliver Cameron、Jeff Hawke
- **核查日期：** 2026-11-08
- **核对范围：** 已通读官方文章从「Physical Agents That Speak the Language of the World」至「An Early Glimpse of Physical Agents」的全文，包括机械臂、Flexion 人形、印度道路驾驶、agent 训练、无人机、游戏迁移案例及末尾局限性叙述；视频展示仅核对页面的文字说明，未据视频推断额外成功率。
- **通读：** 已完成
- **代码：** 页面未公布 Odyssey-3 训练/推理源码和权重下载；非开源复现入口
- **数据：** 页面未提供完整训练数据集下载入口
- **对应节点：** [Odyssey-3](../../wiki/entities/odyssey-3.md)

## 机制与适配边界

- **统一主干：** Odyssey-3 是使用大规模视觉观测预训练的自回归扩散 Transformer 世界模型；用每种目标机器的成对观察—动作数据训练专属 action decoder 或 policy，而不是把视频世界模型当作无需适配的底层电机控制器。
- **机械臂：** 官方声称使用数十小时示范，在误抓或物体掉落时出现未包含于示范中的恢复行为；没有提供跨平台完整成功率表。
- **人形机器人：** Flexion 基于 Odyssey-3 表示及自有机器人学习和控制工程，用数十小时遥操作数据训练实时任务策略；官网称在光照变化条件下优于其测试的 VLA 基线，未披露具体基线名称与量化成功率。
- **自动驾驶：** 官方称冻结预训练 backbone，采用 20 小时**模拟**驾驶数据训练小型 waypoint policy，在印度真实道路闭环运行；纯模拟训练策略在两次安全员接管之间的行驶距离约为以真实数据训练策略的 **77%**。这一比例不是无接管成功率，更不是「零安全员」。
- **无人机：** 结合近期图像、无人机运动状态及高层指令，使用数十小时模拟飞行数据训练 waypoint expert，演示发生于**室内仿真**环境。
- **游戏：** 在主干冻结基础上使用游戏画面和键鼠输入训练控制；官方举例约 2 小时 GTA V 轨迹训练的移动策略零额外训练迁移到 Red Dead Redemption 2 的骑马行为。仅为展示案例，不是系统性成功率证明。
- **Agent 环境：** 模型可生成交互世界供 agent 试错；文章关联 PROWL 研究。该部分是训练愿景与演示，不等同于已验证可替代真实物理引擎。

## 原文中的不确定性

1. 机械臂/人形示范覆盖的任务与扰动有限；文章没有公开可复现的全面测试协议及逐任务数据。
2. 车辆、无人机、游戏采用不同的策略与执行器；「单个基础模型」表示共享预训练主干，**不是同一个动作头直接驱动全部平台**。
3. 首发文章提到未来公开发布，须与 2026-10-08 的研究预览发布状态区分；不能推断模型权重或完整训练代码已经开放。

## 一手入口

- [2026-09-15 官方首发全文](https://odyssey.systems/introducing-odyssey-3)
- [2026-10-08 官方后续发布全文](https://odyssey.systems/meet-odyssey-3)
- [官方 Odyssey-3 在线体验](https://experience.odyssey.systems/)
