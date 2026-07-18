# 机器人控制算法八大体系详解：从 PID 到强化学习

> 来源归档（blog / 微信公众号）

- **标题：** 机器人控制算法八大体系详解：从 PID 到强化学习
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/Kp12BMBiC7YiIiDPi_P8-g
- **发表日期：** 2026-07-18
- **入库日期：** 2026-07-18
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`）；正文约 1.58 万字 / 30 图；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **原始落盘：** [wechat_shenlan_robot_control_eight_paradigms_2026-07-18.md](../raw/wechat_shenlan_robot_control_eight_paradigms_2026-07-18.md)
- **一句话说明：** 从分层闭环架构出发，将控制算法层划分为经典线性反馈、非线性动力学、鲁棒、自适应、位置/力混合、滚动优化/ILC、机器学习驱动与强化学习八类，并逐类给出代表算法原理、术语与工程融合示例。

## 核心摘录（归纳，非全文）

### 分层架构

- 任务规划层 → **控制算法层** → 伺服执行层。
- 控制算法层按建模方式、抗扰能力、数据依赖度分为 **八大类**；前四类为显式建模，后四类分别面向接触、约束、数据补偿与自主习得。

### 八大体系与代表算法

| 体系 | 代表算法 |
|------|----------|
| ① 经典线性反馈 | PID、LQR、极点配置 |
| ② 非线性动力学 | CTC、IDC、反馈线性化 |
| ③ 鲁棒控制 | SMC、H∞、μ 综合 |
| ④ 自适应控制 | MRAC、A-CTC、RLS |
| ⑤ 位置/力混合 | 阻抗、导纳、力位混合、直接力反馈 |
| ⑥ 滚动优化与 ILC | MPC、ILC |
| ⑦ 机器学习驱动 | NN 补偿、GP、模糊逻辑、聚类故障补偿 |
| ⑧ 强化学习智能 | 值函数 RL、策略梯度、MBRL、HRL、鲁棒/自适应 RL、模仿学习 |

### 融合示例（文内）

- 工业臂：CTC + SMC + ILC。
- 人形：MPC + RL；底层仍依赖 PID 电流环与动力学补偿。

## 对 wiki 的映射

| 主题 | 关系 |
|------|------|
| [八大机器人控制体系分类](../../wiki/comparisons/robot-control-eight-paradigms-taxonomy.md) | **主沉淀页**：体系总览、演进关系、融合架构 |
| [经典线性反馈（体系①）](../../wiki/overview/robot-control-paradigm-classical-linear-feedback.md) | PID / LQR / 极点配置 |
| [非线性动力学控制（体系②）](../../wiki/overview/robot-control-paradigm-model-based-nonlinear-dynamics.md) | CTC / IDC / 反馈线性化 |
| [鲁棒控制（体系③）](../../wiki/overview/robot-control-paradigm-robust-control.md) | SMC / H∞ / μ 综合 |
| [自适应控制（体系④）](../../wiki/overview/robot-control-paradigm-adaptive-control.md) | MRAC / A-CTC / RLS |
| [位置/力混合（体系⑤）](../../wiki/overview/robot-control-paradigm-hybrid-position-force.md) | 阻抗 / 导纳 / 力位混合 / 直接力反馈 |
| [滚动优化与 ILC（体系⑥）](../../wiki/overview/robot-control-paradigm-receding-horizon-ilc.md) | MPC / ILC |
| [机器学习驱动（体系⑦）](../../wiki/overview/robot-control-paradigm-ml-driven-control.md) | NN / GP / 模糊 / 聚类补偿 |
| [强化学习智能（体系⑧）](../../wiki/overview/robot-control-paradigm-rl-intelligent-control.md) | 值函数 / 策略梯度 / MBRL / HRL / IL |
