# 人形机器人运动控制：强化学习与策略训练体系详解

> 来源归档（blog / 微信公众号）

- **标题：** 人形机器人运动控制：强化学习与策略训练体系详解
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/mxesB0pGI_NLSkSf-cZYug
- **发表日期：** 2026-08-08（frontmatter）
- **入库日期：** 2026-08-08
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`）；`--no-images`；正文约 1.1 万字 / 21 图；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **原始落盘：** [wechat_shenlan_humanoid_rl_policy_training_2026-08-08.md](../raw/wechat_shenlan_humanoid_rl_policy_training_2026-08-08.md)
- **关联姊妹篇：** [具身 RL 最小闭环](wechat_shenlan_rl_embodied_minimal_closed_loop.md)、[机器人学习五大体系](wechat_shenlan_robot_learning_five_paradigms.md)、[Sim2Real 非训后一步](wechat_shenlan_sim2real_sysid_to_adaptation.md)
- **一句话说明：** 把人形运动控制的数据驱动路径拆成 **五模块闭环**：RL/MDP 交互框架 → Actor-Critic → PPO 裁剪更新 → 多维奖励塑形 → Teacher-Student 蒸馏部署；并与传统 WBC/MPC 对照，收束到「底层传统稳、上层 RL 泛化」的混合架构。

## 核心摘录（归纳，非全文）

### 问题重框

- 人形运动 **没有闭式解析公式**：非线性、接触冲击使精细建模仍是近似；传统「建模 + 轨迹 + 调参」在非结构化场景有上限。
- 数据驱动路径适合求 **近似数值解**：用策略迭代逼近可用控制。

### 五模块体系

| 模块 | 角色 | 文内要点 |
|------|------|----------|
| **RL 基础框架** | 顶层交互载体 | MDP 五元组 $(S,A,P,R,\gamma)$；最大化折扣回报期望；定义状态/动作/奖励循环 |
| **Actor-Critic** | 决策–评估拆分 | Actor 出关节力矩/位置；Critic 拟合价值；优势函数 $A=Q-V$ 指导强化/抑制 |
| **PPO** | 稳定策略更新 | clip 目标限制新旧策略比；防「学新忘旧」；人形高维步态主流优化器 |
| **奖励函数** | 优化目标定义 | 任务 + 平衡 + 平滑 − 惩罚的加权和；无通用模板，机型/任务需独立调权 |
| **Teacher-Student** | 部署轻量化 | 大 teacher 仿真训完后蒸馏小 student；降低推理延迟与算力，衔接 Sim2Real |

### 数据流闭环（文内）

状态 → Actor 出动作 → 环境交互 → 奖励标量 → Critic 拟合价值/优势 → PPO 更新 Actor → 收敛后蒸馏 → 真机部署。

前端三件套（Actor-Critic + 奖励 + PPO）完成仿真内从零训练；师生蒸馏是 **后置独立模块**，不改训练逻辑，只做轻量化迁移。

### 与传统控制的关系

| 路线 | 优势 | 代价 |
|------|------|------|
| WBC / MPC | 稳定性、安全性、轨迹跟踪精度 | 精细建模成本高；非结构化泛化弱 |
| RL 策略体系 | 少依赖闭式模型；复杂场景自适应 | 奖励敏感；需仿真规模与蒸馏部署 |
| **混合（文内主流落地）** | 底层传统保安全，上层 RL 出自适应指令 | 接口与权限边界需工程设计 |

### 文内局限提示

- 奖励无通用模板，权偏会直接产出畸形步态。
- 蒸馏保留能力的前提是 teacher 已收敛；学生仍需任务奖励约束，避免只抄输出分布。
- AMP 等模仿学习在文中作为「底层常沿用 Actor-Critic / PPO」的旁注，非本篇主线。

## 对 wiki 的映射

| 主题 | 关系 |
|------|------|
| [人形 RL 策略训练五模块](../../wiki/overview/humanoid-rl-policy-training-five-modules.md) | **主沉淀页**：五模块闭环与混合控制读法 |
| [具身 RL 最小闭环](../../wiki/concepts/embodied-rl-minimal-closed-loop.md) | MDP/仿真循环入门姊妹 |
| [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md) | RL 方法总页 |
| [PPO](../../wiki/methods/ppo.md) | 裁剪更新算法细节 |
| [Privileged Training](../../wiki/concepts/privileged-training.md) | Teacher-Student / 特权蒸馏工程化 |
| [Humanoid RL Cookbook](../../wiki/queries/humanoid-rl-cookbook.md) | 从零训行走 checklist |
| [Sim2Real](../../wiki/concepts/sim2real.md) | 蒸馏后的真机迁移语境 |
| [人形 RL 身体系统栈](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md) | 论文能力层栈（与本页「训练模块栈」正交） |
| [WBC vs RL](../../wiki/comparisons/wbc-vs-rl.md) | 混合架构对照 |
