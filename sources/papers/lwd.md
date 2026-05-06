# lwd

> 来源归档（ingest）

- **标题：** Learning while Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies
- **类型：** paper
- **来源：** AGIBOT Research（[research/lwd 项目页](https://finch.agibot.com/research/lwd) / [PDF](https://finch-static.agibot.com/LWD/lwd-paper.pdf)）
- **入库日期：** 2026-05-06
- **最后更新：** 2026-05-06
- **一句话说明：** AGIBOT 提出 LWD —— 一个面向 VLA 通用策略的车队级 offline-to-online RL 后训练框架，把部署阶段的异构经验（成功/失败/人为干预）转化为持续改进信号。

## 核心论文摘录（MVP）

### 1) Learning while Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies（AGIBOT Research, 2026）

- **链接：** <https://finch.agibot.com/research/lwd>
- **PDF：** <https://finch-static.agibot.com/LWD/lwd-paper.pdf>
- **核心贡献：**
  - 提出 **Learning While Deploying (LWD)**，一个面向**通用 VLA 策略**的**车队级 (fleet-scale) offline-to-online RL 后训练框架**：把"部署 = 评测终点"改写为"部署 = 训练循环的一部分"。
  - 一个共享 replay buffer 同时承载离线数据（专家演示、历史 rollout、围绕失败模式的探索数据）与在线数据（自动 rollout、人为干预），由集中式学习器统一更新单一 generalist 策略并周期性把改进 checkpoint 推回车队，形成**闭环 RL 数据飞轮**。
  - **离线-在线统一 RL**：与传统"数据飞轮 = 抽好动作再做模仿"路线不同，LWD 不依赖模仿学习，使用 RL 让"成功 / 失败 / 部分进展 / 失败恢复 / 人为干预"全部转化为策略提升信号。
  - 针对车队级 RL 的两大痛点提出两个核心组件：
    - **Distributional Implicit Value Learning (DIVL)**：在异构 replay 上学一个 action-value 的**分布**，从中取**分位数统计量**作为 TD bootstrap target；保留 IQL "in-distribution value learning" 的稳定性，又能在长程任务上配合**多步 TD** 高效传播稀疏终态奖励。
    - **Q-learning with Adjoint Matching (QAM)**：针对 flow-matching 等多步生成式动作头，把"用 critic 梯度引导策略"改写为沿 flow 轨迹的**局部回归**，从而避免对完整生成过程做反向传播，也不要求动作 likelihood 可解析。DIVL 和 QAM 实现了**值估计与策略抽取的解耦**。
  - 在 **Agibot G1 双臂机器人车队**上跨 8 个真实操作任务（如长程的功夫茶、调酒、果汁、装鞋、超市补货）训练**单一 generalist 策略**：随着在线车队经验累积，长程任务上的成功率与平均周期时间显著优于以往后训练 baseline，越是 horizon 长、奖励稀疏的任务收益越大。
- **对 wiki 的映射：**
  - 主页面：[wiki/methods/lwd.md](../../wiki/methods/lwd.md)
  - 概念延伸：[wiki/concepts/data-flywheel.md](../../wiki/concepts/data-flywheel.md)
  - 范式对比：[wiki/comparisons/online-vs-offline-rl.md](../../wiki/comparisons/online-vs-offline-rl.md)
  - 上层语境：[wiki/methods/vla.md](../../wiki/methods/vla.md)、[wiki/concepts/foundation-policy.md](../../wiki/concepts/foundation-policy.md)

## 关键术语

- **Fleet-Scale RL**：在一群部署中的真实机器人（fleet）上聚合异构经验，集中训练单一 generalist 策略，再把改进 checkpoint 推回车队。
- **Offline-to-Online RL**：先用历史/演示数据做 offline RL 初始化，再用在线 rollout 做持续微调；offline 与 online 阶段共享同一个学习器与 RL 目标。
- **DIVL (Distributional Implicit Value Learning)**：把 IQL 的 implicit value learning 从标量回归扩展到对动作-价值分布的拟合，并以分位数作为 TD 目标，对异构、稀疏奖励的车队 replay 更稳定。
- **QAM (Q-learning with Adjoint Matching)**：针对 flow-matching / diffusion 动作头的策略抽取方法，沿生成轨迹做局部回归，避免完整反向传播与 likelihood 求解。
- **Generalist Policy**：单一策略承担车队上的多种任务，而非每个任务一个专精控制器。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 关联 wiki 页面的参考来源段落已添加 ingest 链接
