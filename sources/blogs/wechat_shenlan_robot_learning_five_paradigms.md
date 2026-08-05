# 机器人学习算法五大体系详解：模仿、强化、多模态、持续学习……

> 来源归档（blog / 微信公众号）

- **标题：** 机器人学习算法五大体系详解：模仿、强化、多模态、持续学习……
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/r2zUtQfwH_r0WHrnY4CHuA
- **发表日期：** 2026-08-05（frontmatter）
- **入库日期：** 2026-08-05
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`）；`--no-images`；正文约 1.2 万字 / 21 图；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **原始落盘：** [wechat_shenlan_robot_learning_five_paradigms_2026-08-05.md](../raw/wechat_shenlan_robot_learning_five_paradigms_2026-08-05.md)
- **关联姊妹篇：** [五大具身模型详解（VLM/VLA/VLN/VLX/WM）](wechat_shenlan_five_embodied_model_taxonomy.md)、[机器人控制八范式](wechat_shenlan_robot_control_eight_paradigms.md)
- **一句话说明：** 按 **五种学习信号**（示范、奖励反馈、互联网视频先验、多模态语义、时间维能力保持）拆解机器人学习主范式：模仿学习（ACT / DAgger）→ 强化学习（Isaac Gym / Sim2Real / DR）→ Learning from Video（VideoDex）→ 多模态 VLA（RT-2 / Open X-Embodiment）→ 持续学习（灾难性遗忘与四类策略）；收束判断是 **选型组合** 而非单算法万能。

## 核心摘录（归纳，非全文）

### 问题重框

- 「机器人自主学习」常被夸大：多数能力仍依赖 **人给的示范、奖励、数据与评测边界**。
- 统一问题：面对未完整写进程序的现实环境，机器人依靠 **什么信息** 改进行为？
- 文内比喻（系鞋带）：看示范（IL）→ 语言/奖惩纠正（RL）→ 看别人系不同鞋带（视频）→ 听懂指令（多模态）→ 学会新技能不忘旧技能（持续学习）。

### 五类范式对照

| 范式 | 核心学习信号 | 文内代表 | 关键风险 / 边界 |
|------|--------------|----------|-----------------|
| **模仿学习** | 专家示范（观测→动作） | ALOHA + ACT（动作块）；DAgger | 分布偏移；单步预测误差累积 |
| **强化学习** | 环境奖励 / 累计回报 | Isaac Gym GPU 并行；ANYmal 真机迁移 | 真机试错成本与安全；Sim2Real gap |
| **从视频学习（LfV）** | 互联网人类视频先验 | VideoDex（人手/相机轨迹重定向） | 无机器人动作标签；视角与手–爪形态差 |
| **多模态（VLA）** | 视觉 + 语言 + 动作统一训练 | RT-2（动作作文本 token）；Open X-Embodiment | 开放长程规划 / 系统可靠性未覆盖 |
| **持续学习** | 随时间新增任务与数据流 | 正则化 / 架构扩展 / 经验回放 / 生成式回放 | 多为仿真或静态集评测；工程成本高 |

### 各范式要点（文内）

1. **模仿学习**
   - 高质量示范提供可用策略起点；不等于学会完整协调操作。
   - **ACT**：一次预测一段连续动作，建模短时协调（穿扎带、开半透明杯、插电池等少样本任务）。
   - **DAgger**：让当前策略访问状态，专家补标，把「会犯的错」纳入数据；专家成本与安全仍是瓶颈。

2. **强化学习**
   - 适合「难写逐步示范、但能定义好坏」的任务。
   - 仿真优先；Isaac Gym 把物理与训练放 GPU，数量级加速并行采样。
   - **Sim2Real**：域随机化降低对固定纹理/光照的依赖；真机最终验证不可替代。

3. **Learning from Video**
   - 人类视频 **不是** 带控制信号的机器人数据集；价值在任务时序、交互与物理先验。
   - VideoDex：检测人手 → 姿态/相机轨迹重定向 → 预训练策略，再少量真机示范收尾。

4. **多模态 / VLA**
   - 相对纯视觉策略补任务语义；相对纯 VLM 补可执行动作。
   - RT-2：动作文本化，与互联网 VL 数据共训，评测新物体/未见指令/基础语义推理。
   - Open X-Embodiment：跨平台数据共训观察到正迁移；硬件与动作空间差异仍是硬约束。

5. **持续学习**
   - 核心难题：**灾难性遗忘**。
   - 策略族：参数正则化、动态扩容、记忆回放、生成式回放（可重叠）。
   - 更适合当作研究目标与评估框架，而非无条件可部署通用能力。

### 文内收束判断

- 五类解决的是学习过程中的 **不同子问题**，工程上常 **组合**：示范初始化 + RL 局部优化 + VLM/VLA 语义。
- 选型问题应回到：给定任务、数据与安全边界，**什么学习信号最可靠，什么验证最能反映真实能力**。

## 文末参考文献（标题级索引）

| # | 文献 | 用途（文内） |
|---|------|--------------|
| 1 | Zhao et al., ALOHA / ACT, RSS 2023 | 模仿学习动作块 |
| 2 | Ross et al., DAgger, AISTATS 2011 | 分布偏移纠偏 |
| 3 | Sutton & Barto, RL: An Introduction | RL 教材 |
| 4 | Makoviychuk et al., Isaac Gym, 2021 | GPU 并行仿真训练 |
| 5 | Tobin et al., Domain Randomization, IROS 2017 | Sim2Real |
| 6–8 | Sim2Real survey / Frontiers review / Hwangbo ANYmal | 迁移与四足真机 |
| 9–10 | LfV survey (JAIR 2025)；VideoDex, CoRL 2023 | 视频学习 |
| 11–12 | RT-2；Open X-Embodiment, CoRL 2023 | VLA / 跨具身数据 |
| 13 | Lesort et al., Continual Learning for Robotics, 2020 | 持续学习框架 |

## 对 wiki 的映射

| 主题 | 关系 |
|------|------|
| [机器人学习五大范式（对比）](../../wiki/comparisons/robot-learning-five-paradigms-taxonomy.md) | **主沉淀页**：五类学习信号、边界与组合选型 |
| [Imitation Learning](../../wiki/methods/imitation-learning.md) | ACT / DAgger / 分布偏移 |
| [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md) | 奖励驱动与仿真试错 |
| [VLA](../../wiki/methods/vla.md) | 多模态执行层 |
| [Sim2Real](../../wiki/concepts/sim2real.md) / [Domain Randomization](../../wiki/concepts/domain-randomization.md) | RL 真机迁移 |
| [RL vs IL](../../wiki/comparisons/rl-vs-il.md) | 双主干对照的超集入口 |
| [五大具身模型分类](../../wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md) | 姊妹 taxonomy（模型族 vs 学习信号） |
| [Robot Learning Overview](../../wiki/overview/robot-learning-overview.md) | 学习方法层导航 |
