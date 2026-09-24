# Towards Universal Post-Training for Robotics（Perry Dong）

> 来源归档（blog / 个人研究博客）

- **标题：** Towards Universal Post-Training for Robotics
- **类型：** blog
- **作者：** Perry Dong（Chelsea Finn 联署）
- **原始链接：** <https://pd-perry.github.io/posts/post-training.html>
- **发表日期：** 2026-09（September 2026）
- **入库日期：** 2026-09-24
- **抓取方式：** 官方页直连（WebFetch）
- **一句话说明：** Stanford Perry Dong 论述机器人已处「GPT-2 时刻」——预训练 VLA/WAM 能演示复杂行为但可靠性不足；需像 LLM 一样收敛 **通用 post-training 配方**（算法 + 标准协议）；以 **EXPO(-FT)** 为 value-based 微调扩散/flow 策略的样本，并列出 reward/reset/HIL/超参/初始化等开放默认项。

## 开源 / 项目页核查（步骤 2.5）

| 项 | 结论（截至 2026-09-24） |
|----|-------------------------|
| 博客页 | [pd-perry.github.io/posts/post-training.html](https://pd-perry.github.io/posts/post-training.html) |
| EXPO-FT 项目页 | [pd-perry.github.io/expo-ft/](https://pd-perry.github.io/expo-ft/) — 真机视频、Q 可视化、BibTeX（arXiv:2605.25477）；**页内无 GitHub 链接** |
| Real-Time EXPO-FT | [pd-perry.github.io/real-time-expo-ft/](https://pd-perry.github.io/real-time-expo-ft/) — 同上，**待发布** |
| 代码 / 权重 | **待发布** — 项目页未列可运行仓库；本库 [Real-Time EXPO-FT](../../wiki/entities/paper-real-time-expo-ft.md) 实体已标待发布 |
| 可信度边界 | 观点 + 系统总结文，非 peer-reviewed；EXPO-FT 数字来自 CoRL 2026 论文与项目页 |

## 核心摘录（归纳，非全文）

### 问题陈述

- 2026 年 Physical Intelligence、Generalist、DeepMind 等 **预训练策略** 已能完成复杂任务，但 **95% 成功率** 在家庭/工厂仍不够（需更多 nines）。
- 类比 LLM：**GPT-2/3 流畅但不可靠** → **SFT + RLHF + RLVR** 收敛为可部署配方。
- 机器人部署需 **比 LLM 更高可靠性**——坏动作不能等人审。

### LLM post-training 四步「配方」

1. 强预训练模型
2. 定义环境与 reward（可验证或偏好模型）
3. anchor 参考模型的 RL 优化
4. 监控 reward hacking 等病理

### 机器人为何是不同 RL 问题

| 维度 | LLM / Go | 机器人 |
|------|----------|--------|
| 样本成本 | 并行生成廉价 | 真机每步昂贵；长链任务可达 **500+ 控制步** |
| 奖励 | 单 response 即时可验 | 常 **仅 episode 末** 稀疏成功信号 |
| 环境 | 近似确定性 | 执行误差、接触滑移、传感噪声、扰动 |
| 主流 RL | on-policy policy gradient（PPO/GRPO） | 需 **value-based** 长视界 credit assignment |

### 扩散/flow 策略 + value RL 的四类 prior 工作

1. **Q 梯度反传进扩散链**（DDPG 式；步数多时不稳/贵）
2. **多候选 + Q 选最大**（EMaQ/SfBC/IDQL；策略权重不更新）
3. **引导去噪步**（∇Q 监督中间步；易不稳）
4. **噪声空间 steering**（冻结策略；能力受预训练上限约束）

### EXPO(-FT) 主张（本篇技术锚点）

- **EXPO（ICLR 2026）：** 大 expressive base policy（模仿）+ **轻量 edit policy** 把小修正推向高 Q 区；RL  volatility 隔离在小模型。
- **EXPO-FT（CoRL 2026，arXiv:2605.25477）：** 预训练 **VLA** 在线 RL 微调；VLA 采样多 chunk → edit 修正 → **Q 选最优** → 成功轨迹 **吸收回大模型**。
- **Real-Time EXPO-FT（arXiv:2609.18207）：** 慢 VLA 提案 + 快 edit 应对动态/延迟。
- **数字（博客）：** 六复杂操纵任务 **30/30** 成功，平均 **~19 分钟** 在线交互；对比 SFT、HG-DAgger、DSRL、HIL-SERL。

### 「配方」仍缺的协议层（开放问题）

- **Reward：** 无 LLM RLVR 级默认；手工 detector vs 人判 vs 学习分类器
- **Reset：** 人工 vs 学习 reset policy vs 不可逆任务设计
- **Human-in-the-loop：** 干预频率与如何写入 replay
- **超参：** LR、update-to-data ratio、horizon、控制频率无 SL 级默认
- **初始化：** 离线数据量与在线经验权重

## 对 wiki 的映射

| 主题 | 建议落点 |
|------|----------|
| 通用 post-training 框架 | 新建 `universal-post-training-robotics` concept |
| EXPO-FT 技术细节 | 交叉 [`paper-real-time-expo-ft`](../../wiki/entities/paper-real-time-expo-ft.md)、[`paper-qwm`](../../wiki/entities/paper-qwm.md) |
| VLA 部署可靠性 | [`foundation-policy`](../../wiki/concepts/foundation-policy.md)、[`vla-deployment-guide`](../../wiki/queries/vla-deployment-guide.md) |
| 自博弈 post-training 对照 | [`skild-physical-self-play`](../../wiki/entities/skild-physical-self-play.md) |
