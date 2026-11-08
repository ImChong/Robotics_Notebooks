# 机器人强化学习最近三年在做什么？奖励设计、VLA 后训练、世界模型……（深蓝学院·具身君）

> 来源归档（blog / 微信公众号）

- **标题：** 机器人强化学习最近三年在做什么？奖励设计、VLA后训练、世界模型……
- **类型：** blog（综述性观察文章）
- **作者：** 深蓝学院-具身君（公众号「深蓝AI」，审编具身君；图为「深蓝具身智能」编译）
- **原始链接：** https://mp.weixin.qq.com/s/VjEQ9N8Ncxo4cIQU_iP-QA
- **发表日期：** 2026-11-08
- **入库日期：** 2026-11-08
- **抓取方式：** `scripts/fetch_wechat_article.py` 取正文；Chromium（微信 UA）取回全部 17 张图并看过（含 2 张 GIF 首帧之外的帧未逐帧看）。
- **核对范围：** 正文文字读完至文末「往期推荐」；17 张图都看过：图 1（GIF）与图 3（GIF）只看首帧，其余为论文截图 / 小节标题图 / 封面图。文章**没有给出论文名与出处**，本库根据配图与检索识别出被引论文，并各建独立节点逐篇通读核对（见下表）。文末 18 条评论未读。**没有核对**文章开头「2026 年 NVIDIA 公开的人形机器人训练流程用 PPO」的原始发布页，只在本库已有页面里找到相关记载。
- **通读：** 已完成
- **一句话说明：** 论点：「变的不是优化器，是强化学习被用在什么东西上」——PPO 在机器人里没被淘汰，因为机器人 RL 采样贵、奖励脏、分布会漂；三年主线是 LLM 辅助奖励设计 → 仿真物理参数与奖励协同 → 把 RL 接到预训练 VLA 之后（直接 PPO / 残差 RL / 世界模型内 rollout）。

## 文章要点（作者表述）

- **PPO 为何还在：** GRPO、DAPO 成立靠三个条件——采样便宜（LLM rollout 是 GPU 前向，真机一次尝试是分钟级）、奖励干净（可验证终局奖励 vs 机器人稠密整形奖励）、动作之间互不干扰（组内同分布 vs 机器人每个动作改变后续物理状态）。机器人 RL 的三个硬约束：采样贵、奖励脏、分布会漂，对应 PPO 的 on-policy 加信任域。
- **奖励：** 2023 年起 LLM 写奖励程序、仿真检验；2024 年与仿真物理参数调整结合；自动化的只是搜索，判断学会了什么仍靠人。
- **仿真到真机：** 域随机化改的是优化目标（「在某环境最优」→「在一族环境都不崩」）；单纯堆训练步数解决不了分布外失败。
- **VLA 后训练：** 起点变成已有能力的预训练模型，第一步更新必须保守，这是 PPO 留下的直接原因；2025 年 VLA 对照实验 PPO > GRPO、DPO；三条路线：直接 RL（PPO）、残差 RL、世界模型内 rollout（图 7 对比表）。
- **三个开放问题：** 后训练改进能否迁移到新物体 / 场景 / 任务；省下多少真机采样；奖励设计与策略优化怎么适配机器人任务。

## 被引论文识别与本库核对（逐篇通读后）

| 文章说法 | 识别出的论文 | 本库核对结论 | 节点 |
|----------|--------------|--------------|------|
| 2023 年 LLM 写奖励，图 1 比较 LLM 与人工奖励 | Eureka（arXiv 2310.12931） | 对比在论文图 4（逐任务附录 F），论文图 1 是任务拼图；自动循环反馈的是奖励分量 / 成功率的训练统计，不是机器人动作或视频；转笔仅仿真且依赖课程学习；无真机结果 | [Eureka](../../wiki/entities/paper-eureka-llm-reward-design.md) |
| 2024 年奖励自动设计 + 物理参数调整，图 2 四种地面 | DrEureka（arXiv 2406.01967） | 方法是串行三阶段，不是联合搜索；四种地面（Default / Grass / Sidewalk / Socks）在论文图 4、5，是与人工奖励的真机鲁棒性对比，不是图 2；「湿滑地面」「斜坡」论文没有 | [DrEureka](../../wiki/entities/paper-dreureka-sim-to-real.md) |
| 2025 年 VLA 对照 PPO > GRPO、DPO，图 3 | What Can RL Bring to VLA Generalization?（arXiv 2505.19789，图 3 为论文图 3b 学习曲线） | PPO 对 GRPO / DPO 只比了训练任务上的成功率曲线；执行条件变化下的结果是 PPO 对 SFT；只有一个任务、2 个种子；解释是假设，没有消融 | [VLA 泛化实证](../../wiki/entities/paper-rl-vla-generalization-empirical-study.md) |
| 「2026 年的 VLA 强化学习流程」，图 4 | 图是 SimpleVLA-RL（arXiv 2509.09674） | 论文是 **2025 年**，图是论文图 2；它用 **GRPO**（去 KL、动态采样等 DAPO 技巧）而非 PPO，没做 PPO 对照；与文章「PPO 仍是机器人首选」的主线有张力 | [SimpleVLA-RL](../../wiki/entities/paper-simplevla-rl.md) |
| 2026 年仿真训练残差 RL 叠加预训练 VLA，图 5 | Object-Centric Residual RL（arXiv 2606.18953） | 任务含「关抽屉」没有「开抽屉」；训练快无数据；「补不出新动作」与论文局限（只能修正轻度偏差）及附录 A.9 的定性反例不完全一致；弱基座增益大、强基座 π0.5 在 Pick-and-Place 持平 | [残差 RL](../../wiki/entities/paper-object-centric-residual-rl-vla.md) |
| 2026 年世界模型 + VLA RL，图 6 | VLA-MBPO（arXiv 2603.20607） | 统一世界模型、分支 rollout、Flow-Noise PPO 成立；「接触预测不准」论文没这么写（失败案例是部分可观测、大位移、织物形变）；「预测越长越偏」只有间接证据；真机无在线 RL 基线 | [VLA-MBPO](../../wiki/entities/paper-rcl-2603-20607-towards-practical-world-model-based-reinforcemen.md) |
| DreamerV3（2023）在学到的环境动态里训练策略 | DreamerV3（已有节点） | 本库已有页面，未重读 | [DreamerV3](../../wiki/entities/paper-shenlan-wm-13-dreamerv3.md) |
| 2026 年 NVIDIA 人形训练流程运动策略用 PPO | 未在文章中指明 | 本库已有 [GR00T 全身控制](../../wiki/entities/gr00t-wholebodycontrol.md) 记载 SONIC 用 PPO 加辅助损失在 Isaac Lab 训练；没有核到 NVIDIA 原始发布页 | — |
| GRPO、DAPO 在 LLM 后训练流行；三个成立条件 | 未给出处 | 三个条件是作者的论证，没有引用；SimpleVLA-RL 在仿真里用 0/1 结果奖励与 GRPO 取得大幅提升，说明条件不是绝对的（本库观察） | [GRPO](../../wiki/methods/grpo.md) |

## 对 wiki 的映射

- [机器人强化学习三年观察（2023–2026）](../../wiki/concepts/robot-rl-three-years-2026.md)
