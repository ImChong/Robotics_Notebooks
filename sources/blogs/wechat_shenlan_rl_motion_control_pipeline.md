# 强化学习必备知识②：机器人运动控制完整 pipeline

> 来源归档（blog / 微信公众号）

- **标题：** 强化学习必备知识②：机器人运动控制完整pipeline
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号；《具身智能基础》专栏第 6 篇）
- **原始链接：** https://mp.weixin.qq.com/s?__biz=MzkwMDcyNDUzMQ==&mid=2247505497&idx=1&sn=0f63d89762a07ba7ac642d876bfba5eb
- **发表日期：** 2026-06-25
- **入库日期：** 2026-08-13
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`--no-images` 解析）；直连遇 CAPTCHA，**专辑页同会话跳转**成功；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **专栏专辑：** [《具身智能基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653)
- **专栏姊妹篇：** [RL 最小闭环①](wechat_shenlan_rl_embodied_minimal_closed_loop.md)；[人形 RL 五模块](wechat_shenlan_humanoid_rl_policy_training_system.md)
- **一句话说明：** 把四足运动控制 RL 从「最小闭环」扩成工程管线：观测–动作–环境–奖励 → 高层 DRL（约 50 Hz 目标关节）+ 低层 PD（200–1000 Hz 力矩）→ PPO clip → Teacher-Student 蒸馏特权信息 → 稀疏奖励涌现步态 → 域随机化跨 Sim2Real → GPU 并行仿真。

## 核心摘录（归纳，非全文）

### 基础闭环四要素

| 要素 | 具身含义 |
|------|----------|
| 观测 $o$ | 关节角/角速度、机身倾角、高度图、相机；对应 POMDP 局部观测 |
| 动作 $a$ | 策略网络输出的连续控制指令 |
| 环境 $E$ | 仿真或真机动力学，给出 $o_{t+1}$ |
| 即时奖励 $r$ | 任务指标量化的单步收益 |

训练循环：交互采样 → 存 buffer → 更新策略，最大化折扣累积回报。

### 分层：高层位置、低层力矩

- **高层 DRL**：约 50 Hz，输出目标关节位置，不直接出力矩。
- **低层 PD**：200–1000 Hz，$\tau = K_p(q^*-q) - K_d\dot q$。
- 文内三点好处：降低学习难度、PD 弹簧吸收落地冲击、抹平电机/模型微小差异。

### PPO clip

约束新旧策略概率比，工程 $\varepsilon\approx 0.2$；流程为采集轨迹 → GAE 优势 → clip 更新策略 → MSE 拟合价值。

### Teacher-Student

仿真特权（精确地形、摩擦、质心）只给教师；学生仅用真机可观测，蒸馏损失为师生 KL + 任务 RL 损失。

### 奖励与涌现

不以动作捕捉模仿为必须：速度跟踪 + 能耗惩罚 + 平滑惩罚 + 姿态惩罚即可涌现 Trot——四足结构下小跑常是最省力稳定解。

### 域随机化与并行

每 episode 采样质量 ±20%、摩擦（冰面↔砂纸）、传感器噪声、电机延迟；Isaac Gym 级 GPU 可在单卡并行数千机器人。

### 局限（文末）

仍高度任务专用：新技能 ≈ 新奖励工程 + 新仿真环境。开放世界持续学习尚无满意答案。

## 对 wiki 的映射

- 升格 [`wiki/overview/robot-rl-motion-control-pipeline.md`](../../wiki/overview/robot-rl-motion-control-pipeline.md) 作为专栏 06 管线父节点。
- 交叉：[`wiki/concepts/embodied-rl-minimal-closed-loop.md`](../../wiki/concepts/embodied-rl-minimal-closed-loop.md)、[`wiki/methods/ppo.md`](../../wiki/methods/ppo.md)、[`wiki/concepts/privileged-training.md`](../../wiki/concepts/privileged-training.md)、[`wiki/concepts/domain-randomization.md`](../../wiki/concepts/domain-randomization.md)、[`wiki/overview/humanoid-rl-policy-training-five-modules.md`](../../wiki/overview/humanoid-rl-policy-training-five-modules.md)、[`wiki/overview/shenlan-embodied-ai-fundamentals-series.md`](../../wiki/overview/shenlan-embodied-ai-fundamentals-series.md)。
