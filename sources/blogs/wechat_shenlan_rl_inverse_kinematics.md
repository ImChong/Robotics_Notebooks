# 强化学习求解逆运动学：五类方案

> 来源归档（blog / 微信公众号）

- **标题：** 强化学习求解逆运动学，这五类方案值得重点关注！
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号；《具身智能基础》专栏第 7 篇）
- **原始链接：** https://mp.weixin.qq.com/s?__biz=MzkwMDcyNDUzMQ==&mid=2247506122&idx=1&sn=8119b7177642fe8f467e1068977d24d8
- **发表日期：** 2026-07-09
- **入库日期：** 2026-08-13
- **抓取方式：** Agent Reach v1.5.0 + wechat-article-for-ai（Camoufox）；专辑页同会话跳转；`--no-images` 解析
- **专栏专辑：** [《具身智能基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653)
- **专栏姊妹篇：** [正向运动学](wechat_shenlan_forward_kinematics.md)；[逆运动学](wechat_shenlan_inverse_kinematics.md)
- **一句话说明：** FK 是固定几何映射，工业臂很少用 RL 算正运动学；IK 因多解、约束、奇异、标定漂移更适合 RL。文内把 RL-IK 收成五类：DDPG 单臂、PPO/MAPPO 多约束/双臂、模型基（学 FK 再反传）、雅可比伪逆+零空间 RL 混合、分层 RL 人形全身 IK。结论：**RL 补传统方法不擅长的冗余/多目标/非标，不替代雅可比精度兜底。**

## 核心摘录（归纳，非全文）

### 为何 IK 更吃 RL

1. **多解**：冗余 7 轴 / 双臂，传统局部解不够；RL 可搜索多组可行关节。
2. **多约束**：限位、自碰、障碍、平滑可写入奖励，不必手工拆公式。
3. **奇异**：$det(J)=0$ 时数值 IK 停滞；RL 可绕行。
4. **工况漂移**：磨损/标定偏差下固定公式退化，RL 可在线微调。
5. **FK 例外**：软体连续体、无精准 DH 的非标臂，正模型也可能用网络拟合。

### 五类方案

| 类 | 算法骨架 | 适配 |
|----|----------|------|
| 1 | DDPG 端到端连续 IK | 固定自由度工业臂；避开伪逆爆炸 |
| 2 | PPO 单臂 / MAPPO 双臂 | 多约束、协同；可分层拆大臂/小臂 |
| 3 | 数据驱动正模型 + 模型基 RL | 软体/非标；「学 FK + 梯度逆求解」 |
| 4 | 伪逆主空间 + RL 调零空间 | 高精度避障转运；精度由数值 IK 兜底 |
| 5 | 分层 RL 全身 IK | 浮基人形：上层笛卡尔轨迹、下层分肢 IK |

选型口诀：单臂 DDPG → 多约束 PPO → 双臂 MAPPO → 非标模型基 → 精度+灵活选混合 → 人形分层。

## 对 wiki 的映射

- 升格 [`wiki/comparisons/rl-inverse-kinematics-five-approaches.md`](../../wiki/comparisons/rl-inverse-kinematics-five-approaches.md)。
- 交叉：[`wiki/formalizations/inverse-kinematics.md`](../../wiki/formalizations/inverse-kinematics.md)、[`wiki/formalizations/forward-kinematics.md`](../../wiki/formalizations/forward-kinematics.md)、[`wiki/formalizations/robot-jacobian.md`](../../wiki/formalizations/robot-jacobian.md)、[`wiki/overview/shenlan-embodied-ai-fundamentals-series.md`](../../wiki/overview/shenlan-embodied-ai-fundamentals-series.md)。
