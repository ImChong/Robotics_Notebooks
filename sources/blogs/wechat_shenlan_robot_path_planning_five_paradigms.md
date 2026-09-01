# 机器人路径规划5大主流算法详解：MPC、强化学习、图搜索等

> 来源归档（blog / 微信公众号）

- **标题：** 机器人路径规划5大主流算法详解：MPC、强化学习、图搜索等
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/buw_88K8DlR9Tw70NDTzkw
- **发表日期：** 2026-09-01（frontmatter）
- **入库日期：** 2026-09-01
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`playwright==1.49.1`）；`--no-images`；正文约 9.5k 字符 / 22 图；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **原始落盘：** [wechat_shenlan_robot_path_planning_five_paradigms_2026-09-01.md](../raw/wechat_shenlan_robot_path_planning_five_paradigms_2026-09-01.md)
- **关联姊妹篇：** [机器人控制八范式](wechat_shenlan_robot_control_eight_paradigms.md)、[机器人学习五范式](wechat_shenlan_robot_learning_five_paradigms.md)、[移动机器人导航规划方法对比](../../wiki/comparisons/mobile-robot-navigation-planning-methods.md)
- **一句话说明：** 按 **五条技术路线**（图搜索、采样、人工势场、最优控制/MPC、强化学习与 AI）梳理路径规划范式；以「狭窄通道」场景对照各法优劣，收束为 **分层协同组合** 而非单算法万能。

## 核心摘录（归纳，非全文）

### 问题重框

- 真实导航不只是「抵达目标」：需在拥挤环境、复杂障碍与突发变化中持续做出 **安全、可执行** 的移动决策。
- 统一问题：面对未知与约束的现实环境，机器人依靠 **什么信息** 决定下一步怎么走？
- 文内对照场景：机器人须穿过仅比自身宽 5 cm 的门缝——五类算法同台检验「谁能活下来」。

### 五类范式对照

| 范式 | 核心思路 | 文内代表 | 关键风险 / 边界 |
|------|----------|----------|-----------------|
| **图搜索** | 连续环境离散化为网格/图，搜索连通路径 | Dijkstra、A\*、Hybrid A\* | 依赖预建地图；高维栅格维度灾难；折线未必动力学可行 |
| **采样方法** | 状态空间随机采样 + 碰撞检测拼路径 | RRT、PRM、RRT\* | 狭窄通道采样概率低；路径随机且锯齿，需平滑；动态障碍重规划难 |
| **人工势场** | 目标引力 + 障碍斥力，沿势能下降滑行 | APF | 局部极小值（U 形陷阱）；目标贴墙时斥力抵消；常作近场修正 |
| **最优控制** | 带约束滚动优化，保证动力学可行 | MPC | 依赖模型与快速求解器；未建模接触/传感误差时性能下降 |
| **强化学习与 AI** | 观测→动作学习；LLM 负责高层语义 | DRL、VLM+传统规划分层 | Sim2Real gap；决策难解释；工程上多与传统规划分层而非全替代 |

### 各范式要点（文内）

1. **图搜索**
   - Dijkstra 保证最短但扩展多；A\* 加启发式减少搜索范围（需可采纳启发）。
   - 自主泊车：折线 A\* 不满足 Ackermann → **Hybrid A\*** 把朝向与最大转角纳入状态，再交底层平滑。
   - 高自由度机械臂三维避障：栅格数量爆炸，实时性难保证。

2. **采样方法**
   - RRT 从起点向外「长枝」直至触达终点；PRM 预建概率路线图。
   - 7-DoF 移动操作（底盘+臂抓杯）：关节空间采样 + 碰撞检测。
   - 目标偏置、受限空间采样与路径平滑是常见工程补丁。

3. **人工势场**
   - 服务机器人近场反应：雷达/相机检测障碍，调整斥力项改航向或减速。
   - 计算量小、适合高频更新；**不能单独承担全局导航**。

4. **最优控制 / MPC**
   - 有限时域预测 + 约束优化，只执行第一步再滚动。
   - 四足高动态：落点、力矩、机身姿态纳入优化；显式写入扭矩/摩擦约束。
   - 代价：模型质量与求解速度。

5. **强化学习与 AI**
   - DRL 把激光/相机映射为速度/转向；适合非结构化局部脱困。
   - 更稳妥路线：**LLM/VLM 语义拆解子目标** + 底层仍调用传统搜索与控制（文内家庭「去书房拿杯子」示例）。
   - Sim2Real 与安全验证是落地门槛。

### 文内收束：分层协同

- 五类分别解决：全局连通（图搜索）、高维可行（采样）、近场反应（APF）、物理约束执行（MPC）、感知语义（AI）。
- **组合示例**：低分辨率 A\* 确认门位 → 门附近 RRT 精细化 → 底层 APF 防撞 → MPC 执行微调。
- 选型问题：**给定任务、算力与安全边界，什么算法组合最可靠**——而非寻找完美单算法。
- 系统课程延伸：文内指向高飞（浙江大学 FAST-Lab）主讲《移动机器人运动规划》课程（搜索、采样、动力学规划与 MPC）。

## 对 wiki 的映射

| 主题 | 关系 |
|------|------|
| [路径规划五大范式（对比）](../../wiki/comparisons/robot-path-planning-five-paradigms-taxonomy.md) | **主沉淀页**：五类路线、狭窄通道对照与分层组合 |
| [移动机器人导航规划方法对比](../../wiki/comparisons/mobile-robot-navigation-planning-methods.md) | 工程三层栈（A\* + DWA + 平滑）与本文全局/局部分工互补 |
| [A\* 全局路径规划](../../wiki/methods/a-star.md) | 图搜索代表 |
| [DWA 局部路径规划](../../wiki/methods/dwa.md) | 局部反应式对照（与 APF 同属近场层） |
| [MPC](../../wiki/methods/model-predictive-control.md) | 最优控制代表 |
| [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md) | DRL 导航/脱困 |
| [VLN](../../wiki/tasks/vision-language-navigation.md) | LLM+地图语义锚点的高层导航 |
| [机器人控制八范式](../../wiki/comparisons/robot-control-eight-paradigms-taxonomy.md) | 姊妹 taxonomy（规划 vs 控制） |
| [机器人学习五范式](../../wiki/comparisons/robot-learning-five-paradigms-taxonomy.md) | 姊妹 taxonomy（规划 vs 学习信号） |
