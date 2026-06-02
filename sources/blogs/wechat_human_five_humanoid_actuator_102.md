# (2万字）Humanoid 执行器 入门 102

> 来源归档（blog / 微信公众号）

- **标题：** (2万字）Humanoid 执行器 入门 102
- **类型：** blog
- **作者：** human five（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/zinp6ulTorzfqmCR_HaI5A
- **发表日期：** 2026-06-02
- **入库日期：** 2026-06-02
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0 + `wechat-article-for-ai`（Camoufox）；正文约 3.6 万字 / 34 图
- **原始落盘：** [wechat_humanoid_actuator_102_2026-06-02.md](../raw/wechat_humanoid_actuator_102_2026-06-02.md)
- **姊妹篇：** [Humanoid Hardware 入门 101](wechat_human_five_humanoid_hardware_101.md)（部件级 BOM 与供应链；<https://mp.weixin.qq.com/s/10hYwFzC1EuCypFVzC6QGQ>）
- **参考资料索引：** [humanoid_actuator_102_reference_catalog.md](../papers/humanoid_actuator_102_reference_catalog.md)
- **一句话说明：** 从行走冲击物理（~5000 步/时、2–3× 体重、亚毫秒退让）出发，论证质量惩罚螺旋、旋转谐波+直线滚柱分离架构、N² 反射惯量与 QDD/SEA 光谱，并给出三大「物种」决策矩阵与 >15Nm/kg 等关节指标；与 101 互补：101 讲部件清单，102 讲执行器为何在腿上失效。

## 核心摘录（归纳，非全文）

### 八组章节（对应 wiki 分类 hub）

| 组 | 公众号章节 | 分类 hub |
|----|------------|----------|
| **01 负载与质量螺旋** | I 行走难题、II 质量惩罚螺旋 | [humanoid-actuator-102-load-and-mass-spiral](../../wiki/overview/humanoid-actuator-102-load-and-mass-spiral.md) |
| **02 分离架构** | III 趋同解法 | [humanoid-actuator-102-split-architecture](../../wiki/overview/humanoid-actuator-102-split-architecture.md) |
| **03 减速与反射惯量** | IV 齿轮减速权衡 | [humanoid-actuator-102-gear-reflected-inertia](../../wiki/overview/humanoid-actuator-102-gear-reflected-inertia.md) |
| **04 热学与控制** | V 热学现实、VI PWM→力矩 | [humanoid-actuator-102-thermal-and-control](../../wiki/overview/humanoid-actuator-102-thermal-and-control.md) |
| **05 柔顺与感知** | VII 串联弹性、VIII 神经系统 | [humanoid-actuator-102-compliance-sensing](../../wiki/overview/humanoid-actuator-102-compliance-sensing.md) |
| **06 工业陷阱** | IX 平方律陷阱 | [humanoid-actuator-102-industrial-actuator-trap](../../wiki/overview/humanoid-actuator-102-industrial-actuator-trap.md) |
| **07 决策与指标** | X 决策矩阵、XI 设计要求 | [humanoid-actuator-102-decision-species](../../wiki/overview/humanoid-actuator-102-decision-species.md) |
| **08 未来路线** | XII 人工肌肉 + 参考文献导读 | [humanoid-actuator-102-future-artificial-muscle](../../wiki/overview/humanoid-actuator-102-future-artificial-muscle.md) |

### 文内收束判断（策展）

- **可行门槛：** 腿部比力矩通常需 **>10–15 Nm/kg**（竞争力 >25）；直线比力 **>4000 N/kg**。
- **架构趋同（重载通用人形）：** 主关节 **谐波旋转** + 膝踝等 **行星滚柱丝杠直线**（Tesla / Figure / Apptronik）；动态跳跃向 **QDD**（Unitree）；柔顺向 **SEA**（Digit）。
- **三大物种：** 工厂工人（滚柱+谐波）、敏捷快递（SEA/QDD）、家庭助手（低减速谐波或 QDD）。
- **隐藏选型因素：** 刚性高减速执行器 **更易仿真**，利 RL sim2real（文内 Tesla 论点）。

## 对 wiki 的映射

| 主题 | 关系 |
|------|------|
| [Humanoid Actuator 102 技术地图](../../wiki/overview/humanoid-actuator-102-technology-map.md) | 父节点 |
| [Humanoid Hardware 101 技术地图](../../wiki/overview/humanoid-hardware-101-technology-map.md) | 姊妹篇：部件与 BOM |
| [Hardware 101 · 集成执行器](../../wiki/overview/humanoid-hardware-101-integrated-actuators.md) | 部件层对照 |
| [Hardware 101 · 传动链](../../wiki/overview/humanoid-hardware-101-actuation-sensing-chain.md) | QDD vs 谐波背景 |
