# 强化学习真机部署：机器人在仿真里，究竟学会了什么？

> 来源归档（blog / 微信公众号）

- **标题：** 强化学习真机部署：机器人在仿真里，究竟学会了什么？
- **副标题（文内）：** 足式机器人：从仿真到真机部署，有哪些你不知道的「坑」？
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/d_eV4jRgpEf9v4BMqACGWw
- **发表日期：** 2026-09-18
- **入库日期：** 2026-09-18
- **抓取方式：** 移动端 WeChat UA + HTML 解析（`curl`；`--no-images`）；本环境未预装 `wechat-article-for-ai`；Jina Reader / WebFetch 对微信超时或 CAPTCHA
- **原始落盘：** [wechat_shenlan_rl_sim2real_deployment_2026-09-18.md](../raw/wechat_shenlan_rl_sim2real_deployment_2026-09-18.md)
- **一句话说明：** 从「策略成立条件」视角拆解足式 RL 真机部署：动作语义、观测来源、时序延迟、执行器/地面模型、奖励与 DR 边界、RMA 在线适应与长时测试——强调仿真里学会的是**特定身体+信息+环境**下的有效动作，而非可随处复用的「走路本领」。

## 核心摘录（归纳，非全文）

### 主判断

- **误区：** 把 `.pt` 模型「原样搬进真机」；数组维度对上但**语义/时序/身体**任一错位，行为即变。
- **正读：** 部署 = 弄清仿真策略依赖的**条件**在真机上如何变化，并让策略、模型与控制系统共同消化这些变化。
- **与姊妹篇关系：** [SysID→适应闭环文](wechat_shenlan_sim2real_sysid_to_adaptation.md) 偏**全流程工程**；本篇偏**部署侧条件对齐**与**失效前诊断**。

### 七类「条件变化」（文内主线）

| # | 侧面 | 要点 | 文内锚点 |
|---|------|------|----------|
| 1 | **动作语义** | 网络输出 → 缩放 → 叠加默认姿态 → PD 目标；关节顺序/符号/增益须与训练一致 | Unitree-RL-GYM 部署代码 |
| 2 | **观测来源** | 仿真可读特权状态；真机靠 IMU/编码器估计；IMU 位姿/滤波节奏改变即换世界 | 身体速度、接触标志 |
| 3 | **感知→支撑** | 高程图「看见表面」≠ 可承重落脚（草丛等） | 图2 植被 vs 地图起伏 |
| 4 | **时序/延迟** | 策略低频更新 vs 底层高频执行；端到端延迟使指令对应「旧身体」；实时 = 截止期 + 稳定 jitter | ROS 2 实时说明 |
| 5 | **身体/执行器** | 理想电机 vs 学习执行器模型（ANYmal）；外形对齐 ≠ 响应对齐 | 图3 力矩预测对比 |
| 6 | **地面/接触** |  deformable 地面 ≠ 换摩擦系数；支撑反馈改变闭环 | 图5 泡沫 vs 木板 |
| 7 | **运行中变化** | 载荷、温升限流、机载多任务争用算力；**成功演示 ≠ 任务能力** | 图7 长时测试 |

### 训练侧补充

- **奖励：** 速度/姿态/平滑/力矩项共同塑造步态；单项惩罚可能压制必要调整。
- **域随机化：** 教策略应对**训练环境实际改变**的条件；扩大摩擦 ≠ 产生下陷/软土；观测噪声 ≠ 通信队列延迟。**单位/映射错误应直接修正**，不宜用 DR 掩盖。
- **RMA：** 用近期状态–动作历史压缩环境摘要，边走边适应载荷/摩擦等变化；**不能替代**接口与时序对齐。

### 测试与失效分析

- 真机测试应**有对照含义**：换地面→接触；加载荷→身体响应；延长时间→热/限流。
- **失效前数据**比倒地画面更有价值：目标、响应、时间戳如何错开 → 决定改模型 / DR / 观测 / 时序。
- 结论句：足式机器人在仿真里学的是**在特定条件下产生有效动作的方法**；只有目标载荷、地面、工作时长下**反复完成**，才可称能力已落到真实世界。

## 对 wiki 的映射

- [rl-sim2real-deployment-conditions](../../wiki/queries/rl-sim2real-deployment-conditions.md)（本次升格主页面）
- [robot-policy-debug-playbook](../../wiki/queries/robot-policy-debug-playbook.md) — 症状决策树互补
- [sim2real-closed-loop-engineering](../../wiki/queries/sim2real-closed-loop-engineering.md) — SysID→适应闭环
- [sim2real](../../wiki/concepts/sim2real.md)、[domain-randomization](../../wiki/concepts/domain-randomization.md)、[paper-rma-rapid-motor-adaptation](../../wiki/entities/paper-rma-rapid-motor-adaptation.md)
- [unitree-rl-gym](../../wiki/entities/unitree-rl-gym.md) — 文内动作/观测转换代码锚点

## 可信度与使用边界

- 策展解读 + 课程宣传尾部已截断；工程细节以官方部署代码与论文为准。
- 图1–7 为微信 CDN，未纳入 wiki 正文。
