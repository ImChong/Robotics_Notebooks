# Science Robotics：伯克利×斯坦福，用扩散模型让人形机器人学会全身运动

> 来源归档（blog / 微信公众号 · 深蓝具身智能）

- **标题：** Science Robotics：伯克利×斯坦福，用扩散模型让人形机器人学会全身运动
- **类型：** blog / wechat / survey / humanoid-control
- **作者：** 深蓝具身智能（编辑｜小小怪博士；审编｜具身君）
- **原始链接：** https://mp.weixin.qq.com/s/6RWZDUz00tX8pwXKVk_Lcw
- **入库日期：** 2026-09-10
- **原始抓取落盘：** [`wechat_shenlan_beyondmimic_science_robotics_2026-09-10.md`](../raw/wechat_shenlan_beyondmimic_science_robotics_2026-09-10.md)
- **姊妹索引：** [42 篇 RL 运动控制](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)（身体系统栈 #15/42）；[arXiv:2508.08241](../papers/humanoid_rl_stack_15_beyondmimic_from_motion_tracking_to_versatile_hu.md)
- **一句话说明：** 对 *Science Robotics* 正式发表的 BeyondMimic（DOI [10.1126/scirobotics.adx8924](https://doi.org/10.1126/scirobotics.adx8924)）的中文深度导读——两阶段「RL 跟踪 → 潜空间扩散 + classifier guidance」、G1 零样本下游任务与消融结论；**复用既有** [`beyondmimic`](../../wiki/methods/beyondmimic.md) 方法页，不新建实体。

## 核心摘录（归纳，非全文）

### 总判断

BeyondMimic 把「模仿动作」与「组合技能解决未知任务」串成一条链：**训练阶段只学运动先验（任务无关）**，**测试阶段用可微代价函数 + classifier guidance 零样本适配**（航点、摇杆、关键帧补全、避障）。与 DeepMimic / 分层规划+跟踪器解耦 / VAE 多任务先验等路线的对照见文内「底层矛盾」小节。

### 阶段 ①：可扩展 RL 运动跟踪（compact MDP）

| 设计点 | 要点 |
|--------|------|
| **锚点相对跟踪** | 以躯干为锚，约束连杆相对姿态；允许全局合理漂移，提升 sim2real 鲁棒性 |
| **极简统一奖励** | 跟踪主项 + 关节限位 / 动作平滑 / 自碰撞三项正则；无大量手工启发式 |
| **单步观测（无历史堆叠）** | 跟踪策略不堆叠长历史，避免过拟合仿真特有时序（与论文 III-B 一致） |
| **失败率自适应采样** | 长序列按片段失败率加权 reset，加速侧手翻等难点收敛 |
| **适度域随机 + 精确 armature** | 摩擦 / 关节零位 / 躯干 CoM 有限随机；过度 DR 反而保守僵硬 |

数据与部署：约 **2.5 h** 人类动捕；**21** 个代表性片段零样本上 **Unitree G1**（侧手翻、旋踢、连续特技、舞蹈、倒地起立等）。

### 阶段 ②：潜空间状态–动作扩散 + classifier guidance

- **蒸馏路径**：条件 VAE（DAgger 采教师 rollout）→ 潜轨迹上训练 Transformer 去噪器；**联合建模未来状态与动作**（非仅关节角）。
- **测试时引导**：任务代价梯度注入扩散去噪；航点 / 速度 / SDF 避障 / 关键帧硬约束可**相加组合**。
- **与「先规划后跟踪」差异**：规划与控制在同一扩散模型内闭环滚动下发，规避规划–控制器失配。

### 实验与消融（文内数字，以 DOI 原文为准）

| 类别 | 数字 / 结论 |
|------|-------------|
| 用户调研 | 77 人；**70.8%** 偏好 BeyondMimic 行走/跑步拟人度；跑步 **84.7%** |
| 高动态实机 | 室外非理想地面 180° 侧手翻、连续旋踢、360° 翻转踢；腾空骨盆角速度最高 **15.7 rad/s** |
| 下游速度跟踪 | 仿真行走 / 奔跑速度误差 **12.14% / 13.65%** |
| Rot6D 表示 | 相对四元数 / 轴角，跟踪误差显著降低 |
| 延迟敏感性 | **5 ms** 通信延迟即可失败 → 低延迟 C++ 部署栈是前提 |
| 预测视野 | 扩散策略约 **0.64 s** 前瞻，适合局部反应式控制 |

### 局限（文内强调）

- 模式切换起止瞬态易踉跄；引导权重需人工调；**无原生视觉**（障碍靠外部 SDF 代价）；能力上限受 RL 教师约束；精细动作控制弱于粗粒度代价目标。

### 开源核查（步骤 2.5，2026-09-10）

| 组件 | 链接 | 结论 |
|------|------|------|
| 跟踪阶段 | [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking) | **已开源**（Isaac Lab + WandB Registry） |
| 扩散 / 部署 | [beyondmimic.github.io](https://beyondmimic.github.io/) | 项目页演示完整两阶段；扩散与 C++ 部署细节以论文 / 社区 fork（如 [beyondmimic-reproduction](../../sources/repos/beyondmimic-reproduction.md)）为准 |
| 生态复现 | MJLab、Unitree RL Lab 等 | 文内称已纳入参考实现；跨机型配方迁移 |

## 对 wiki 的映射（全部复用既有节点）

- **主更新页：** [BeyondMimic 方法页](../../wiki/methods/beyondmimic.md) — 补 *Science Robotics* 正式发表、实验数字、阶段 ① 无历史观测口径、局限与参考来源
- **对比：** [SONIC vs BeyondMimic vs SD-AMP vs Heracles](../../wiki/comparisons/sonic-vs-beyondmimic-vs-sdamp-vs-heracles.md)
- **选型：** [人形运动跟踪方法选型](../../wiki/queries/humanoid-motion-tracking-method-selection.md)
- **流水线：** [Whole-Body Tracking Pipeline](../../wiki/concepts/whole-body-tracking-pipeline.md)
- **硬件：** [Unitree G1](../../wiki/entities/unitree-g1.md)

## 可信度与使用边界

- 第三方中文解读，不是论文原文；DOI / arXiv / 项目页优先。
- **不**因本推文新建 `paper-beyondmimic` 或重复方法页——BeyondMimic 已在 wiki 图谱中（`wiki-methods-beyondmimic`）。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 复用既有 BeyondMimic 方法页，不重复造节点
- [x] 补 Science Robotics DOI 与 G1 零样本下游实验摘要
- [x] 交叉对比页 / 流水线概念页轻量互链
