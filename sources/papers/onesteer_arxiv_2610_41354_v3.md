# OneSteer：单步扩散全身控制与测试时运动引导（arXiv:2610.41354v3）

> 来源归档（paper）；与旧名 DiffWBC（v1）是**同一篇论文**，统一映射 [paper-diffwbc](../../wiki/entities/paper-diffwbc.md)，不新增第二个论文实体。

- **英文标题：** OneSteer: Unlocking Steerable General Humanoid Motion Tracking via One-Step Diffusion Whole-Body Control
- **中文标题：** OneSteer：通过单步扩散全身控制实现可引导的通用人形运动跟踪
- **作者：** Yifei Huang, Yazhao Yang, Li Weijia, Tianle Shi, Ji Jin, Zhihui Cai, Zihao Liu, Jianzhuang Zhao, Songlin Dong, Xinyuan Chang, Yihong Gong, Dian Shen, Zhiheng Ma
- **机构：** 东南大学；南方科技大学；香港理工大学；深圳大学；深圳理工大学（Shenzhen University of Advanced Technology）；西北工业大学；Primotion；西安交通大学
- **论文身份：** arXiv:2610.41354，2026-10-30 v1 首发（题名 DiffWBC），2026-11-11 更新为 v3（题名 OneSteer）；仍为预印本
- **全文：** https://arxiv.org/html/2610.41354v3
- **PDF：** https://arxiv.org/pdf/2610.41354v3
- **项目主页：** https://onesteer.github.io/
- **源码状态：** 2026-11-14 项目页写「Code (coming soon)」，尚无公开可运行仓库、模型权重或部署入口
- **入库日期：** 2026-11-14
- **核对范围：** 核对 arXiv v3 §1–§5、方法公式 (1)–(7)、Tables 1–3、Q1–Q4、S1 实现设置、S2 评测协议、S3 真机/地形/对比结果、S4 steering 消融与 S5 failure self-assessment，核对官网 Real robot、Method、Terrain、Results、Self-assessment 与资源区；不把 v1 的旧评测数值当成 v3。
- **通读：** 已完成

## 核心机制（v3 §3 / S1）

1. **参考与反馈：** 每 20 ms 输入本体状态及未来 10 帧参考（相隔 0.1 s），将 root、legs、upper body 分组，由 MLP+FSQ 编码并可遮蔽。生成器是 4 层、512 宽度、8 头的 Transformer（22.8M 参数）。
2. **单步计划：** 从高斯噪声 z 和条件 c 单次前向得到未来计划 X（24 个 token：前 16 个密集，后 8 个每四 tick，覆盖 0.96 s）。每个 token 含 73 维状态及 29 维动作目标；不是多步去噪链。
3. **闭环执行：** inverse-dynamics / plan-conditioned action head 读取实时反馈、前 4 个预测状态、参考和自身 action mask，计算本 tick 控制命令；下一 tick 再生成计划。生成计划里的动作并非直接逐帧开环执行。
4. **冻结权重引导：** 任务成本 C(X) 沿生成器反传到输入 z，而不是更新模型权重；可单步归一化梯度迭代。参考 mask（m_ref）决定生成条件，执行 head 的 mask（m_act）避免它把放开的身体部位拉回原参考，引导 mask（M_guide）约束允许改动的输入通道。三个 mask **不能混用**。
5. **双次生成：** 常规跟踪一调用即可；steering 采用第一次出草稿，第二次将草稿与新噪声重混合，对第二次输入做两步梯度调整。真实时策略不是「两次调用各两步」：那是较慢的独立消融。两条训练路线分别为 MimicLite 教师在线 DAgger 蒸馏，以及 learned world model 下的 model-based RL 从零学习。
6. **失效发现：** 固定参考与反馈，以 K=8 种噪声采样计划，对关节位置的跨计划标准差按时间取 95 分位数评分，筛选困难片段再微调；它是经验性异常评分，而非校准后的失败概率。

## 实验数据与口径（v3 §4 / S2–S5）

| 项目 | v3 原文结果 | 边界 |
|---|---|---|
| 从零训练，AMASS / BONES 局部跟踪误差 | 26.5 mm / **22.6 mm** | AMASS 150 窗口、BONES **512** 窗口；仅统计成功片段的局部误差 |
| 从零训练，BONES 成功率 | **96.5%** | 18 条片段依赖场景地形，平地 harness 不可能成功；不要沿用 v1 的 97.3% |
| 蒸馏版，BONES 误差与成功率 | 27.3 mm / 94.9% | 与独立的 RL-from-scratch 模型区分 |
| 仿真：1.25 m 低杆 clean pass | 不引导 0.02 → **0.90**（仅第 2 次调用、2 步梯度）；闭环参考编辑最高 0.11 | 79 条未见过演员的行走窗口、3 次运行；不是所有方法都实时 |
| 仿真：跨栏 feasible、手部 reaching hit | 0.68 / 0.57 | 同一实时引导设置（第 2 次调用、2 步梯度） |
| 真机：普通跟踪 | **27/27** 不摔倒 | G1、9 组运动各 3 次、带松弛安全绳 |
| 真机：LiDAR SDF 避障 | **1.1 m 低杆 4/5**、**12 cm 障碍 3/3** 不跌倒 | 预先由机载 LiDAR 建图且每次固定；steering 使用系留 RTX 4090，并非纯机载执行 |
| 延迟 | RTX 4090 单次普通前向 3.8 ms；双次普通 5.3 ms；实时引导 10.6–13.7 ms | Orin NX 对应引导 14.4 ms 是离线计时；控制循环 50 Hz |
| 仿真：地形条件模型 | 21×21 height scan → 36 tokens；全地形成功率 **0.43 → 0.90** | RL 微调，88 条跨地形片段、6 地形族、10 个难度、4 次评估；**未在真机验证** |
| 无仿真失效挖掘 | 7,884 clips 中 537 次失败；top 2% precision **0.91** | 以子窗口评分；整段评分近 0.90，基准失败率 6.8% |
| 329 条筛选难例微调 | 修复 **48±4%** 既有失败 | 3 个训练种子；已掌握片段退化 1.8±0.2%；筛查比每片 5 次仿真 rollout 省 11–28 倍 |

## 版本差异与边界

- v1（2026-10-30）标题 **DiffWBC**、作者仅 5 人；v3（2026-11-11）正式标题 **OneSteer**、作者 13 人、机构扩展；**arXiv ID 不变**。旧版存档保留：[v1 原文笔记](./diffwbc_arxiv_2610_41354_reading.md)。
- v1 的 BONES 20.1 mm / 150 条窗口、1.25 m 低杆引导 0.84 / 闭环参考编辑 0.10、默认引导 20.3 ms 与「真机仅普通跟踪」等结论**不能沿用到 v3**。现以 v3 的 512 BONES 窗口、22.6 mm / 96.5%、0.90 / 0.11、14.4 ms Orin NX 离线时延与真机 SDF steering 为准。
- 论文展示 G1 真机 SDF 避障，但没有证明所有任务（弱电机/托盘/VR 目标）均在实机成功；地形扩展目前仅在仿真；自评分数不是失败概率保证。
- v3 §5 明确持续接触力尚需 wrench/tactile 表示；视觉任务条件及真实动态障碍仍属未来工作。

## 对 wiki 的映射

- [唯一论文详情：OneSteer / DiffWBC](../../wiki/entities/paper-diffwbc.md)
- [项目官网归档](../sites/diffwbc-project.md)

## 参考来源

- https://arxiv.org/html/2610.41354v3
- https://arxiv.org/abs/2610.41354
- https://onesteer.github.io/
