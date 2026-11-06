# RoboDojo 论文通读（统一 sim-and-real 评测视角）

> 来源归档（原文通读 · 支撑 `wiki/entities/robodojo.md`）

- **标题：** RoboDojo: A Unified Sim-and-Real Benchmark for Comprehensive Evaluation of Generalist Robot Manipulation Policies
- **作者：** Tianxing Chen、Yue Chen 等（MMLab@HKU、UC Berkeley、THU、PKU 等联合）
- **类型：** paper / benchmark / manipulation / sim2real / evaluation
- **原文：** <https://arxiv.org/abs/2607.04434>（2026-11 通过 arXiv PDF v3 获取全文，pdftotext 提取）
- **一句话说明：** 提出 42 仿真任务（五维）+ 18 真机任务（三本体）的统一评测，配 Isaac Sim 异构并行、RoboDojo-RealEval 云真机与 XPolicyLab 统一策略接口；30 个策略的冻结榜（2026-07-03）显示最强仿真平均成功率仅 8.80%、真机最强 12.8%，人类遥操 76.03% / 100%。
- **核对范围：** 通读摘要、§1–§8 正文（含两张榜单、六条仿真 Finding、三条真机 Finding、效率与稳定性实验、Future Extensions）及附录 A（榜单治理与验证细则）、B（基准对比表）、C（真机稳定性）、D/E（训练数据与评测设置）、F（仿真平台）、G（RealEval 硬件）、H（XPolicyLab 接口与协议）、K（各策略训练配置）；附录 I/J 的逐任务说明（指令文本、数据来源）仅按任务列表浏览，未逐条核对；参考文献列表未逐条核对。
- **通读：** 已完成

## 核心摘录（面向 wiki 补强）

- **仿真任务：** ARX X5 双臂（臂基座相距 0.6 m），42 任务 = Generalization 12 / Memory 6 / Long-Horizon 8 / Precision 8 / Open 8；Generalization 干扰物最多 25 个（RoboTwin 2.0 至多 10）。
- **训练数据：** 仿真 35 个任务目录 3,500 条轨迹、20.66 h（25 Hz），每任务 100 条；Open 维不提供训练演示；另有 100 条域随机化 DLC 辅助轨迹。真机 1,800 条、17.91 h，每任务 100 条、4 名操作员。
- **评测设置：** 仿真每任务 50 episode（Generalization 为 25 标准 + 25 随机），总分为五维均值而非任务均值；horizon = 演示长度 P90 × 1.2（自动合成短任务 ×1.5）。真机每任务 10 trial，三名评分员双盲平均，horizon = P90 × 1.5。
- **榜单（2026-07-03 冻结）：** 仿真 Hy-Embodied-0.5-VLA 平均 13.07 分 / 8.80%，Spatial Forcing 8.04%，π0.5 6.91%；人类专家 80.42 / 76.03%。真机 π0.5 22.9 分 / 12.8% 居首；人类 100%。Open 维最佳仅 1.67%（π0.5）。
- **Finding：** 随机化场景使几乎所有策略崩塌（如 Hy-Embodied 标准 21.98 → 随机 1.57，降 92.9%；Spatial Forcing 降 67.2% 优于其 π0.5 基座 72.2%）；Precision 失败多为缺少状态条件纠错（如 insert_key 失败后仍继续开环执行）并伴动作抖动；Memory 维显式记忆不足以保证成功；真机排名与仿真仅部分一致，设计上不做任务配对的 sim-to-real 迁移测量。
- **效率：** 异构并行 77.4 vs 非异构 40.0 interactions/s（1.94×），含 π0.5 推理 64.0 vs 39.2（1.63×）；真机 18 任务 180 trial 约 202 分钟。
- **稳定性：** 三卡 RTX 4090 跨 GPU 仿真平均成功率标准差 ≤ 0.5 pp；真机三轮重复总成功率标准差 ≤ 1.3 pp，但单任务方差较大（如 store_in_safe 最高 23.1 pp）。
- **治理与验证：** 官方榜须经在线评测、仿真三 seed、真机三本体、隐藏布局校验（与公开布局成功率差异显著则判无效）、经 XPolicyLab 公开 checkpoint 与训推代码、公开评测视频；缺产物的结果只进独立的 closed-source 非官方轨道。
- **XPolicyLab 协议：** Model 抽象含 update_obs / get_action / reset，并行时 update_obs_batch / get_action_batch；WebSocket + MessagePack；位姿统一为 [x,y,z,qw,qx,qy,qz]；真机 EE 目标用 Pink 转为可执行运动。
- **论文自述的局限：** 真机只评了仿真榜的子集（10 策略）；真机策略仅训 1 个 seed；真机与仿真任务不一一对应；未来计划扩展灵巧手、人形全身、触觉、移动操作。
