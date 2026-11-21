# 解读｜聊一聊最近看到的灵巧手 System 0 的新变量

> 二手报道 / 访谈线索归档。本文仅保存核验要点与知识映射，尊重原作者版权，不转载全文。

- **作者：** 具身纪元
- **发布时间：** 2026-11-21 10:00（文章所标）
- **原文：** <https://mp.weixin.qq.com/s/oLcRb5Y5bnjObGDt6Lk-iw>
- **类型：** 微信公众号访谈 / 行业技术解读
- **核对范围：** 已通读用户提供的文章全文（引言、01–05 节及三项演示）；微信公众号原站链接在当前读取环境未能直接打开，以下访谈数字以提供的完整正文为据，非独立复核的实验数据。
- **通读：** 已完成
- **主题：** 从 CoRL Amazon Pick Anything / TeleDexter 到 ManuPrax Robotics 的 ARC-0 Preview；System 2→System 0 接口、手物接触 RL、对象泛化。

## 核心摘录（忠实于报道，不等同官方复现实验）

1. **任务边界：** 上层负责场景理解/任务拆解/物体目标；System 0 专注物体到手后持续变化的接触、摩擦、换指和发力。
2. **数据路线：** 人手与物体协同参考运动经重定向形成子目标；仿真 RL 负责探索机器人自己的接触策略。报道认为 off-embodiment 位姿参考不直接提供机器人所需力与执行动作。
3. **TeleDexter → ARC-0：** 前者主要是物体专用 co-tracking 遥操作小脑；报道称 ARC-0 增加跨同类未见实例泛化，并接入语言高层任务规划。
4. **采访给出的量化口径：** 320 个训练参考物体、80 小时参考数据，连续至少 50 个位姿目标成功率 46.6%；报道提及 K=10/30/50 曲线。**未从 ManuPrax 官方公开的短公告独立核验数值与统计协议**，勿和 TeleDexter 的 75.2% 七任务成功率横比。
5. **采集工程：** play data 多样性优先；高精度动捕用于长尾小物体，4–5 台 RGB-D 的固定工位用于较明显/较大物体；重视训练重置、回放筛选、难度与奖励平衡，而非只扩网络参数。均为采访陈述。
6. **层间接口：** 文章称 System 2 提供目标物体的 6-DoF 位姿轨迹，System 0 协调指头、接触与发力；物体形变、视觉与触觉接入仍是未来挑战。
7. **演示：** 换灯泡、桌面整理杯具、用刀切黄瓜；成功案例不等于开放测试集上的泛化评测。
8. **训练与开放性：** 文章未提供 ARC-0 可运行代码、权重或训练数据集的正式下载入口；此项需依官方后续发布核验。

## 对 Wiki 的映射

- [ARC-0 Preview 项目实体](../../wiki/entities/manuprax-arc0.md) — 可确认的分层系统、参考运动与未解边界。
- [TeleDexter 论文实体](../../wiki/entities/paper-teledexter.md) — 论文级验证的 co-tracking 与连续子目标。
- [遥操作纵深 Stage 4](../../roadmap/depth-teleoperation.md) — 从人手参考到低层控制器。
- [接触丰富操作纵深](../../roadmap/depth-contact-manipulation.md) — 接触控制与 Sim2Real。

## 一手核对入口

- [ManuPrax 官方 ARC-0 Preview 发布公告](https://x.com/ManuPraxRobo/status/2122567192994836693)
- [TeleDexter 官方项目页](https://bigai-dex.github.io/blog/teledexter/)
- [TeleDexter 原论文](https://arxiv.org/abs/2607.11481)
