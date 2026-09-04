# 2026 具身智能 Infra 全景总结：真正拉开差距的，已经不只是模型

> 来源归档（blog / 微信公众号）

- **标题：** 2026 具身智能 Infra 全景总结：真正拉开差距的，已经不只是模型
- **类型：** blog
- **作者：** Lumina具身智能 / Lumina社区
- **原始链接：** https://mp.weixin.qq.com/s/qVqpihnA4GezsE2MIJjKDw
- **发表日期：** 2026-09-03
- **入库日期：** 2026-09-04
- **抓取方式：** 直接拉取 `mp.weixin.qq.com` HTML 后去标签（Jina Reader 对公众号返回验证页）
- **原始抓取落盘：** [`sources/raw/wechat_lumina_embodied_infra_2026-09-03.md`](../raw/wechat_lumina_embodied_infra_2026-09-03.md)
- **一句话说明：** 把具身 Infra 读成「记录 → 学习 → 检查 → 复盘」闭环；竞争力看闭环周转时间，而不是单点模型分。文内 30 条参考文献**映射到已有独立 `paper-*` / 实体页**，缺失项只列待升格、**不重复造页**。

## 核心摘录（归纳，非全文）

作者用货架抓薯片失败当贯穿案例：失败应变成「哪台机器人、哪版程序、何时偏离、是否接管、是否恢复」的证词，而不是一段躺在硬盘里的视频。Infra 被压成四个动作——**记录、学习、检查、复盘**——对应数据、模型、评测、后训练；前一环输出是后一环输入。

文内判断：2026 年研究重心从「能演示」转向「系统持续运转」。公共层需要共享动作格式、质量标准、隐藏测试与失败分类；企业层沉淀脏污、抖动、磨损、排队、权限、回滚与安全停机等运营资产。分层交换：企业留原始客户数据，公开结构、匿名失败类型、评测套件与接口。

### 四环 → 本库节点（已有则复用）

| 环 | 文内代表工作 | arXiv | 本库节点（复用 / 待升格） |
|----|--------------|-------|--------------------------|
| 数据 | DROID | 2403.12945 | 既有 DROID / 操作数据页（检索 `droid`） |
| 数据 | COBALT | 2605.19138 | 待升格 |
| 数据 | HumanNet | 2605.06747 | [humannet](../../wiki/entities/humannet.md) |
| 数据 | AXIS | 2607.21588 | 待升格 |
| 数据 | MagicSim | 2606.17511 | 待升格（RoboDojo 源已互指） |
| 数据 | Data Pyramid | 2607.24744 | [paper-data-pyramid-embodied-manipulation](../../wiki/entities/paper-data-pyramid-embodied-manipulation.md) |
| 数据 | ACE-Data-0 | 2607.28625 | [paper-ace-data-0](../../wiki/entities/paper-ace-data-0.md) |
| 数据 | Xiaomi-Robotics-U0 | 2607.11643 | [xiaomi-robotics-u0](../../wiki/entities/xiaomi-robotics-u0.md) |
| 模型 | OpenVLA | 2406.09246 | [paper-openvla](../../wiki/entities/paper-openvla.md) |
| 模型 | LingBot-VLA（文称 Pragmatic VLA） | 2601.18692 | [lingbot-vla](../../wiki/entities/lingbot-vla.md) |
| 模型 | τ₀-VLA | 2608.16885 | [paper-tau0-vla](../../wiki/entities/paper-tau0-vla.md) |
| 模型 | Qwen-VLA | 2605.30280 | [qwen-vla](../../wiki/entities/qwen-vla.md) |
| 评测 | RoboTwin 2.0 | 2506.18088 | 既有 RoboTwin 实体（检索 `robotwin`） |
| 评测 | RoboSynChallenge | 2608.12416 | [paper-robosynchallenge](../../wiki/entities/paper-robosynchallenge.md) |
| 后训练 | DEED | 2607.20345 | [paper-deed](../../wiki/entities/paper-deed.md) |
| 后训练 | HELP / AtomVLA / LifeLong-RFT 等 | 2607.09776 / 2603.08519 / 2602.10503 | 待升格 |

文内关键数字（作者转述，以原论文为准）：AXIS 使 π₀.₅ 未见任务总体成功率 **+5.8%**；HumanNet 设定下 **1000 h** 第一人称视频持续训练超过 **100 h** 真机；Xiaomi-U0 将 π₀.₅ 陌生环境成功率 **36.9%→63.2%**；DEED 把实验室到商店的主障碍写成系统集成而非再发明网络。

## 对 wiki 的映射

- 阅读坐标：[具身 Infra 2026 全景](../../wiki/overview/embodied-infra-2026-panorama.md)（**非**论文详情替代）
- 交叉：[VLA](../../wiki/methods/vla.md)、[Data Pyramid](../../wiki/entities/paper-data-pyramid-embodied-manipulation.md)、[DEED](../../wiki/entities/paper-deed.md)
- **本次未**把文内 30 篇全部新建实体：已有页只回链；缺页标「待升格」。用户另行指定的 WM-LOCO / Safe-Stop / FOCUS **另有独立节点**，与本文参考文献集合不相交。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 30 条参考文献对照既有 wiki（复用 / 待升格，**0 重复造页**）
