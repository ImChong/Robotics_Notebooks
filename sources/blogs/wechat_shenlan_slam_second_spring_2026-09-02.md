# 深度｜具身智能时代：SLAM的“精华”与“糟粕”！

> 来源归档（blog / 微信公众号）

- **标题：** 深度｜具身智能时代：SLAM的“精华”与“糟粕”！
- **类型：** blog / salon 纪要
- **作者：** 深蓝具身智能（深蓝学院-具身君）
- **原始链接：** https://mp.weixin.qq.com/s/0MUtW7aaPPltT9oO3SUtSg
- **发表日期：** 2026-09-02
- **活动：** 2026-08-28 北京五道口「ROBO MIXER·具身学术夜」
- **入库日期：** 2026-09-04
- **抓取方式：** 直接拉取 `mp.weixin.qq.com` HTML 后去标签
- **原始抓取落盘：** [`sources/raw/wechat_shenlan_slam_second_spring_2026-09-02.md`](../raw/wechat_shenlan_slam_second_spring_2026-09-02.md)
- **一句话说明：** 沙龙纪要，不是论文盘点。嘉宾把 VLA naive scaling、仿真评测对齐与「SLAM 数学内核 vs 人为中间表示」拆开；文末声明不代表学院/嘉宾当前官方立场。
- **嘉宾：** 高翔（智身科技 /《视觉 SLAM 十四讲》）、史雪松（银河通用）、石成玉（众擎机器人）

## 核心摘录（归纳，非全文）

主题原设为「具身智能时代，SLAM 的第二春？」。现场把焦虑收成三条：

1. **更大是否更强：** 高翔称数据集扩大十倍后真机能力未显著超过一年前；转述一项 1228 篇 VLA 论文分析——诊断已从「缺数据」变成「naive scaling 失效」，精选 **5%** 子集可恢复全量 **85–90%** 性能。
2. **VLA 是行为克隆、没有 Planning：** A→B 用轨迹均值复读，遇到 C→D 只能靠覆盖；训练 loss 像人 ≠ 任务成功，仿真也难对齐打碎瓶子等后果。
3. **世界模型是大筐：** 视频预测、latent dynamics、仿真只要能预测都可以自称 WM；有人预期两年内仍会快长，也可能碰壁。

史雪松用世界人形机器人运动会**场景赛**当公开体检：餐饮/商超/家庭长程 + 外卖打断；要点不是单项 SR，而是自主性是否覆盖随机订单、缺货与任务切换。高翔后半段主张：SLAM 的精华是时空基准与可微几何约束，糟粕是给人看的工程中间图；长期地图可能是「SLAM Token」高维表示。呼应史雪松「机器人 ≠ 视觉理解 + Action 输出」。

## 对 wiki 的映射

- 写回：[具身时代 SLAM 第二春？](../../wiki/queries/slam-second-spring-embodied.md)
- 交叉：[VLA](../../wiki/methods/vla.md)、[状态估计枢纽](../../wiki/overview/hub-state-estimation.md)、[LiDAR SLAM 选型](../../wiki/comparisons/lidar-slam-lio-vio-selection.md)
- **本文不是论文列表**，不新建 `paper-*` 详情节点。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 沙龙论点提炼（非论文盘点）
