# 刷完今年 Github 上所有 VLA 项目之后，最推荐复现这几个……

> 来源归档（blog / 微信公众号）

- **标题：** 刷完今年Github上所有VLA项目之后，最推荐复现这几个……
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/k_i-1NEBP-lEzth19HOHkQ
- **发表日期：** 2025-12-23（frontmatter）；文内统计截至 2025-12-22
- **入库日期：** 2026-05-22
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0 + `wechat-article-for-ai`（Camoufox）；正文约 1.4 万字 / 28 图
- **姊妹篇：** [极具「影响力」的12个VLA开源项目](https://mp.weixin.qq.com/s?__biz=MzkwMDcyNDUzMQ==&mid=2247494473&idx=1&sn=28c95bea437f22cc8e9ed7ca3308a071)（文内未逐条复述，本篇为「复现向」策展）
- **一句话说明：** 以 GitHub star>400、代码可获取与活跃度为筛选，策展 2025 年 11 个高可见 VLA 相关开源栈（通用策略、轻量 VLA、RL 训练系统、跨本体、VLA+世界模型、空间/驾驶/灵巧抓取、推理-执行解耦），强调「可跑通代码」重于论文叙事。

## 筛选标准（作者）

- VLA 开源项目快速增长；本文 **非穷尽**，以 **star > 400**、**代码可获取**、**项目活跃** 为主。
- 统计时间：**2025-12-22**（star 数为文内快照，入库时不复写易过期数字）。

## 项目索引（文内编号 01–12，缺 02）

| # | 项目 | 机构/团队 | GitHub（文内） | 定位摘要 |
|---|------|-----------|----------------|----------|
| 01 | **OpenPI** | Physical Intelligence | `Physical-Intelligence/openpi` | π0 / π0-FAST / π0.5；VLM + flow matching 动作；多平台真机数据 |
| 03 | **VLA-Adapter** | 北邮、西湖、浙大、OpenHelix 等 | `OpenHelix-Team/VLA-Adapter` | ~0.5B 轻量主干；Bridge Attention 注入 VL 条件；低预训练数据依赖 |
| 04 | **RLinf** | 清华、北大等 | `RLinf/RLinf` | 具身/推理 RL **训练系统**；弹性流水线、自适应通信；π-RL 底座 |
| 05 | **SimpleVLA-RL** | 清华、上交等 | `PRIME-RL/SimpleVLA-RL` | 基于 veRL 的 VLA RL；OpenVLA-OFT 微调；长时序与动作空间探索现象 |
| 06 | **UniVLA** | 香港大学 | `OpenDriveLab/UniVLA` | 视频潜动作表示；跨平台迁移；轻量解码到具体机器人 |
| 07 | **RynnVLA-002** | 阿里达摩院 | `alibaba-damo-academy/RynnVLA-002` | 自回归 **动作-世界模型** 统一；LeRobot SO100 真机数据 |
| 08 | **StarVLA** | — | `starVLA/starVLA` | 模块化 VLM→VLA 工程框架；单卡可训 |
| 09 | **SpatialVLA** | 上海 AI Lab、浙大、上交等 | `SpatialVLA/SpatialVLA` | Ego3D 位置编码；>110 万真机轨迹预训练 |
| 10 | **OpenDriveVLA** | 慕尼黑工大 | `DriveVLA/OpenDriveVLA` | 端到端 **自动驾驶** VLA；2D/3D 实例 + 分层 VL 对齐 |
| 11 | **DexGraspVLA** | 北大等 | `Psi-Robot/DexGraspVLA` | 分层：VLM 规划 + 扩散低层；灵巧抓取零样本泛化叙事 |
| 12 | **DeepThinkVLA** | 华中科大等 | `wadeKeith/DeepThinkVLA` | 推理（因果注意力）与动作（双向注意力）**混合解码** |

> 文内章节编号从 01 直跳 03，**无第 02 项**（抓取版如此，不补臆测条目）。

## 按「复现目标」分组（归纳）

| 你想复现什么 | 优先看 |
|-------------|--------|
| 通用操作策略 + π 系 | OpenPI → 本站 [π0 Policy](../../wiki/methods/π0-policy.md)、[π0.7](../../wiki/methods/pi07-policy.md) |
| 低算力 / 小模型 VLA | VLA-Adapter |
| VLA 的 RL 后训练 | SimpleVLA-RL；系统层见 RLinf |
| 跨机器人形态 | UniVLA |
| VLA + 世界模型一体 | RynnVLA-002 → [世界模型训练闭环](../../wiki/overview/robot-world-models-training-loop-taxonomy.md) |
| 工程脚手架、换 backbone | StarVLA → [StarVLA](../../wiki/methods/star-vla.md) |
| 空间几何强化 | SpatialVLA |
| 自动驾驶 VLA | OpenDriveVLA |
| 灵巧抓取长程 | DexGraspVLA |
| 语言推理链 + 并行动作 | DeepThinkVLA |

## 对 wiki 的映射

- [vla-open-source-repro-landscape-2025](../../wiki/overview/vla-open-source-repro-landscape-2025.md)（本次升格主页面）
- [vla](../../wiki/methods/vla.md)、[star-vla](../../wiki/methods/star-vla.md)、[manipulation-vla-architecture-selection](../../wiki/queries/manipulation-vla-architecture-selection.md)

## 可信度与使用边界

- star 数、榜单名次会随时间变化；复现前以各仓库 README / issue 活跃度为准。
- 部分项目文内仅给概述，未覆盖训练细节与 License；不构成选型唯一依据。
- 微信 CDN 图不入库；外链 GitHub 为一手代码入口。

## 当前提炼状态

- [x] Agent Reach 正文抓取与 11 项索引
- [x] 复现向分组与 wiki 主页面映射
