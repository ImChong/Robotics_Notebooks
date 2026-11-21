# X2Real 项目页

- **类型：** site / 官方项目页
- **链接：** <https://x2robot.com/en/pages/x2real>
- **关联论文：** <https://arxiv.org/abs/2609.27449>
- **代码仓库：** <https://github.com/X-Square-Robot/x2real>
- **项目方：** X Square Robot（自变量机器人）
- **再核日期：** 2026-11-21
- **核对范围：** 解析 `https://x2robot.com/en/pages/x2real` Next.js RSC 负载全文（指标卡、44 项任务分类、10 维度说明、3 轴 OOD 规模、≈300 h 各任务小时数与 episode 分布、4 模型桌面排行榜）并复核 `X-Square-Robot/x2real` 仓库已上线
- **通读：** 已完成
- **一句话说明：** X2Real 项目入口，围绕通才机器人操作策略的仿真评测、任务体系、Mana 平台与 sim-to-real 对照验证。

## 核验说明

- 2026-11-21 重新抓取 `https://x2robot.com/en/pages/x2real`（Next.js 前端渲染，正文位于 RSC 负载中），确认核心公开口径：10 个能力维度、44 项层级长程任务、Sim2Real 线性相关 **0.84**、约 **300 h** 示教数据；三轴 OOD 覆盖 **41 个房间、92 种桌面材质、89 个 HDRI**。
- 页面 Leaderboard（Tabletop 4 模型 OOD SR / Score 排序）：`wall-oss-preview`（ID 67.25 / 53.9% → OOD 56.63 / 34.3%）、`PI0.5`（46.19 / 26.1% → 37.75 / 16.1%）、`DreamZero`（34.21 / 4.9% → 32.12 / 3.5%）、`Wall-OSS-0.5`（17.85 / 4.1% → 14.10 / 3.2%）。
- **代码仓更新（2026-10-10）：** 早期核查时 `github.com/X-Square-Robot/x2real` 尚不可访问（404）；2026-10-10 官方发布了 X2Real v0.9.0（Apache-2.0），开放了 ID 评测运行时与手动采数入口（详见 [x2real.md](../repos/x2real.md)），OOD 评测定义与自动数据合成管线仍未公开。

## 来源与关联详情

- [X2Real 论文归档](../papers/x2real_arxiv_2609_27449.md)
- [X2Real 源码归档](../repos/x2real.md)
- [X2Real Simulation Assets 数据集](../datasets/x2real-assets.md)
- [X2Real 论文详情](../../wiki/entities/paper-x2real.md)
- [X2Real 项目详情](../../wiki/entities/x2real-project.md)
