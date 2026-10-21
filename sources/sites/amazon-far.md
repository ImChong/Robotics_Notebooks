# Amazon FAR：团队性质与公司技术路线

- **类型：** site / 企业研究团队官方资料
- **名称：** Amazon Frontier AI & Robotics（FAR）
- **官方技术入口：** <https://github.com/amazon-far>
- **官方研究门户：** <https://www.amazon.science/research-areas/robotics>（Amazon 整体机器人研究，非 FAR 专属首页）
- **官方团队说明：** <https://www.amazon.jobs/en/jobs/10422560/member-of-technical-staff-science-frontier-ai-robotics-far>
- **团队扩充事件：** <https://www.aboutamazon.com/news/company-news/amazon-covariant-ai-robots>
- **入库日期：** 2026-10-15
- **一句话说明：** Amazon 内部研究与工程团队，研究机器人基础模型、全身系统与真机部署；在公司路线中以企业团队身份收录。

## 组织性质与日期口径

GitHub 组织自述为 Amazon 的 Frontier AI & Robotics Org。Amazon 官方招聘页称其为 Amazon 的 FAR team，职位雇主为 Amazon.com Services LLC；据此按 Amazon 内部团队归类，不建立独立公司或独立研究所身份。

2024 年 Amazon 官方公告披露：引入 Pieter Abbeel、Peter Chen、Rocky Duan 以及部分 Covariant 研究工程人员，获得机器人基础模型的非独占许可，并计划扩充湾区 AI 与机器人团队。公告没有直接写出 FAR 名称的成立日期。路线的 `founded_year: 2024` 仅为该团队扩充事件的排序锚点，确切成立日期仍未确认；也不表示收购了 Covariant 整家公司。

## 技术路线与既有笔记映射

本次复用已有项目节点，不新增同名实体。以下月份沿用已收录论文的 arXiv v1 月份；代码公开时间、会议发表时间分别读取详情。

| 时间 | 项目 | 路线角色 | 既有笔记 |
| --- | --- | --- | --- |
| 2025-09 | OmniRetarget | 交互保留重定向、参考数据增广 | [OmniRetarget](../../wiki/entities/paper-hrl-stack-03-omniretarget.md) |
| 2025-10 | ResMimic | 通用运动跟踪先验上的残差全身操作 | [ResMimic](../../wiki/entities/paper-resmimic.md) |
| 2025-11 | TWIST2 | 便携全身遥操作、数采与 visuomotor 学习 | [TWIST2](../../wiki/entities/paper-twist2.md) |
| 2026-02 | RPL | 多视角深度与复杂地形行走 | [RPL](../../wiki/entities/paper-rpl-robust-humanoid-perceptive-locomotion.md) |
| 2026-02 | Perceptive Humanoid Parkour | 动作匹配、技能链与深度感知跑酷 | [PHP](../../wiki/entities/paper-hrl-stack-22-perceptive_humanoid_parkour.md) |
| 2026-06 | LadderMan | 深度感知攀梯、梯上遥操作 | [LadderMan](../../wiki/entities/paper-ladderman-humanoid-perceptive-ladder-climbing.md) |
| 未确认首发月份 | holosoma | 训练、推理与重定向框架 | [holosoma](../../wiki/entities/holosoma.md) |

## 开放范围

2026-10-15 核查 GitHub 组织有 holosoma、TWIST2 与 php_parkour 等公开仓库。可确认存在公开代码入口；不能从组织公开推断每个项目的训练、权重与数据全部开放。各项目范围沿用既有 [holosoma 归档](../repos/holosoma.md)、[TWIST2 归档](../repos/twist2.md)、[PHP 归档](../repos/php_parkour.md) 与论文详情注明日期的核查；本次未重新验证全部运行环境和资产。

## 对 wiki 的映射

- [公司路线对照](../../wiki/comparisons/robot-foundation-model-company-paths-2026.md)：新增 Amazon FAR 行、组织定位与来源。
- 公司路线展示数据：`docs/company-roadmaps.json`；首页入口：`docs/index.html`。
