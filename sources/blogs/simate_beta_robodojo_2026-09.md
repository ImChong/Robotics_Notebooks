# Simate-beta 首版模型与 RoboDojo 上榜（媒体报道 + 榜单数据）

> 来源归档

- **主题：** Simate（硅基伙伴，Silicon Mate）首款模型 Simate-beta、AutoResearch 平台与公司背景
- **类型：** 媒体报道（公司披露口径）+ 公开榜单数据
- **核查日期：** 2026-10-09
- **沉淀到 wiki：** [`wiki/entities/simate-beta.md`](../../wiki/entities/simate-beta.md)、[`wiki/entities/simate.md`](../../wiki/entities/simate.md)

## 来源

| 来源 | 日期 | 链接 |
| --- | --- | --- |
| 新智元《力压 GPT-6，中国物理 AI 黑马登顶第一》（36氪转载） | 2026-09-24 | <https://eu.36kr.com/zh/p/3996798461628297> |
| 量子位《AI 开始研究 Physical AI：FSD 级团队亮出首版模型 Simate-beta，空降 RoboDojo》（36氪转载） | 2026-09-28 | <https://eu.36kr.com/zh/p/3999916051157129> |
| RoboDojo 榜单前端数据（News + sim leaderboard 表） | 2026-09-23 条目 | <https://robodojo-benchmark.com/leaderboard> |
| Simate 官网（演示视频托管于 `mate-robot.cn`） | 无日期 | <https://mate-robot.cn/home/>、<https://simate.ai/> |

## 要点（均为公司披露或媒体转述，除榜单数字外未独立核实）

- **公司：** Simate（Silicon Mate，中文「硅基伙伴」）。量子位称「成立仅三个月」（按 2026-09 报道倒推约 2026-06 成立，推测）。创始人兼 CEO 张颖，此前为国内头部自动驾驶公司一段式端到端智驾技术负责人之一；团队还包括港科大助理教授占方能、前 ARI（后被 Meta 收购）创始成员季马泽宇（新智元）。
- **融资：** 两篇报道均称已连续完成多轮「数亿元人民币」级融资，未披露轮次与投资方。
- **Simate-beta：** 定位「通用物理快系统」（System 1），把快系统参数做大以探索零样本泛化；技术抓手为 **4D 物理感知** 与 **分层时序记忆**；具体架构与参数规模「将在后续技术报告中披露」。真机展示围绕演示驱动的任务适应、记忆、复杂长程执行与精细操作。团队称参评模型未针对 RoboDojo 做专门优化。
- **RoboDojo（仿真榜）：** 榜单 News 2026-09-23「Add Simate-beta to the sim leaderboard (33.95 / 27.96%, contributed by Simate)」。分项（Score / SR）：generalization-std 40.54 / 33.22，generalization-rand 29.63 / 22.67，precision 34.35 / 26.92，long-horizon 57.84 / 43.42，memory 33.33 / 33，open 9.12 / 8.5，average 33.95 / 27.96。
- **榜单名次随时间变化：** 上榜当日为第一（公司披露）；2026-09-28 HKU MMLab × Kinetix 的 Physical RSI 1.0 以 Score 36 / SR 31% 列 Overall 第一（见 [Physical RSI 归档](../sites/mmlab-physical-rsi.md)），二者为不同机构。
- **AutoResearch：** 人类研究员提出假设、设定目标与约束，引擎自动拆解实验、执行与回传结果；公司称 MIT、加州理工、清华、北大等高校研究者参与内测。与 Sinfra（训练 / 仿真 / 推理基础设施）、Sipai（可插拔模型框架）共同构成研发体系；公司把长期路线称为「Physical RSI」（与港大 MMLab 的同名项目无关）。
- **后续计划：** 年底推出面向复杂任务零样本泛化的阶段性成果；模型与自动化研究等工作将通过论文与技术报告陆续公布，并「分阶段开源」。

## 开源核查（2026-10-09）

| 资源 | 状态 |
| --- | --- |
| Simate-beta 代码 / 权重 | 未公开；无技术报告 |
| AutoResearch / Sinfra | 内测 / 申请制，未开源 |
| 数据 | 未公开 |
| 榜单分项 | RoboDojo 前端公开 |
