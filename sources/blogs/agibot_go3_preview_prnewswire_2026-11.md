# AGIBOT Unveils GO-3 Preview, Takes Top Spot on Three Embodied AI Benchmarks（新闻稿归档）

> 来源归档（ingest）

- **标题：** AGIBOT Unveils GO-3 Preview, Takes Top Spot on Three Embodied AI Benchmarks
- **类型：** 厂商新闻稿（PR Newswire APAC，署名 AGIBOT）
- **URL：** <https://www.prnewswire.com/apac/news-releases/agibot-unveils-go-3-preview-takes-top-spot-on-three-embodied-ai-benchmarks-302927277.html>
- **发布日期：** PR Newswire 页面标注 2026-11-03 11:28 CST（正文 dateline「SHANGHAI, Nov. 3, 2026」）。智元官网新闻中心同稿 <https://www.agibot.com/article/231/detail/133.html> 标注发布时间 **2026-10-29 17:17:42**（正文「October 29, 2026」）。两处日期不一致，本库以「官网首发 2026-10-29、PR Newswire 转发 2026-11-03」记录（推测：后者为海外分发）。
- **入库日期：** 2026-11-06
- **论文 / 技术报告：** 未找到（稿中未给 arXiv、技术报告链接）
- **核对范围**：2026-11-06 用 curl 取 PR Newswire 与智元官网两处全文，通读正文全部章节（摘要、WRAM/CoAE/SVP/KASO、Pre/Post/Post-Deployment Scaling、GO-1→GO-3 谱系结语），两处正文一致；图片与演示视频未逐帧核对；榜单数据另行核对（见下）。
- **通读：** 已完成
- **核查方式：** curl 全文、去标签阅读；榜单数据直接取自各榜单官方页面的公开 JSON / API（见下）。

## 新闻稿要点（作者表述）

- GO-3 Preview：支持预训练、后训练、部署后「全生命周期」学习的具身基础模型；称在 RoboDojo、RoboTwin、RoboColiseum 三个基准上排名第一；称将在 2026 年底前「fully open-sourced」。
- 架构 WRAM（World Reasoning Action Model）：统一任务理解、未来状态预测与动作生成。三项核心设计：
  - CoAE（Control-Oriented Autoencoder）：提取面向控制的视觉表征，保留物体位置、形状与空间关系。
  - SVP：基于当前观测与任务条件一步生成完整的未来状态预测；动作生成模块学习从当前状态走向预测结果。
  - KASO：针对物理环境多种可能结果，选择与训练中实际所执行动作更一致的未来预测，用于联合学习。
- 数据：人类第一人称视频 + 机器人轨迹。稿中未给数据规模、参数量、训练算力。
- 零样本 / 少样本：稿称零样本无需在实体机器人上做任务特定监督微调；LEGO 装配为少样本示例（少量示教 + 微调，由具身智能体编排步骤）。
- 后训练 Genie Evolver：分布式混合环境训练系统，100 台实体机器人 + 物理仿真 + 世界模型；实体侧用 CEL（Collective Experience Learning）聚合多机经验。演示：桌面冰壶（称几小时练习后胜率高于普通人类玩家）、抽积木、投篮到移动篮筐。均无量化数字。
- 部署后 Genie Master：具身智能体框架，提出 EEP（Embodied Execution Protocol），标准化四类接口：capabilities、guidance、feedback、continuity；经验通过「每日归并为长期记忆」与「轨迹与反馈用于再训练动作策略」两条路径回流。在便利店用 Genie G2 演示，称无需人类示教或手工调参即持续改进，并自称为真实商业环境中的 Physical RSI。
- 谱系叙述：GO-1 的 ViLLA 框架 → GO-2 的动作思维链 → GO-3 Preview。
- 稿中**没有**：任何榜单分数、名次数字、榜单 URL、评测口径、对手名单、模型规模、网络结构图、代码 / 权重 / 许可 / 论文链接。

## 榜单核对（2026-11-06 抓取官方公开数据）

### RoboTwin 2.0 Leaderboard（<https://robotwin-platform.github.io/leaderboard>）

- 数据 JSON：<https://robotwin-platform.github.io/data/robotwin_leaderboard.json>，页面 updated 2026-10-30。
- 口径：Aloha-AgileX，训练 50 条 demo_clean × 50 任务（共 2500 条），每任务 100 次试验，分别在 demo_clean（easy）与 demo_randomized（hard）评测；默认排序为 Average（c2c+c2r）。
- 条目 `GO-3 Nano Preview`（contributor AGIBOT，Co-train 赛道，listed_at 26.10.25，不需深度输入）：easy 88.54、hard 81.48，两者均值 85.01（我计算）。
- Co-train 赛道共 28 条：按 Average 排名第 1（第 2 PrismWAM* 91.84 / 75.50，均值 83.67；第 3 PICA-WAM 90.68 / 74.70，均值 82.69；第 4 CloudRobo-WM 88.90 / 75.00）；按 hard（clean2random）第 1；按 easy（clean2clean）第 6（PrismWAM* 91.84 最高）。
- 与稿件一致：Average 默认榜第一成立。榜单收录政策要求公开代码、权重与技术报告，而 GO-3 权重 / 报告我未找到，二者存在张力（我的观察，原因未知）。

### RoboDojo（<https://robodojo-benchmark.com>）

- 数据 JSON：<https://robodojo-benchmark.com/data/leaderboard.generated.json>，updatedAt 2026.11.03；含 sim（66 条）与 realworld（14 条）两榜；页面自述 42 个仿真任务、18 个真机任务。
- sim 榜条目 `GO-3 Nano Preview`（AGIBOT）：Average score 61.9（成功率 56.42%）；Avg. w/o Open 61.95；Generalization 36.45；Precision 78.65；Long-Horizon 68.21；Memory 64.5；Open 61.71。
- **与稿件不一致：** 抓取时 `Q0-Preview`（Qoo Tech）Average 74.2（成功率 68.87%）高于 GO-3 的 61.9，GO-3 按 Average 为第 2 / 66。GO-3 仅在 Precision 单项排名第 1（78.65，Q0-Preview 65.3）；Long-Horizon 第 2，Open 第 2，Generalization 第 6，Memory 第 6。JSON 数组顺序并非按分数排序（GO-3 排在数组首位），JSON 条目无逐行日期，Q0-Preview 何时上榜无法确定（推测：可能晚于新闻稿发布）。
- realworld 榜：未见 GO-3（榜首 SLA-Preview 42.33）。
- 因此「RoboDojo 第一」我只能确认：GO-3 在 RoboDojo 仿真榜曾 / 现位居前列且 Precision 单项第 1，Average 第 1 在抓取时点不成立。

### RoboColiseum（<https://robocoliseum.ai/leaderboard>，智元 Genie Sim 3.0 仿真挑战赛）

- 前端是 SPA，数据取自公开 API `https://robocoliseum.ai/api/challenge/leaderboard?board=<instruction|spatial|manip|robust>`；该 API 没有 overall 汇总榜（`metric=overall` 之类请求被拒绝，页面 UI 是否另有总榜未能核实）。
- 条目 `GO-3 Nano Preview`（AGIBOT，提交时间 2026-10-26）：
  - manip：0.688，第 1 / 37（第 2 Prime-0 0.6613，第 3 HiDream-O1-Embodied 0.6442）。
  - instruction：0.891，第 9 / 56（榜首 GPT-6 Astra 0.9935）。
  - spatial：0.7462，第 7 / 36（榜首 MXI SimWorld v0 Preview 0.9812）。
  - robust：0.8278，第 5 / 36（榜首 MXI SimWorld v0 Preview 0.9814）。
- 因此「RoboColiseum 第一」只在 manip 分榜成立，其余三个分榜不是第一。

## 发布 / 开源状态核查

| 资源 | 结果 | 方式 |
|------|------|------|
| 论文 / 技术报告 | 未找到 | 新闻稿正文无链接；WebSearch 与官网稿件无 arXiv 或报告链接 |
| GitHub | 未找到 | 官网稿件无链接；`github.com/AgibotTech/GO-3` 等猜测路径不可达（`git ls-remote` 要求认证）；已克隆 `AgibotTech/genie_sim`，仓内无 GO-3 / GO3 字样 |
| Hugging Face | 未找到 | HF API 列出 `agibot-world` 作者的模型仅 GO-1、GO-1-Air、Genie-Envisioner、EnerVerse-AC、EWMBench-model 等，无 GO-3 |
| 许可 | 未找到 | 无任何许可声明 |
| 开源计划 | 厂商表述 | 「will be fully open-sourced by the end of 2026」，无仓库、无日期细则 |
| 项目页 | 未找到 | 仅有官网新闻稿页 |

## 对 wiki 的映射

- 新建 [agibot-go-3-preview](../../wiki/entities/agibot-go-3-preview.md)：GO-3 Preview 单独实体（库内此前无 GO-3 节点，且与 GO-1、GO-2 是不同模型代际）。
- 关联既有节点：[GO-2](../../wiki/entities/go-2.md)、[AGIBOT WORLD / GO-1](../../wiki/entities/agibot-world-2026.md)、[RoboDojo](../../wiki/entities/robodojo.md)、[RoboTwin 2.0](../../wiki/entities/robotwin.md)、[RoboColiseum](../../wiki/entities/robocoliseum.md)、[Physical RSI](../../wiki/entities/physical-rsi.md)。
