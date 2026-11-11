# CoRL 2026 官方会议议程、主题演讲与论文检索入口（Program）

> 来源归档（site）。本条升级已有归档，**不创建第二份同 URL source**；这是一份会议导航资料，并非某篇论文的技术报告或源码。

- **标题：** Conference on Robot Learning 2026 — Program
- **类型：** site（官方会议日程、主题演讲、论文与报告导航）
- **URL：** <https://2026.corl.org/program>
- **大会首页：** <https://2026.corl.org/>
- **Keynotes：** <https://2026.corl.org/program/keynotes>
- **Accepted Papers：** <https://2026.corl.org/program/accepted-papers>
- **OpenReview 会议入口：** <https://openreview.net/group?id=robot-learning.org/CoRL/2026/Conference>
- **报告说明：** <https://2026.corl.org/contributions/instruction-for-presentations>
- **关联论文：** <https://arxiv.org/abs/2610.28480>（SuperMimic，CoRL 2026 Oral；论文方法须独立核验）
- **入库/复核日期：** 2026-11-11
- **核对范围：** 已阅读 Program 页可公开读取的全部正文、会议首页、Keynotes 页三场讲座的官方题目与摘要、Instruction for Presentations 页报告规则；打开 Accepted Papers 页但该页在当前文本抓取方式下未呈现论文列表。**未**访问 Cvent App 的登录/动态日程数据、未下载全部论文、未核实逐场直播/回放的独立视频链接。
- **通读：** 已完成
- **一句话说明：** CoRL 2026 官方 Program 是机器人学习会议日程与现场报告的一级导航；研究论文证据应追溯到相应作者原文和 OpenReview，而非仅依赖会议日程摘要。

## 会议时间、地点与会场

- **日期：** 2026 年 11 月 9–12 日，Austin, Texas, USA；主会 11 月 9–11 日，Workshops 11 月 12 日。
- **JW Marriott Austin：** poster、exhibits、午餐及休息；官方 Program 的 badge pickup 也设在 Marriott 二楼 Room 201。
- **Hilton Austin：** 前两天的报告与 keynote；Program 声明 Hilton Stage 的内容向 Marriott viewing rooms 转播。
- **Bass Hall：** reception 与第三天 keynote。以上为大会首页公布的场地分工，不能外推某篇论文的确切讲台、时间或录像链接。
- **动态个人日程：** 官方建议使用 Cvent App，查找“CoRL Event”；本归档仅确认此导航说明，不声称已经获取 App 内完整时段表。

## 官方发布状态与论文检索

| 官方来源 | 已核实信息 | 使用边界 |
|---|---|---|
| [大会首页](https://2026.corl.org/) | 2026-10-19 公告 accepted papers list available；2026-11-09 公告 “Live streams posted on program!” | **公告存在**不等于任何特定论文的录像已经公开或可独立直链 |
| [Program](https://2026.corl.org/program) | 主会/Workshop 日期、转播与现场后勤说明 | 目前可读取的静态文字**未给出完整逐场论文时刻表** |
| [Accepted Papers](https://2026.corl.org/program/accepted-papers) | 大会首页有官方入口 | 本次抓取未取得可核验论文条目；不据此编造 Oral、Spotlight、Poster 的逐篇归属 |
| [OpenReview](https://openreview.net/group?id=robot-learning.org/CoRL/2026/Conference) | 大会投稿说明指向该会议分组，并说明录用论文及评审公开 | 各论文标题、作者、技术细节、评测结论须在对应条目/原文中逐一核对 |
| [报告说明](https://2026.corl.org/contributions/instruction-for-presentations) | 所有录用论文参加 poster；72 篇 spotlight 各有最多 3 分钟报告；oral 每篇 8 分钟报告 + 2 分钟问答 | 这里的 **72 是 spotlight 数量**；不能据此推算全部接收数或具体安排 |

## Keynotes：官方题目与研究视角

以下均来自 [CoRL 2026 官方 Keynotes 页面](https://2026.corl.org/program/keynotes)，是演讲摘要而不是论文结果；仅概括演讲定位，不转述为经验证的算法性能。

| 演讲者 | 官方题目 | 关注问题 |
|---|---|---|
| Wolfram Burgard | *Challenges to Realize Embodied AI* | 不完美感知/执行带来的不确定性；概率机器人与深度学习，以及基础模型处理真实环境变化的潜力 |
| Russ Tedrake | *Learning in the factories* | 在真实生产环境中部署机器人，如何反过来推动操作大模型持续学习；摘要介绍 Walden Robotics 的工厂实践 |
| Fei-Fei Li | *Atlas: A Unified World Model for Generation and Reconstruction* | 世界模型统一图像/帧生成与三维场景重建；报告基于 World Labs Atlas 研究 |
| Lisa Su × Fei-Fei Li | *Fireside Chat* | 官方 Keynotes 页面注明两人的炉边对谈；未将其归为第四篇研究报告 |

**与机器人研究主线的联系（归纳，不是主办方排名）：**
1. **感知/控制的鲁棒性：** Burgard 对真实传感器与执行器误差的强调，可以作为 Sim2Real 与不确定性建模的研究问题入口。
2. **部署驱动的数据闭环：** Tedrake 的演讲把学习问题连接到工厂上线、持续评估与新数据回流。
3. **三维世界表示：** Fei-Fei Li 的演讲聚焦生成式世界模型与重建，但官方摘要没有给出 benchmark 结果，不能自行填入数字。

## 如何从议程检索并 ingest 单篇论文

1. **识别会议中的项目：** 从 Program / Cvent 日程或 Accepted Papers 找到论文原始标题与作者（无法核实时保留“待核实”）。
2. **确认论文身份：** 进入对应 [OpenReview 会议分组](https://openreview.net/group?id=robot-learning.org/CoRL/2026/Conference) 与作者的 arXiv/项目主页，核实会议接收类型、版本及归属，避免同名项目误合并。
3. **通读一手原文：** 阅读论文方法、实验、消融、局限和必要附录；检查项目主页中的 Code/Data 状态。只有正式得到数据才记具体数字。
4. **映射既有实体：** 先搜 Robotics_Notebooks 既有 `project_id`、arXiv、官网与 repo；重复项目只补充现有 wiki，源码有可运行入口时另核对 Mermaid 源码时序图。
5. **区分视频与论文：** 会议总直播、单场 oral、spotlight 录制与论文方法互不等价；未取得可验证单场 URL 前，只记录 Program 总入口。

```text
CoRL Program / Accepted Papers（会议导航）
    -> OpenReview / 作者原文（论文身份与证据）
    -> 项目页 / 官方源码（可复现性核验）
    -> Robotics_Notebooks 已有节点去重 -> 更新 wiki 与 sources
```

## 已知限制与后续待办

- Program 页面存在转播说明，首页宣布发布直播；**本次仍未定位到可确认属于 SuperMimic 或其他单篇论文的独立录像 URL**。不以首页公告替代视频核验。
- Accepted Papers 页面在本次文本读取中未返回动态论文表，故**未**构建完整论文名单、没有擅自填入口头报告/时间/海报编号。
- 对报告内容的总结仅基于官方 Keynotes 摘要，未看到会后演讲全文或视频，不能当作演讲者完整观点。
- 后续若取得官方会议导出、OpenReview 完整 accepted 列表或视频页面，应重新逐条核对与去重后分批 ingest，不把第三方整理的汇总数字当作官方事实。

## 对 wiki 的映射

- [SuperMimic 论文实体](../../wiki/entities/paper-supermimic.md) — 已存在的 CoRL 2026 Oral 论文节点；继续复用此 source 作为官方会议导航，算法与数字以论文原文为准。
- [SuperMimic 论文原始资料](../papers/supermimic_arxiv_2610_28480.md) — 方法、实验与局限的实证依据。
- **本次未新建会议“项目实体”或大量论文占位页：** 此 URL 本质是活动导航，仍留在 `sources/sites/`；真正有足够原创技术证据的论文应逐篇归并至其已有主实体。
