# 具身智能机器人最新开源论文速览

> 来源归档（blog / 微信公众号）

- **标题：** 具身智能机器人最新开源论文速览
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/-UqboKHaoG5eu79u9XQU0w
- **发表日期：** 2026-09-03
- **入库日期：** 2026-09-03
- **抓取方式：** Agent Reach + `wechat-article-for-ai`（Camoufox）；`--no-images`
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_8_papers_open_source_2026-09-03.md`](../raw/wechat_embodied_station_8_papers_open_source_2026-09-03.md)
- **一句话说明：** 2026 年新系列首期，汇总 8 篇近期机器人与具身开源论文，主线为长视野意图保持、3D-aware WAM、接触丰富 HRC 基准、人形安全停止、灵巧手单示范泛化、多视角 3D 点跟踪、测试时智能综述与多视角证据融合；**8/8 均有独立 `paper-*` 详情节点**（本 ingest **新建 7**、**复用 DemoMimic 既有 complete**；同一 arXiv **不重复造页**）。

## 核心摘录（归纳，非全文）

文内判断：具身智能正从「看懂-出动作」单步范式转向系统闭环——意图长期保持、世界模型纳入 3D 几何、接触与安全被物理一致评估、部署期用反馈与证据来源管理风险。

### 8 篇 → 本库节点

| # | 论文 | arXiv | 开源结论（入库日） | wiki |
|---|------|-------|-------------------|------|
| 01 | HINT | [2609.02653](https://arxiv.org/abs/2609.02653) | **待发布**：项目页无 GitHub | [paper-hint-robot-manipulation](../../wiki/entities/paper-hint-robot-manipulation.md) |
| 02 | SA-WAM | [2609.02531](https://arxiv.org/abs/2609.02531) | **待发布**：项目页无代码仓 | [paper-sa-wam](../../wiki/entities/paper-sa-wam.md) |
| 03 | Physics-Consistent HRC Benchmark | [2609.02402](https://arxiv.org/abs/2609.02402) | **部分/待发布**：匿名预览仓，benchmark 实现 Coming soon | [paper-physics-consistent-hrc-benchmark](../../wiki/entities/paper-physics-consistent-hrc-benchmark.md) |
| 04 | Safe-Stop | [2609.02358](https://arxiv.org/abs/2609.02358) | **待发布**：项目页无 GitHub | [paper-safe-stop-humanoid](../../wiki/entities/paper-safe-stop-humanoid.md) |
| 05 | DemoMimic | [2609.01938](https://arxiv.org/abs/2609.01938) | **待发布**（复用既有页；项目页 Code coming soon） | [paper-demomimic](../../wiki/entities/paper-demomimic.md) |
| 06 | TAPVid-MV | [2609.01899](https://arxiv.org/abs/2609.01899) | **部分开源**：基准/评测/Perpetua 生成器经项目页发布；无独立 GitHub | [paper-tapvid-mv](../../wiki/entities/paper-tapvid-mv.md) |
| 07 | TTI Survey | [2609.01679](https://arxiv.org/abs/2609.01679) | **已开源** `mr-eggplant/awesome_test_time_intelligence` | [paper-test-time-intelligence-survey](../../wiki/entities/paper-test-time-intelligence-survey.md) |
| 08 | PACT | [2609.01662](https://arxiv.org/abs/2609.01662) | **已开源** `ZekaiJ/PACT` | [paper-pact-hrc-action-admission](../../wiki/entities/paper-pact-hrc-action-admission.md) |

### 文内要点速记

1. **HINT** — 模式切换处稀疏语义推理 + 多视角 grounding/跟踪；Wall-OSS-0.5 与 π₀.₅ 上提升长视野成功率。
2. **SA-WAM** — RGB-D 几何注入 WAM 扩散骨干；RoboCasa / LIBERO-Plus SOTA；UR5 真机随机环境增益明显。
3. **HRC Benchmark** — 辅助洗浴接触丰富任务；任务成功经区域/力安全筛查后显著下降。
4. **Safe-Stop** — reach-avoid 可停止性估计 + 阻尼 fallback；G1 OOD 停止成功率 96.4%。
5. **DemoMimic** — 局部接触几何 + AR/SCR；16 物体真机 71%（既有实体页）。
6. **TAPVid-MV** — 284 序列多视角 3D 点跟踪基准；30+ baseline 均未接近解决。
7. **TTI Survey** — 统一 test-time adaptation / learning / scaling 视角。
8. **PACT** — provenance-conserving 多视角融合；区分计算重复与证据独立来源。

## 对 wiki 的映射

- **8/8 独立详情节点**：每篇对应唯一 `wiki/entities/paper-*.md`；静态站 `detail.html?id=entity-paper-…` 均可直达。
- **本 ingest 新建 7** 个实体；**DemoMimic** 先前 ingest 已有 complete 页 → **只回链博客与 arXiv，不重复造页**。
- 阅读坐标：[开源系统可靠性 8 篇技术地图](../../wiki/overview/open-source-system-reliability-8-papers-technology-map.md)（**非**论文详情替代，仅作横切面索引）。
- 交叉：[VLA](../../wiki/methods/vla.md)、[World Action Models](../../wiki/concepts/world-action-models.md)、[Manipulation](../../wiki/tasks/manipulation.md)、[Humanoid](../../wiki/tasks/humanoid.md)。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 8 篇独立节点核查（7 新建 / 1 复用 / **0 重复 arXiv 节点**）
- [x] 项目页与仓库开源状态核查（步骤 2.5）
