# 9篇具身智能新作资源汇总：代码、数据与项目页一站式直达

> 来源归档（blog / 微信公众号）

- **标题：** 9篇具身智能新作资源汇总：代码、数据与项目页一站式直达
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/LOvIa6vyWVntc8_UPzHAkg
- **发表日期：** 2026-09-06
- **入库日期：** 2026-09-06
- **抓取方式：** WebFetch（Jina / 桌面 UA 对公众号常返回验证页）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_9_papers_resources_2026-09-06.md`](../raw/wechat_embodied_station_9_papers_resources_2026-09-06.md)
- **一句话说明：** 2026 开源系列续期，汇总 9 篇机器人与具身论文（评测、数据检索、VLA 鲁棒性、长程操作、3D WAM、接触 HRC 基准、人形急停、灵巧泛化、多视角 3D 跟踪）；**9/9 均有独立 `paper-*` 详情节点**（本 ingest **新建 3**、**复用 6**；同一 arXiv **不重复造页**）。

## 核心摘录（归纳，非全文）

文内判断：瓶颈正从「能不能完成任务」转向数据扩展、感知可靠、几何正确、接触安全与可信评测；开源资源从单仓扩展到权重、基准资产与交互式浏览器。

### 9 篇 → 本库节点

| # | 论文 | arXiv | 开源结论（入库日） | wiki |
|---|------|-------|-------------------|------|
| 01 | R2S-Eval | [2609.03276](https://arxiv.org/abs/2609.03276) | **待发布**：项目页无官方实现仓；基于 Isaac Lab-Arena | [paper-r2s-eval](../../wiki/entities/paper-r2s-eval.md) |
| 02 | RoboTok | [2609.03199](https://arxiv.org/abs/2609.03199) | **已开源** [Rice-RobotPI-Lab/RoboTok-Code](https://github.com/Rice-RobotPI-Lab/RoboTok-Code) | [paper-robotok](../../wiki/entities/paper-robotok.md) |
| 03 | EGR | [2609.03142](https://arxiv.org/abs/2609.03142) | **待发布**：仓 [YY-GX/EGR](https://github.com/YY-GX/EGR) README 写 Coming soon | [paper-egr](../../wiki/entities/paper-egr.md) |
| 04 | HINT | [2609.02653](https://arxiv.org/abs/2609.02653) | **待发布**（复用） | [paper-hint-robot-manipulation](../../wiki/entities/paper-hint-robot-manipulation.md) |
| 05 | SA-WAM | [2609.02531](https://arxiv.org/abs/2609.02531) | **待发布**（复用） | [paper-sa-wam](../../wiki/entities/paper-sa-wam.md) |
| 06 | Physics HRC Benchmark | [2609.02402](https://arxiv.org/abs/2609.02402) | **部分/待发布**（复用） | [paper-physics-consistent-hrc-benchmark](../../wiki/entities/paper-physics-consistent-hrc-benchmark.md) |
| 07 | Safe-Stop | [2609.02358](https://arxiv.org/abs/2609.02358) | **待发布**（复用） | [paper-safe-stop-humanoid](../../wiki/entities/paper-safe-stop-humanoid.md) |
| 08 | DemoMimic | [2609.01938](https://arxiv.org/abs/2609.01938) | **待发布**（复用 complete 页） | [paper-demomimic](../../wiki/entities/paper-demomimic.md) |
| 09 | TAPVid-MV | [2609.01899](https://arxiv.org/abs/2609.01899) | **部分开源**（复用） | [paper-tapvid-mv](../../wiki/entities/paper-tapvid-mv.md) |

## 对 wiki 的映射

- **9/9 独立详情节点**：每篇对应唯一 `wiki/entities/paper-*.md`；静态站 `detail.html?id=entity-paper-…` 均可直达。
- **本 ingest 新建 3**（R2S-Eval、RoboTok、EGR）；**6 复用**既有实体并回链本博客。
- 阅读坐标：[具身资源与可靠性 9 篇技术地图](../../wiki/overview/embodied-resources-reliability-9-papers-technology-map.md)。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 9 篇独立节点核查（3 新建 / 6 复用 / **0 重复 arXiv 节点**）
- [x] 项目页与仓库开源状态核查（步骤 2.5）
