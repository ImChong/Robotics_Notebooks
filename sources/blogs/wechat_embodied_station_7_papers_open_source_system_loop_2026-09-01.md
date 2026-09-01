# 机器人圈开源速递：LightNav-0、Zeva、Motus2 等 7 篇新作

> 来源归档（blog / 微信公众号）

- **标题：** 机器人圈开源速递：LightNav-0、Zeva、Motus2等7篇新作
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/IkK6lFCu4hjBX0sA1hMqgA
- **发表日期：** 2026-09-01
- **入库日期：** 2026-09-01
- **抓取方式：** Agent Reach + `wechat-article-for-ai`（Camoufox）；`--no-images`
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_7_papers_open_source_system_loop_2026-09-01.md`](../raw/wechat_embodied_station_7_papers_open_source_system_loop_2026-09-01.md)
- **一句话说明：** 汇总 7 篇近期机器人与具身论文，主线为语言驱动 QD 技能档案、VLM 通用导航、在线因果记忆、自演化世界模型、VLA 推理期纠错、多智能体编排与光学挑战场景单目几何；**7/7 均有独立 `paper-*` 详情节点**（本 ingest **新建 6**、**复用 Motus2 既有 complete**；同一 arXiv **不重复造页**）。

## 核心摘录（归纳，非全文）

文内判断：具身智能正进入「**开源入口 + 系统闭环**」阶段——从自然语言生成技能档案，到 VLM 导航、因果记忆、世界模型自演化，再到无需训练的语言纠错和复杂场景几何感知，机器人能力更强调可调用、可验证、可迁移。

### 7 篇 → 本库节点

| # | 论文 | arXiv | 开源结论（入库日） | wiki |
|---|------|-------|-------------------|------|
| 01 | Language-driven QD | [2608.30983](https://arxiv.org/abs/2608.30983) | **已开源** `EGarrabe/Language-driven-robotic-QD` | [paper-language-driven-robotic-qd](../../wiki/entities/paper-language-driven-robotic-qd.md) |
| 02 | LightNav-0 | [2608.30935](https://arxiv.org/abs/2608.30935) | **已开源** `lightorigins/LightNav-0` + HF 权重 | [paper-lightnav-0](../../wiki/entities/paper-lightnav-0.md) |
| 03 | Zeva | [2608.30880](https://arxiv.org/abs/2608.30880) | **已开源** `air-embodied-brain/Zeva` + HF 模型 | [paper-zeva](../../wiki/entities/paper-zeva.md) |
| 04 | Motus2 | [2608.30237](https://arxiv.org/abs/2608.30237) | **未开源**（复用既有页；项目页无代码仓） | [paper-motus2](../../wiki/entities/paper-motus2.md) |
| 05 | CorrectVLA | [2608.29967](https://arxiv.org/abs/2608.29967) | **已开源** `owenk3/correct_vla` | [paper-correctvla](../../wiki/entities/paper-correctvla.md) |
| 06 | EMERGE-Policy | [2608.29896](https://arxiv.org/abs/2608.29896) | **已开源** `EMERGE-Policy/EMERGE-Policy` | [paper-emerge-policy](../../wiki/entities/paper-emerge-policy.md) |
| 07 | OptiGeo | [2608.29881](https://arxiv.org/abs/2608.29881) | **已开源** `mx-liu6/OptiGeo` + HF 权重 | [paper-optigeo](../../wiki/entities/paper-optigeo.md) |

### 文内要点速记

1. **Language-driven QD** — 自由形式任务语言自动探索 fitness/BD 函数空间；multi-BD MAP-Elites success；Genesis 四操作任务优于经典 QD。
2. **LightNav-0** — dual-channel pointing + RVQ action tokenizer 统一 VLM 空间意图；2K+ 场景 / 4K+ h 数据；10 个导航仿真 SOTA + 跨本体零样本真机。
3. **Zeva** — 冻结策略 + 因果交互提取与双时间尺度记忆；部署无梯度更新自进化；RoboCasa365-Atomic5 平均 76.8%。
4. **Motus2** — policy/simulator/evaluator 三接口共享权重 GWM；人数据金字塔 + DiffusionNFT MBRL；灵巧双手真机（既有实体页）。
5. **CorrectVLA** — 任务级语言反馈 → 加性动作幅度修正；LIBERO-90 上 execution misalignment 可恢复、语义崩溃不适用。
6. **EMERGE-Policy** — 图结构多智能体编排；Main Agent + 感知/验证/记忆 Sub Agents；无需额外微调。
7. **OptiGeo** — bias-aware training 修复透明/反光/镜面监督偏差；30M 参数超 300M 单目基线。

## 对 wiki 的映射

- **7/7 独立详情节点**：每篇对应唯一 `wiki/entities/paper-*.md`；静态站 `detail.html?id=entity-paper-…` 均可直达。
- **本 ingest 新建 6** 个实体；**Motus2** 先前 ingest 已有 complete 页 → **只回链博客，不重复造页**。
- 阅读坐标：[开源系统闭环 7 篇技术地图](../../wiki/overview/open-source-system-loop-7-papers-technology-map.md)（**非**论文详情替代，仅作横切面索引）。
- 交叉：[VLA](../../wiki/methods/vla.md)、[生成式世界模型](../../wiki/methods/generative-world-models.md)、[World Action Models](../../wiki/concepts/world-action-models.md)、[具身导航](../../wiki/tasks/navigation.md)。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 7 篇独立节点核查（6 新建 / 1 复用 / **0 重复 arXiv 节点**）
- [x] 项目页与仓库开源状态核查（步骤 2.5）
