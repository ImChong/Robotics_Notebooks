# CLAP 代码模型全开源！9 篇开源论文串起跨本体世界模型与 VLA

> 来源归档（blog / 微信公众号）

- **标题：** CLAP代码模型全开源！9篇开源论文串起跨本体世界模型与VLA
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/J62q2IVvvBDyT_8OTR9KZQ
- **发表日期：** 2026-08-31
- **入库日期：** 2026-08-31
- **抓取方式：** Agent Reach + `wechat-article-for-ai`（Camoufox）；`--no-images`
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_clap_9_papers_open_source_2026-08-31.md`](../raw/wechat_embodied_station_clap_9_papers_open_source_2026-08-31.md)
- **一句话说明：** 汇总 9 篇近期机器人与具身论文，主线为 CLAP / Riemann-1.0 扩展世界模型与本体边界，FlashVLA 流式 VLA，TrapVLA / ESRP 暴露安全与长时程规划难题；**9/9 均有独立 `paper-*` 详情节点**（本 ingest **新建 6**、**复用 3 既有 complete**；同一 arXiv **不重复造页**）。

## 核心摘录（归纳，非全文）

文内判断：具身系统正从单一动作预测走向可模拟、可流式执行、可诊断并可跨本体迁移的完整闭环。三层主线：**CLAP 与 Riemann-1.0** 扩展世界模型数据与本体边界；**FlashVLA** 把动作生成改造成稳定流式过程；**TrapVLA 与 ESRP** 暴露安全与长时程规划新难题；**MILO、ViTaR、AlloEgo-VLM、MistyPilot** 分别从三维交互、触觉校正、空间语义与智能体编排补齐感知—执行接口。

### 9 篇 → 本库节点

| # | 论文 | arXiv | 开源结论（入库日） | wiki |
|---|------|-------|-------------------|------|
| 01 | MILO | [2608.27407](https://arxiv.org/abs/2608.27407) | **已开源** `ac5113/MILO` | [paper-milo](../../wiki/entities/paper-milo.md) |
| 02 | CLAP | [2608.27406](https://arxiv.org/abs/2608.27406) | **已开源** 代码与模型（复用既有页） | [paper-clap-cross-embodiment](../../wiki/entities/paper-clap-cross-embodiment.md) |
| 03 | FlashVLA | [2608.27384](https://arxiv.org/abs/2608.27384) | **已开源** `z-lab/flashvla`（复用既有页） | [paper-flashvla](../../wiki/entities/paper-flashvla.md) |
| 04 | ESRP | [2608.27371](https://arxiv.org/abs/2608.27371) | **未开源**：项目页 + ESRP-Bench 说明；无公开训练仓 | [paper-esrp](../../wiki/entities/paper-esrp.md) |
| 05 | Riemann-1.0 | [2608.27033](https://arxiv.org/abs/2608.27033) | **确认未开源**（复用既有页） | [paper-riemann-1](../../wiki/entities/paper-riemann-1.md) |
| 06 | TrapVLA | [2608.26578](https://arxiv.org/abs/2608.26578) | **未开源**：`John-liua/TrapVLA` 仅为项目页静态站 | [paper-trapvla](../../wiki/entities/paper-trapvla.md) |
| 07 | ViTaR | [2608.15816](https://arxiv.org/abs/2608.15816) | **待发布**：项目页 Code Coming soon | [paper-vitar](../../wiki/entities/paper-vitar.md) |
| 08 | AlloEgo-VLM | [2608.15605](https://arxiv.org/abs/2608.15605) | **已开源** `CKL9001/AlloEgo-VLM` | [paper-alloego-vlm](../../wiki/entities/paper-alloego-vlm.md) |
| 09 | MistyPilot | [2608.15549](https://arxiv.org/abs/2608.15549) | **已开源** `WangXiaoShawn/MistyPilot` | [paper-mistypilot](../../wiki/entities/paper-mistypilot.md) |

### 文内要点速记

1. **MILO** — LRM 网格作 HOI 几何脚手架；单图人—物三维交互；InterCap / HODome / IMHD SOTA。
2. **CLAP** — 跨本体视频 WM；LAM→EE 课程；零样本规划 \(\pi_{0.5}\) / MolmoAct-2；全代码与模型开源。
3. **FlashVLA** — 流式噪声缓冲 + chunk 因果注意力；真机单卡 ≥30 Hz。
4. **ESRP** — 第一视角 + 俯视目标布局下的家具重排；ESRP-Bench 5400+ 场景 / 8200 物体。
5. **Riemann-1.0** — 全因果 WAM；232K+ h 预训练；RoboCasa365 62.6%、真机 85% SR。
6. **TrapVLA** — 配置化 VLA 后门；Trap-LIBERO / Trap-RoboTwin 四类失败模式。
7. **ViTaR** — 冻结 VLA + 视触觉有界残差；UniVTAC +30.6 pt。
8. **AlloEgo-VLM** — 消歧 allocentric / egocentric 参照系；Isaac Sim 开放物体搜索验证。
9. **MistyPilot** — 多智能体 LLM 技能编排；Misty 真机路由 / 传感器绑定 / 状态复用。

## 对 wiki 的映射

- **9/9 独立详情节点**：每篇对应唯一 `wiki/entities/paper-*.md`；静态站 `detail.html?id=entity-paper-…` 均可直达。
- **本 ingest 新建 6** 个实体；**CLAP / FlashVLA / Riemann-1.0** 先前 ingest 已有 complete 页 → **只回链博客，不重复造页**。
- 阅读坐标：[CLAP / 跨本体 WM / VLA 9 篇技术地图](../../wiki/overview/clap-cross-embodiment-vla-wm-9-papers-technology-map.md)（**非**论文详情替代，仅作横切面索引）。
- 交叉：[生成式世界模型](../../wiki/methods/generative-world-models.md)、[VLA](../../wiki/methods/vla.md)、[World Action Models](../../wiki/concepts/world-action-models.md)、[Manipulation](../../wiki/tasks/manipulation.md)。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 9 篇独立节点核查（6 新建 / 3 复用 / **0 重复 arXiv 节点**）
- [x] 项目页与仓库开源状态核查（步骤 2.5）
