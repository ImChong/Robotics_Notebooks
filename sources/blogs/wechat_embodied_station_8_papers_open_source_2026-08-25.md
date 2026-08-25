# 机器人圈开源清单：8篇新作，太空采矿、Q-Planning、视觉触觉全齐了

> 来源归档（blog / 微信公众号）

- **标题：** 机器人圈开源清单：8篇新作，太空采矿、Q-Planning、视觉触觉全齐了
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/71jZDzvcWZ3SsoHOEA8sgQ
- **发表日期：** 2026-08-25
- **入库日期：** 2026-08-25
- **抓取方式：** `wechat-article-for-ai`（Camoufox）；`--no-images`
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_8_papers_open_source_2026-08-25.md`](../raw/wechat_embodied_station_8_papers_open_source_2026-08-25.md)
- **一句话说明：** 汇总 8 篇近期机器人/具身论文（文内均给项目页或代码/数据链），主线从多模态物理感知、价值引导自改进、主动探索到显式安全控制与传感器物理层安全；**8/8 均有独立 `paper-*` 详情节点**（本 ingest **新建 5**、**复用 3 既有 complete**；同一 arXiv **不重复造页**）。

## 核心摘录（归纳，非全文）

文内判断：具身智能能力边界正从「看见并模仿」扩展为理解物理世界、主动获取信息，并在真实部署中持续变得更可靠。**ViTacPhys** 与 **DreamHand** 扩展物理与人体运动信号；**Q-Planning** 与 **PhysCaP** 让策略从部署反馈与主动试探中获益；**SRL-MPC** 把学习适应性嵌入显式安全结构；**TOSS** 与 **GhostTac** 分别提醒把人类教师与传感器攻击面纳入系统设计；**太空采矿综述** 开放资源清单补齐数据、仿真与验证基础设施。

### 8 篇 → 本库节点

| # | 论文 | arXiv | 开源结论（入库日） | wiki |
|---|------|-------|-------------------|------|
| 01 | Space Mining Survey | [2608.21358](https://arxiv.org/abs/2608.21358) | **已开源** 研究清单仓 `OpenSpace-Lab/Space-Mining-with-Robotics-List` | [paper-space-mining-with-robotics](../../wiki/entities/paper-space-mining-with-robotics.md) |
| 02 | ViTacPhys | [2608.21355](https://arxiv.org/abs/2608.21355) | **待发布**（复用既有节点） | [paper-vitacphys](../../wiki/entities/paper-vitacphys.md) |
| 03 | Q-Planning | [2608.21204](https://arxiv.org/abs/2608.21204) | **已开源**（复用既有节点） | [paper-qplanning](../../wiki/entities/paper-qplanning.md) |
| 04 | SRL-MPC | [2608.21175](https://arxiv.org/abs/2608.21175) | **待发布** 仓已建，README 写明录用后释码 | [paper-srl-mpc](../../wiki/entities/paper-srl-mpc.md) |
| 05 | TOSS Framework | [2608.21083](https://arxiv.org/abs/2608.21083) | **已开源** OSF 数据集与材料 | [paper-toss-framework](../../wiki/entities/paper-toss-framework.md) |
| 06 | PhysCaP | [2608.21031](https://arxiv.org/abs/2608.21031) | **未开源** 项目页无 GitHub | [paper-physcap](../../wiki/entities/paper-physcap.md) |
| 07 | GhostTac | [2608.20817](https://arxiv.org/abs/2608.20817) | **已开源** `GhostTac/GhostTac_CCS` 演示代码 | [paper-ghosttac](../../wiki/entities/paper-ghosttac.md) |
| 08 | DreamHand | [2608.20308](https://arxiv.org/abs/2608.20308) | **待发布**（复用既有节点） | [paper-dreamhand](../../wiki/entities/paper-dreamhand.md) |

### 文内要点速记

1. **Space Mining** — 六阶段勘探–采样–提取架构 + 持续更新研究资源库。
2. **ViTacPhys** — 人体视触觉示范预测质量/刚度/摩擦；ID 95.0%、OOD 83.4% 抓取成功率。
3. **Q-Planning** — 冻结大 BC，只训小 Q；LIBERO-10 93→99%，双臂叠杯 40→90%。
4. **SRL-MPC** — RL 调 MPC 参数 + 形状感知 HOCBF；25 机器人密集场景 86.7% 成功率。
5. **TOSS** — Triggers/Objectives/Signals/Strategies 四维教学决策 + 开放数据集。
6. **PhysCaP** — Code-as-Policy + 本体感觉估计质量/刚度；双代理 Planner/Prioritizer。
7. **GhostTac** — 非接触 EMI 操纵触觉输出；15 种传感器验证。
8. **DreamHand** — VDM 作几何编码器恢复遮挡双手 3D 轨迹；ARCTIC/HOT3D MPJPE-p ↓30%/40%。

## 对 wiki 的映射

- **8/8 独立详情节点**：每篇对应唯一 `wiki/entities/paper-*.md`；静态站 `detail.html?id=entity-paper-…` 均可直达。
- **本 ingest 新建 5** 个实体；**ViTacPhys / Q-Planning / DreamHand** 在先前 ingest 已有 complete 页 → **只回链博客，不重复造页**。
- 阅读坐标：[开源 8 篇技术地图](../../wiki/overview/open-source-8-papers-technology-map.md)（**非**论文详情替代，仅作横切面索引）。
- 交叉：[VLA](../../wiki/methods/vla.md)、[模仿学习](../../wiki/methods/imitation-learning.md)、[Model Predictive Control](../../wiki/methods/model-predictive-control.md)、[触觉感知](../../wiki/concepts/tactile-sensing.md)、[Sim2Real](../../wiki/concepts/sim2real.md)。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 8 篇独立节点核查（5 新建 / 3 复用 / **0 重复 arXiv 节点**）
- [x] 项目页与仓库/数据集开源状态核查（步骤 2.5）
