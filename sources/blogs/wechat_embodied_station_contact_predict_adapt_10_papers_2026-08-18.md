# 具身智能下一站：这10篇机器人论文，把“看懂”推进到“会接触、会预测、会适应”

> 来源归档（blog / 微信公众号）

- **标题：** 具身智能下一站：这10篇机器人论文，把“看懂”推进到“会接触、会预测、会适应”
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/IxmKI4_JYy1KBfp_JCZFLw
- **发表日期：** 2026-08-18
- **入库日期：** 2026-08-18
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + `wechat-article-for-ai`（Camoufox）；`--no-images`；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_contact_predict_adapt_10_papers_2026-08-18.md`](../raw/wechat_embodied_station_contact_predict_adapt_10_papers_2026-08-18.md)
- **一句话说明：** 汇总 10 篇近期具身/机器人论文（文内均给项目页或代码链），主线从「看懂场景」推进到 **稳健接触、可控预测、长期适应**；本库 **复用 1 个已有 complete 节点（Seeker），新建 9 个独立论文实体，不重复造页**。

## 核心摘录（归纳，非全文）

文内判断：这批工作不再把视觉–语言理解当终点，而是把 **触觉/力觉、约束优化、时序预测、社会上下文** 接到学习闭环。落地关键不是单点模型变大，而是接触要能调力、预测要忠实于动作、策略要能迁移、安全与社会规范要能被显式控制。

### 10 篇 → 本库节点

| # | 论文 | arXiv | 开源结论（入库日） | wiki |
|---|------|-------|-------------------|------|
| 01 | TF-ART 触觉/力觉综述 | [2608.07558](https://arxiv.org/abs/2608.07558) | **已开源** Awesome 清单 + 项目页；无可运行训练入口 | [paper-tf-art-tactile-force-survey](../../wiki/entities/paper-tf-art-tactile-force-survey.md) |
| 02 | AutoPSO | [2608.07539](https://arxiv.org/abs/2608.07539) | **已开源** EvoX/PyTorch 双层 PSO 搜索 | [paper-autopso](../../wiki/entities/paper-autopso.md) |
| 03 | HUI360 | [2608.11051](https://arxiv.org/abs/2608.11051) | **已开源** 基线仓 + HF 标注；原始全景需 DTA | [paper-hui360](../../wiki/entities/paper-hui360.md) |
| 04 | 顶层布料分割 | [2608.10648](https://arxiv.org/abs/2608.10648) | GitHub **空仓**（size 0） | [paper-top-layer-fabric-seg](../../wiki/entities/paper-top-layer-fabric-seg.md) |
| 05 | BooST | [2608.10600](https://arxiv.org/abs/2608.10600) | 项目页已发；代码仓 **仅 github.io** | [paper-boost-skill-transfer](../../wiki/entities/paper-boost-skill-transfer.md) |
| 06 | 真机双臂灵巧抓取 | [2608.10383](https://arxiv.org/abs/2608.10383) | **已开源** DDPM 训练/推理 + 数据样例；全集网盘 | [paper-real-bi-dex-grasp](../../wiki/entities/paper-real-bi-dex-grasp.md) |
| 07 | 接近–安全约束分解跟随 | [2608.10056](https://arxiv.org/abs/2608.10056) | **已开源** CrowdNav 扩展 + PPO-Lagrangian | [paper-nav-ps-balance](../../wiki/entities/paper-nav-ps-balance.md) |
| 08 | DreamX-Phi 1.0 | [2608.13489](https://arxiv.org/abs/2608.13489) | **部分开源**：占位 README，权重待赛后 | [paper-dreamx-phi](../../wiki/entities/paper-dreamx-phi.md) |
| 09 | Mind the Context（EDD） | [2608.13448](https://arxiv.org/abs/2608.13448) | **已开源** 训练/评测 notebook；数据集需自备 | [paper-mind-the-context](../../wiki/entities/paper-mind-the-context.md) |
| 10 | Seeker | [2608.13422](https://arxiv.org/abs/2608.13422) | **已开源**（复用既有节点） | [paper-seeker](../../wiki/entities/paper-seeker.md) |

### 文内要点速记

1. **TF-ART** — 触觉/力觉从补传感器升为主线；taxonomy 同时覆盖多模态与多阶段策略–控制管线（266 篇）。
2. **AutoPSO** — 外层搜 PSO 组件、内层实例化求解；EvoX 张量化；神经进化机器人控制也受益。
3. **HUI360** — 移动机器人 360° 第一人称野外 HRI 预测；1M 标注 + 跨数据集评估。
4. **顶层布料分割** — edge-aware + shape-aware（CAD mask）监督 encoder-decoder。
5. **BooST** — 语义意图与运动动态进同一 VQ-VAE，再蒸馏轻量策略；LIBERO-90 10 demo **0.70**。
6. **双臂灵巧抓取** — 单视角点云 + DDPM 关节配置 + 在线力细化；IROS 2026。
7. **接近–安全跟随** — 稀疏任务奖励 + 独立 cost 阈值；ID 成功率 **78.08%**。
8. **DreamX-Phi** — 动作条件视频 WM；PRoPE 式 SE(3) 注入；WorldArena 2.0 Track 1 第一。
9. **Mind the Context** — 环境/社会双分支 + replay；社交适当性持续学习。
10. **Seeker** — 动作监督 ROI，无需 gaze/框；已有 complete 节点。

## 对 wiki 的映射

- **复用** [Seeker](../../wiki/entities/paper-seeker.md)；**新建 9** 个独立 `paper-*` 详情节点；阅读坐标见 [接触–预测–适应 10 篇技术地图](../../wiki/overview/contact-predict-adapt-10-papers-technology-map.md)。
- 交叉：[Tactile Sensing](../../wiki/concepts/tactile-sensing.md)、[Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md)、[生成式世界模型](../../wiki/methods/generative-world-models.md)、[双臂操作](../../wiki/tasks/bimanual-manipulation.md)、[模仿学习](../../wiki/methods/imitation-learning.md)、[LIBERO](../../wiki/entities/libero-benchmark.md)、[PGIF-MPPI](../../wiki/entities/paper-pgif-mppi.md)。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 10 篇独立节点核查（1 复用 / 9 新建 / 0 stub 重复）
- [x] 项目页与仓库开源状态核查（步骤 2.5）
