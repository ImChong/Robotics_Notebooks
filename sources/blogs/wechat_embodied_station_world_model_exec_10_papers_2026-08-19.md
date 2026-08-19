# 10篇机器人论文速览：世界模型很热，但真实执行才是硬门槛

> 来源归档（blog / 微信公众号）

- **标题：** 10篇机器人论文速览：世界模型很热，但真实执行才是硬门槛
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/NJ6M3CnsmDrtu9baRo8lgQ
- **发表日期：** 2026-08-19
- **入库日期：** 2026-08-19
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + `wechat-article-for-ai`（Camoufox）；`--no-images`；Jina Reader 对该链接触发微信 CAPTCHA，未采用
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_world_model_exec_10_papers_2026-08-19.md`](../raw/wechat_embodied_station_world_model_exec_10_papers_2026-08-19.md)
- **一句话说明：** 汇总 10 篇近期具身/机器人论文（文内均给项目页或代码链），主线从「看懂场景」推进到 **跨本体世界模型、社会/类人导航、空间记忆、开词汇主动感知、合成数据真机裁决、SMPC 示范 + 稀疏 RL、统一 VLA token 流、运动提示与手部可见性**；本库 **复用 4 个已有 complete 节点，新建 6 个独立论文实体，不重复造页**。

## 核心摘录（归纳，非全文）

文内判断：世界模型正在被要求跨过 **人到机器人** 的本体差异；导航研究开始把社会距离、类人动力学和 **主动感知** 纳入闭环；VLA 与空间记忆在探索不改模型或统一 token 流的推理–行动路径；控制侧则用 **SMPC** 示范和合成数据缓解稀疏奖励与真实数据不足。下一阶段关键不是单点刷榜，而是真实机器人上的 **空间证据、动作接口与可靠性估计** 是否闭环。

### 10 篇 → 本库节点

| # | 论文 | arXiv | 开源结论（入库日） | wiki |
|---|------|-------|-------------------|------|
| 01 | H2R-Bench | [2608.13049](https://arxiv.org/abs/2608.13049) | **部分开源**：项目页 + 仓已建，评测代码与标注 **未发布** | [paper-h2r-bench](../../wiki/entities/paper-h2r-bench.md) |
| 02 | DRL Proxemics 社会导航 | [2608.12917](https://arxiv.org/abs/2608.12917) | **未开源**：项目页 Code 链为 `#` 占位 | [paper-drl-proxemics-social-nav](../../wiki/entities/paper-drl-proxemics-social-nav.md) |
| 03 | HumanoidVLN | [2608.12860](https://arxiv.org/abs/2608.12860) | **待发布**（复用既有节点） | [paper-humanoidvln](../../wiki/entities/paper-humanoidvln.md) |
| 04 | Spatial Memory Agent (SMA) | [2608.12743](https://arxiv.org/abs/2608.12743) | **待发布**：项目页 **Code Coming Soon** | [paper-spatial-memory-agent](../../wiki/entities/paper-spatial-memory-agent.md) |
| 05 | SAP-Nav | [2608.12707](https://arxiv.org/abs/2608.12707) | **待发布**：仓仅 GitHub Pages，README 写 soon | [paper-sap-nav](../../wiki/entities/paper-sap-nav.md) |
| 06 | RoboSynChallenge | [2608.12416](https://arxiv.org/abs/2608.12416) | **已开源** 框架 + HF 数据/权重 | [paper-robosynchallenge](../../wiki/entities/paper-robosynchallenge.md) |
| 07 | SMPC→稀疏 RL 移动操作 | [2608.12063](https://arxiv.org/abs/2608.12063) | **已开源**（复用既有节点） | [paper-smpc2rl-loco-manipulation](../../wiki/entities/paper-smpc2rl-loco-manipulation.md) |
| 08 | Galaxea G0.5 | [2608.11739](https://arxiv.org/abs/2608.11739) | **已开源**（复用既有节点） | [paper-galaxea-g05](../../wiki/entities/paper-galaxea-g05.md) |
| 09 | Motion-as-Prompt (MaP) | [2608.11655](https://arxiv.org/abs/2608.11655) | **已开源** 训练无关框架（无 MaP 权重） | [paper-motion-as-prompt](../../wiki/entities/paper-motion-as-prompt.md) |
| 10 | Hand Visibility Detector | [2608.11574](https://arxiv.org/abs/2608.11574) | **已开源**（复用既有节点） | [paper-hand-visibility-detector](../../wiki/entities/paper-hand-visibility-detector.md) |

### 文内要点速记

1. **H2R-Bench** — 人类第一视角操作视频 → 指定机器人本体视频；五维诊断（目标状态、动作事件、功能接触、本体正确性、视频质量）；11 个生成模型仍弱于跨本体一致性。
2. **DRL Proxemics** — Hall 近体学径向高斯混合场作社会代价；接入已有 DRL 导航，社会指标升、效率仍 competitive。
3. **HumanoidVLN** — Isaac Sim 四本体物理 VLN；933 episode；JanusVLN 均值 SR 43.55%；G1 DualVLN sim–real r=0.935。
4. **SMA** — 冻结 VLM + verifier-guided 过程记忆；TRS 校准可靠性；5 benchmark × 4 base VLM 各 block 最高 macro avg。
5. **SAP-Nav** — 在线 Queryable Spatial-Semantic Representation + Active Viewpoint Verification；LangMap / HM3D-OVON SOTA，region SR +12.2%。
6. **RoboSynChallenge** — 合成 state-action 训练、**仅真实世界**评测；Transformer / Diffusion / VLA / WAM 基线同台。
7. **SMPC→RL** — 仿真 SMPC 作 expert 离线数据 + 稀疏任务奖励 off-policy RL；Spot 臂与 G1 真机部署。
8. **G0.5** — 单解码器自回归 CoT + 动作 token；跨本体 ActionCodec + 视觉记忆；七类 regime 超 prior art。
9. **MaP** — 轨迹画在帧间作 visual prompt；冻结 MLLM；CLEVRER / SSv2 运动推理涨点。
10. **Hand Visibility** — per-joint 可见性独立任务；visibility-weighted 三角化降 reprojection error。

## 对 wiki 的映射

- **复用 4** 个 complete 节点（HumanoidVLN、SMPC2RL、G0.5、Hand Visibility）；**新建 6** 个独立 `paper-*` 详情节点。
- 阅读坐标：[世界模型与真实执行 10 篇技术地图](../../wiki/overview/world-model-exec-10-papers-technology-map.md)。
- 交叉：[生成式世界模型](../../wiki/methods/generative-world-models.md)、[VLN](../../wiki/tasks/vision-language-navigation.md)、[VLA](../../wiki/methods/vla.md)、[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)、[WiLoR](../../wiki/methods/wilor.md)、[SAP-Nav](../../wiki/entities/paper-sap-nav.md)。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 10 篇独立节点核查（4 复用 / 6 新建 / 0 stub 重复）
- [x] 项目页与仓库开源状态核查（步骤 2.5）
