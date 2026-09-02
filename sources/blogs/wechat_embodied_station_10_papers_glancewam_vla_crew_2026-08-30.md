# 48ms 世界模型来了：10 个开源项目串起 VLA、多机协作与人形机器人

> 来源归档（blog / 微信公众号）

- **标题：** 48ms世界模型来了！10个开源项目串起VLA、多机协作与人形机器人
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/MdCtmijSM_VfYp19f-nZQw
- **发表日期：** 2026-08-30
- **入库日期：** 2026-08-30
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + `wechat-article-for-ai`（Camoufox）；`--no-images`；Jina Reader 对该链接触发微信 CAPTCHA，未采用
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_10_papers_glancewam_vla_crew_2026-08-30.md`](../raw/wechat_embodied_station_10_papers_glancewam_vla_crew_2026-08-30.md)
- **一句话说明：** 汇总 10 篇近期具身/机器人论文（文内均给项目页或代码链），主线是把关键结构从隐式学习改成显式接口：异步想象、行为意图、多机编排、双臂模态遮蔽、意图级模仿评测、导航–操作衔接、规划约束、仿真证书与模块化人形系统；**10/10 均有独立 `paper-*` 详情节点**（本 ingest **新建 8**、**既有 complete 2**；同一 arXiv **不重复造页**）。

## 核心摘录（归纳，非全文）

文内判断：具身下一阶段不是把 VLM 做得更大，而是重写决策链条——想象何时发生、行为目标如何进入解码器、多个机器人由谁验证动作、模拟结果如何变成现实证书、模块怎样在仿真与真机之间保持一致。开放资源也分层：部分已给训练/评测仓，部分只开项目页、数据集或评测平台；DreamMimic 仍写 Codes coming soon。复现时应区分论文数字、项目演示和可下载资产。

### 10 篇 → 本库节点

| # | 论文 | arXiv | 开源结论（入库日） | wiki |
|---|------|-------|-------------------|------|
| 01 | GlanceWAM | [2608.23927](https://arxiv.org/abs/2608.23927) | **已开源** MIT + HF 数据/权重 | [paper-glancewam](../../wiki/entities/paper-glancewam.md) |
| 02 | Indi（Intention Distillation） | [2608.23478](https://arxiv.org/abs/2608.23478) | **未开源** 仅项目页（复用既有节点） | [paper-indi](../../wiki/entities/paper-indi.md) |
| 03 | Physical Agentic AI | [2608.22657](https://arxiv.org/abs/2608.22657) | **已开源** MIT；mock / Gazebo / 真机分层 | [paper-physical-agentic-ai](../../wiki/entities/paper-physical-agentic-ai.md) |
| 04 | M3（Modality Masking） | [2608.22419](https://arxiv.org/abs/2608.22419) | **未开源** 仅项目页 | [paper-m3-modality-masking](../../wiki/entities/paper-m3-modality-masking.md) |
| 05 | The Imitator Game | [2608.22301](https://arxiv.org/abs/2608.22301) | **已开源** MIT 仓 + HF IG-10K / Assets；真机评测走 Arena（2026-09-02 再核） | [paper-imitator-game](../../wiki/entities/paper-imitator-game.md) |
| 06 | TONAV | [2608.22296](https://arxiv.org/abs/2608.22296) | **部分开源** 遥操作仓待齐；学习代码 Coming Soon | [paper-tonav](../../wiki/entities/paper-tonav.md) |
| 07 | DreamMimic | [2608.22278](https://arxiv.org/abs/2608.22278) | **待发布** Code Coming soon（复用既有节点） | [paper-dreammimic](../../wiki/entities/paper-dreammimic.md) |
| 08 | Meta-Ctrl | [2608.22149](https://arxiv.org/abs/2608.22149) | **未开源** 仅项目页 | [paper-meta-ctrl](../../wiki/entities/paper-meta-ctrl.md) |
| 09 | Bet4Sim2Real | [2608.21572](https://arxiv.org/abs/2608.21572) | **已开源** 证书实现 + 三组真机回放 | [paper-bet4sim2real](../../wiki/entities/paper-bet4sim2real.md) |
| 10 | GOLEM | [2608.21550](https://arxiv.org/abs/2608.21550) | **待核实** 项目页列 org，GitHub API 404 | [paper-golem-humanoid](../../wiki/entities/paper-golem-humanoid.md) |

### 文内要点速记

1. **GlanceWAM** — 异步 proposer 在后台生成数秒后的单帧前瞻，动作头在潜空间以 **48 ms** 解码动作块；RoboCasa 24 项 **72.2%**、LIBERO **99.0%**，A100 比同步基线快 24×。
2. **Indi** — 冻结教师 VLM 把行为级意图蒸馏进 VLA 解码器；GR00T-N1.7 SimplerEnv-Bridge **64.3→84.7%**，真机 **62.0→68.7%**。
3. **Physical Agentic AI** — Mission Planner 无执行权，Robot Orchestrator 逐项验证技能调用；检索把技能落地抬到 96%，错误派发仍 >20%；确定性编排器把错误派发降到 **0%**。
4. **M3** — 训练期随机屏蔽腕相机/语言/查询通道；RoboTwin 2.0 Clean **+21.7**、Clean2Rand **+11.4**；真机长时程完整任务成功率 **+30** 以上。
5. **Imitator Game** — L0–L3 四级基准 + IG-10K（2 万余组配对）；九个先进模型在 L3 功能替代上崩溃，未见任务零样本均 <13%。
6. **TONAV** — 任务导向导航 + 位置–速度动作块；导航从一开始服务接触任务，而不是「靠近后再操作」。
7. **DreamMimic** — RSSM 作预测表征与多步监督，而非在线规划；代码 Coming soon。
8. **Meta-Ctrl** — 元令牌拆开语法/语义约束；受约束解码内存从 **>107 TB 降至 <2 GB**；小开源 LM 在 WAH-NL 超过 GPT-4 子目标成功率。
9. **Bet4Sim2Real** — 仿真结果组合成逐次下注，证书 anytime-valid；相对基线平均收窄 **51.6%±16%**，≤30 样本仍收窄 **32.26%±8%**。
10. **GOLEM** — Unitree H1-2 模块化拆解系统；6 m 导航误差 **13.0 cm**；Ioniq 5 紧固件抓取 系留 97% → 自由站立 87% → 导航扰动 37%。

## 对 wiki 的映射

- **10/10 独立详情节点**：每篇对应唯一 `wiki/entities/paper-*.md`；静态站 `detail.html?id=entity-paper-…` 均可直达。
- **本 ingest 新建 8** 个实体；**Indi / DreamMimic** 在先前 ingest 已有 complete 页 → **只回链博客，不重复造页**（lint 禁止同一 arXiv 多 canonical 节点）。
- 阅读坐标：[48ms WAM / 编排 / 证书 10 篇技术地图](../../wiki/overview/glancewam-vla-crew-10-papers-technology-map.md)（**非**论文详情替代，仅作横切面索引）。
- 交叉：[World Action Models](../../wiki/concepts/world-action-models.md)、[VLA](../../wiki/methods/vla.md)、[Action Chunking](../../wiki/methods/action-chunking.md)、[Sim2Real](../../wiki/concepts/sim2real.md)、[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)、[双臂操作](../../wiki/tasks/bimanual-manipulation.md)。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 10 篇独立节点核查（8 新建 / 2 既有 complete / **0 重复 arXiv 节点**）
- [x] 项目页与仓库开源状态核查（步骤 2.5）
