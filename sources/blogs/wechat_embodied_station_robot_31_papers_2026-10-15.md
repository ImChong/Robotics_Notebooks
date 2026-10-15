# 机器人出错后，退一步能否做得更好？｜附论文与复现线索（31 篇速览）

> 来源归档（blog / 微信公众号）

- **标题：** 机器人出错后，退一步能否做得更好？｜附论文与复现线索
- **类型：** blog
- **作者：** 具身智能小站（微信公众号，与 [10 篇速览归档](wechat_embodied_station_world_model_exec_10_papers_2026-08-19.md) 同号）
- **原始链接：** https://mp.weixin.qq.com/s/ZfE6_UgqmHFePROtf2eE2w
- **发表日期：** 2026-10-15
- **入库日期：** 2026-10-15
- **抓取方式：** WebFetch 返回微信「环境异常」验证页；改用浏览器 UA 直接取 HTML，本地解析 `#js_content` 得到正文纯文本（未下载图片）
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_robot_31_papers_2026-10-15.md`](../raw/wechat_embodied_station_robot_31_papers_2026-10-15.md)
- **一句话说明：** 当日汇总 31 篇具身智能新论文（arXiv 2610.19xxx–2610.20xxx），按阅读优先级分为「深读 1 / 跟进 2 / 扫读 28」；主题覆盖 VLA 动态适配与执行恢复、人形控制与移动操作、触觉/力觉数据、仿真生成与评测。

## 核心摘录（归纳，非全文）

文内主线：固定参数的 VLA 难同时胜任多种操作；执行失败后「回退重试」需要学习何时中止、退回何处；接触与力觉数据的难点在同步与质量控制而不只是数量。文内反复强调 **同预算比较、仿真与真机结论区分、代价（延迟/算力）一起看**。

### 深读 / 跟进（文内标注优先级）

| # | 论文 | arXiv | 文内列出的代码/资源 |
|---|------|-------|---------------------|
| 1 | MorphAct：Robot Policies Morphing with the Worlds They Act In（深读） | [2610.20019](https://arxiv.org/abs/2610.20019) | [代码](https://github.com/MINT-SJTU/MorphAct)、项目页 |
| 2 | ACRO：Actor-Critic Rollout Orchestration（跟进） | [2610.19677](https://arxiv.org/abs/2610.19677) | [代码](https://github.com/JunHoo-Lee/ACRO)、项目页 |
| 3 | TacClaw-Infinity：视触觉语言动作数据集（跟进） | [2610.20320](https://arxiv.org/abs/2610.20320) | [资源](https://github.com/daimon-robotics/TacClaw-Infinity)、项目页 |

### 扫读 28 篇（文内每篇一句判断）

| # | 论文 | arXiv | 文内列出的代码/资源 |
|---|------|-------|---------------------|
| 4 | Atomic-WAM：执行反馈校准原子动作推理 | [2610.20157](https://arxiv.org/abs/2610.20157) | 代码、项目页 |
| 5 | ForceBench：腕力融合基准 | [2610.19604](https://arxiv.org/abs/2610.19604) | 项目页 |
| 6 | Cracking Mobile Manipulation：移动操作失败诊断与智能体系统 | [2610.19600](https://arxiv.org/abs/2610.19600) | 代码 |
| 7 | Hyperspherical Contrastive Representations（C-Slerp） | [2610.20409](https://arxiv.org/abs/2610.20409) | 资源、项目页 |
| 8 | HiFun：灵巧手功能操作分层框架 | [2610.19518](https://arxiv.org/abs/2610.19518) | 项目页 |
| 9 | GR-BC：带人工干预的图重加权行为克隆 | [2610.20504](https://arxiv.org/abs/2610.20504) | 项目页 |
| 10 | RAIN：自适应动作块精修 | [2610.19765](https://arxiv.org/abs/2610.19765) | 代码、项目页 |
| 11 | Active Generative Simulation（MetaGenSim） | [2610.19479](https://arxiv.org/abs/2610.19479) | 项目页 |
| 12 | When Do Discrete Actions Make Good Continuous Controllers? | [2610.20429](https://arxiv.org/abs/2610.20429) | 项目页 |
| 13 | HiRe-MoMa：分层残差策略学习（移动操作） | [2610.20607](https://arxiv.org/abs/2610.20607) | 项目页 |
| 14 | PICES：扰动感知的 VLA 推理加速 | [2610.19656](https://arxiv.org/abs/2610.19656) | 代码（anonymous 4open） |
| 15 | PhySGA：物理空间 3D 组合智能体 | [2610.20560](https://arxiv.org/abs/2610.20560) | 资源、项目页 |
| 16 | HM3D-M3：多智能体多模态多目标导航基准 | [2610.20121](https://arxiv.org/abs/2610.20121) | 代码、项目页 |
| 17 | What Happens to the Goal? 目标错泛化 | [2610.19731](https://arxiv.org/abs/2610.19731) | 代码 |
| 18 | Robota-Embed：统一 VLA 嵌入用于上下文学习 | [2610.19495](https://arxiv.org/abs/2610.19495) | 项目页 |
| 19 | Dungeon and Visibility World Generator | [2610.19523](https://arxiv.org/abs/2610.19523) | 代码 |
| 20 | Towards Agentic Robots（综述） | [2610.20442](https://arxiv.org/abs/2610.20442) | 资源（Awesome 列表） |
| 21 | PingpongDuo：人形乒乓双打 | [2610.20579](https://arxiv.org/abs/2610.20579) | 项目页 |
| 22 | KinoGen：运动学约束视频生成的人形参考 | [2610.20552](https://arxiv.org/abs/2610.20552) | 项目页 |
| 23 | ObjDream：VLA 的物体级理解与未来状态预测 | [2610.20223](https://arxiv.org/abs/2610.20223) | 项目页 |
| 24 | FOMI：力反馈操作接口 | [2610.19839](https://arxiv.org/abs/2610.19839) | 项目页 |
| 25 | One-shot Human Video-Guided Adaptation（CapPro） | [2610.19746](https://arxiv.org/abs/2610.19746) | 项目页 |
| 26 | Modeling and Understanding Neural Weights of Humanoid Policies with Diffusion | [2610.20606](https://arxiv.org/abs/2610.20606) | 项目页 |
| 27 | Human Demonstrations as Post-Training Data for Any Robot Body | [2610.20351](https://arxiv.org/abs/2610.20351) | 项目页 |
| 28 | Tactile-Guided Robot Learning（综述） | [2610.20134](https://arxiv.org/abs/2610.20134) | 项目页 |
| 29 | Unified Continuum Robot Actuation Platform | [2610.19798](https://arxiv.org/abs/2610.19798) | 项目页 |
| 30 | Eye, Robot：VLM 引导的注视动力学 | [2610.19635](https://arxiv.org/abs/2610.19635) | 代码（anonymous 4open） |
| 31 | Topology-Preserving Pipe Centerline Refinement | [2610.19403](https://arxiv.org/abs/2610.19403) | 代码 |

> 注：文内未列代码/项目页的论文，表中以空白表示；「代码」列为文章所写链接，**未逐一打开核查**（见下文「当前提炼状态」）。

## 对 wiki 的映射

- **本次未新建 wiki 实体页**：31 篇中多数为扫读级别，且与既有节点的对应关系未核查；按 [schema/ingest-workflow.md](../../schema/ingest-workflow.md)「一次一条资料，深度到位再推进」，暂只归档。
- **候选升格（待定）**：#1 MorphAct（文内深读项；有代码与项目页）、#2 ACRO、#3 TacClaw-Infinity。若需要升格，应先按步骤 2.5 打开各项目页核查源码开放状态。
- 交叉目标（待核对）：可能相关的已有页 [VLA](../../wiki/methods/vla.md)；具体条目需逐篇核对既有 arXiv 节点后再链接。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档（31 个 arXiv 链接已对齐文内 31 篇）
- [ ] 逐篇打开项目页 / 代码仓库核查开源状态（步骤 2.5）— **未做**
- [ ] wiki 实体页升格 — **未做**，待确认优先对象
