# [2026-10-08] ingest | sources/blogs/boston_dynamics_robot_hands_modern_ai_real_work.md — 收录 Atlas 灵巧手设计

- 意图：整理 Boston Dynamics 官方博客中 Atlas 13-DoF 手的机构、仿真和 sim-to-real 设计。
- 开源核查：该文未列手部 CAD、仿真资产、训练代码或数据集；仅记录本文未提供复现材料。
- 关键页：[Boston Dynamics Atlas 13-DoF 灵巧手](../wiki/entities/boston-dynamics-atlas-13dof-hand.md)；[Boston Dynamics](../wiki/entities/boston-dynamics.md)。

# [2026-10-08] ingest | LegoFlow — agent-native SWE 数据工程流水线

- 意图：将 LegoFlow 项目博客、Apache-2.0 代码仓库与公开 LegoFlow-SWE 任务/轨迹数据归档到一个项目实体节点。
- 开源核查：代码已开源；HF 数据可访问，dataset card 独立许可证未明确核实。
- 关键页：[LegoFlow](../wiki/entities/legoflow.md)；[AI Agent 评测](../wiki/concepts/ai-agent-evaluation.md)。

# ingest McByte++

**意图：** 收录截图中 Inria STARS 的 McByte++ 体育视频长时多目标跟踪项目。

**开源结论：** 官方 McBytePlusPlus 仓库已公开并声明 Apache-2.0；论文为 arXiv:2608.15688。节点记录在线 Re-ID、轻量掩码传播与条件相机运动补偿，同时注明掩码实现需一次载入全部帧，官方 FPS 不含检测耗时。

**关键页：** [McByte++ 实体页](../wiki/entities/mcbyte-plus-plus.md)；与 [Roboflow Sports](../wiki/entities/roboflow-sports.md) 互链以区分专用跟踪器和一般体育 CV 工具。

# 2026-10-08 — HarnessPAI

- **操作：** 更新已有项目/论文节点（不新建重复实体），修订论文与仓库来源归档，新增官方项目页来源归档。
- **来源：** arXiv:2609.29166；Darwin Agent Team 项目页；Darwin-Agent/HarnessPAI README。
- **关键更正：** 旧条目称“已开源”；官方 README 截至 2026-10-08 明确研究代码尚未公开，安装与复现说明未提供。GitHub 仓库目前是论文/项目页配套仓库。
- **内容：** 补充作者机构、双时间尺度方法、评测口径与跨论文基线边界；将论文方法流程与代码运行时序图区分。
- **目标节点：** [HarnessPAI](../wiki/entities/paper-harnesspai.md)
- **来源归档：** [论文](../sources/papers/harnesspai_arxiv_2609_29166.md)、[代码仓库](../sources/repos/harnesspai.md)、[项目页](../sources/sites/harnesspai.md)

# Ingest 日志片段

- **日期：** 2026-10-09
- **意图：** 根据截图入库 Emil Kowalski 的设计工程 Agent Skills。
- **开源结论：** 上游公开仓库与 MIT License 已核查；本次归档固定到 2026-10-02 的 README/commit 快照。
- **关键页：** `wiki/entities/emil-kowalski-skills.md`；补充 Agent Skills 发现页的反向链接。

# Ingest 日志片段

- **日期：** 2026-10-09
- **意图：** 根据截图更新 HumanLayer `show-me` Skill 的已有知识节点。
- **开源结论：** `humanlayer/skills` 已有入库节点，本次补入官方博客并刷新 README 技能清单快照，避免重复建实体。
- **关键页：** `wiki/entities/humanlayer-skills.md`；新增博客来源归档。

# Ingest 日志片段

- **日期：** 2026-10-09
- **意图：** 根据截图入库 Microsoft VibeVoice-ASR 与相关开源模型系列。
- **开源结论：** 主仓 MIT；VibeVoice-ASR 模型卡亦标 MIT；权重与代码分开核验。
- **关键页：** `wiki/entities/microsoft-vibevoice.md`；连接到长音频说话人归因转写与人形语音交互。

# Ingest 日志片段

- **日期：** 2026-10-09
- **意图：** 将 rohitg00 的 LLM Wiki v2 Gist 作为 Karpathy LLM Wiki 的扩展来源写回现有方法页。
- **版本/许可：** 页面显示 7 次修订，最近活跃于 2026-10-08；未发现明确许可证，按公开文章/idea file 归档，不标注开源代码。
- **关键页：** 更新 `wiki/references/llm-wiki-karpathy.md`，新增 v2 来源摘录。

---
date: 2026-10-09
op: ingest
---

将 arXiv:2608.30773 纳入软体机器人与接触丰富操作知识脉络，整理其“交互获取信息”的方法主张、循环策略及两阶段 Sim2Real。当前提交资料未提供官方代码入口，页面明确标注复现边界，并与接触丰富操作概念页建立回链。

# HuMBLE ingest

- **日期：** 2026-10-10
- **论文：** [HuMBLE: Human Motion-Driven Behavior Learning for Embodied Locomotion](https://arxiv.org/abs/2610.10489)
- **类型：** arXiv 论文 / 人形机器人 locomotion 方法
- **来源记录：** `sources/papers/humble_arxiv_2610_10489.md`
- **实体页：** `wiki/entities/paper-humble-human-motion-driven-behavior.md`
- **摘要：** 以人体步态 retarget 动捕训练全身参考 teacher，蒸馏为 SE(2) 速度指令 + proprioception 策略，再以参考模仿和目标跟踪双任务 PPO 微调；在 Atlas R1/D1 与 Unitree G1 验证。
- **状态核查：** 论文与补充材料公开；未找到作者公开代码/权重入口。G1 数据被描述为将发布，Atlas 数据因专有属性不公开。
- **交叉链接：** `wiki/tasks/humanoid-locomotion.md`、`wiki/overview/humanoid-amp-motion-prior-survey.md`

# [2026-10-10] ingest | TEMPO 动态操作

- **意图：** 归档 arXiv:2609.16864、项目页与官方源码，将这项以时间上下文提升动态操作 VLA 的工作接入知识图谱。
- **开源结论：** 官方代码仓库提供数据预处理、微调和服务入口；README 仍将训练数据、检查点与 YAM 部署列为 TODO，论文发布的 TEMPO-Bench 下载入口尚未从 README 核实。
- **关键节点：** 独立记录 TEMPO 动态操作论文实体，链接 VLA 与 Manipulation；明确与另一篇同名 TEMPO（arXiv:2608.07314）区分。

- **操作：** ingest
- **意图：** 收录人类数据驱动人形控制与动作保真机器人世界模型，并核对相关项目的代码/数据开放状态。
- **开源结论：** VioLA 论文承诺后续发布代码与 checkpoint，尚未核实可访问入口；DreamTrue 官方仓库已开放 calibration、wmvideo、reward 组件及部分数据，完整数据待发布。SGS（arXiv:2610.12465）已在 main 有完整实体，本次复用既有节点、不重复建页。
- **关键知识节点：** VioLA 人形控制实体、DreamTrue 世界模型实体；补入 VLA、人形 locomotion、WAM、manipulation 的关联入口。
- **日期：** 2026-10-11

- **操作：** ingest（既有实体增量更新）
- **意图：** 根据 NVIDIA Long-WAM 项目页新增的 RoboCasa365 agent 组合与真机结果更新来源归档和既有论文实体。
- **更新：** 记录 GPT-6 Astra + Long-WAM 的 50-task RoboCasa365 结果，并补充 G1 移动杯抓取、YAM 长时任务报告；明确组合系统与 Long-WAM 单体结果的归属边界。
- **去重：** Long-WAM 已有 canonical 节点，本次不新建实体。
- **日期：** 2026-10-11

# 2026-10-11 ingest

接入 MemCompiler，归档论文、项目页和官方代码入口，提炼逐步状态条件编译、Brief State 与 Soft-Mem 双通道方法，并与现有具身记忆/部署期适应节点互链。论文称代码与数据可用；README 有 ALFWorld、ScienceWorld 运行入口，复现仍依赖外部环境数据、模型/API 配置。Zetta 与 Zeva 的 arXiv 节点已存在主库，未重复创建。

---
op: ingest
date: 2026-10-11
---
RoboDuet（RA-L 2025，arXiv:2403.17367）新增独立项目实体并归档论文、项目页、训练实现和真机部署代码；训练与部署仓库均开放，后者覆盖 Go1/Go2 + ARX5。

# 2026-10-12 — Ingest Draft

- 新增 [Draft 项目实体](../wiki/entities/paper-draft-parametric-robot-design.md)：记录参数规格到 MJCF 的编译链、数据拟合与 twin 验证、统一 RL 课程实验、开源代码入口和适用边界。
- 新增论文摘录与源码归档：[论文](../sources/papers/draft_arxiv_2609_38405.md)、[官方仓库](../sources/repos/draft.md)。
- 与 MuJoCo/MJCF 及自动形态搜索路线建立交叉引用。
- Wiki Lint 暴露一篇超过 180 天未复核的 RL 方法对比页；复核并修正其绝对化比较及不可横向比较的固定步数示例，使质量检查不被仓库级陈旧提醒阻塞。\n- 未修改 Actions/workflows、catalog、log 汇总或生成文件。

---
date: 2026-10-13
type: ingest
---

- Ingest: **Robotic Servo Tracking of Moving Targets with Dynamic Imitation Constraints** (arXiv:2609.14589; IROS 2026).
- 新增论文实体与来源摘录；补充 Visual Servoing 方法页反向链接及轨迹级动态约束对比。
- 核对 Kinova/RealSense 真机结果、理论假设与代码开放状态；截至入库日未找到官方可运行代码入口。
- 未更新 catalog.md、log.md 或导出文件。

# 2026-10-13 — Ingest VICAR

- 新增 [VICAR 实体](../wiki/entities/paper-vicar-contact-aware-video-imitation.md)：总结视频/场景恢复、接触与速度保持的两阶段重定向、movable-contact generator、PPO 训练、仿真/真机结果与边界。
- 归档 [arXiv 论文](../sources/papers/vicar_arxiv_2610_13079.md)、[官方项目页](../sources/sites/vicar.md) 和 [官方仓库](../sources/repos/vicar.md)。
- 与 VisualMimic、动作重定向流水线、接触丰富操作等知识节点交叉链接。
- 仅新增与更新知识/来源源文件；未改 Actions/workflows 或生成文件。

---
date: 2026-10-13
type: ingest
---

- Ingest: geo-sleuth（Oldcircle/geo-sleuth），开源跨 Agent 照片地理定位 skill。
- 新增 wiki 实体与仓库来源归档；补入 OSIRIS 与 LAYS 节点的反向链接。
- 记录脚本分层、地图/地形/影像候选核验、官方示例的精度边界，以及隐私/用途限制。
- 未修改 catalog.md、log.md 或导出文件；本次仅核读官方仓库文档，未本地运行定位流程。

# 2026-10-13 — SimVLA ingest

- 新增 SimVLA 论文实体与论文、项目主页、代码仓库来源归档。
- 记录 SimAction/SimVQA 预训练、SimDeploy rollout 后训练、数据规模与真机零样本评测。
- 明确公开仓库边界：仿真/数据工作流可用，训练 recipe、SimDeploy 训练适配器与 checkpoint 未包含。
- 关联 VLA、manipulation、loco-manipulation 导航。

- **操作：** ingest
- **说明：** 归档 Axis Robotics Platform 官方技术报告，补充既有 Axis Robotics 实体的跨仿真器数据链路、LIBERO 轨迹清洗验证和部分开源边界。

- **操作：** ingest
- **说明：** 入库 WING（arXiv:2610.03607），整理交互中心潜动作、低频 DCT 引导、仿真/真机评测和代码待发布状态，并关联 WAM 知识页。

# InternW0-Δ 入库记录

- 日期：2026-10-13
- 唯一实体：`wiki/entities/paper-internw0-delta.md`（沿用已有实体，未重复建节点）
- 一手来源：arXiv:2609.31394v1、官方项目页、官方 GitHub 仓库。
- 本次更新：重写浅层旧条目，补方法结构、训练期未来监督与推理边界、数据口径、仿真和真机结果、开源状态、限制及 Mermaid 训练/推理图；新增论文/项目页/仓库归档；连接 WAM 概念页。
- 状态校正：官方代码仓已发布训练、评测和部署代码（MIT）；权重仍未见发布并标记 forthcoming；处理数据受原始来源许可制约。
- 资料核验：2026-10-13。
- 未修改 Actions/工作流配置、生成目录或 catalog/export 文件。

# Ingest: eFlesh

- 日期：2026-10-13
- 新增：[eFlesh](../wiki/entities/eflesh.md)；论文、仓库及项目页来源归档。
- 更新：触觉感知概念页链接该磁性软触觉皮肤。
- 一手来源：<https://arxiv.org/abs/2506.09994>、<https://github.com/notvenky/eFlesh>、<https://e-flesh.com/>。

# Ingest: Motion2MixamoRig

- 日期：2026-10-13
- 新增：[Motion2MixamoRig](../wiki/entities/motion2mixamorig.md)；[来源归档](../sources/repos/motion2mixamorig.md)
- 更新：GVHMR 实体关联该视频/照片到 Mixamo 的工程化下游。
- 一手来源：<https://github.com/MaxLiu22/Motion2MixamoRig>；上游模型信息见 <https://github.com/zju3dv/GVHMR>。

# Ingest: EGalahad sim2real stack and leaderboard

- 日期：2026-10-13
- 新增：sim2real deployment/runtime 与 Motion Tracking Leaderboard 总览；补齐 SONIC 与 GRIT v0.0.1 项目实体。
- 映射：15 个 leaderboard policy rows → 10 个唯一项目家族节点。版本变体共用项目实体，不重复建页。
- 更新：Sim2Real 概念页及排行榜涉及的既有项目实体添加交叉链接。
- 一手来源：<https://github.com/EGalahad/sim2real>、<https://egalahad.github.io/sim2real/>、<https://egalahad.github.io/sim2real/leaderboard/>、<https://nvlabs.github.io/GEAR-SONIC/>、<https://github.com/mrzuang/GRIT_teleop_deploy>。

# Ingest — Test-Time Compute for Humanoid Behavior Foundation Models

- **日期：** 2026-10-16
- **意图：** 收录 TTCO 论文、官方项目页与运行代码；在 BFM 适应分类中加入测试时命令优化路线。
- **源码核验：** 项目页直链官方 GitHub；主代码 Apache-2.0，SPIDER sampler 保留 CC BY-NC 4.0。代码可运行，但需另外准备 HEFT checkpoint 与 G1 机器人资产；未发现项目专属权重或独立数据集下载页。
- **关键页：** [TTCO 论文实体](../wiki/entities/paper-ttco-test-time-compute-humanoid-bfm.md)；[BFM 概念页](../wiki/concepts/behavior-foundation-model.md)。

2026-10-16 ingest — 归档 UVTA 论文、项目页、MIT 代码与 Hugging Face 数据集；建立跨本体视触觉动作建模实体，并关联触觉感知知识页。

# Ingest: TRACE 跨具身人形控制迁移

- **意图：** 收录 TRACE（arXiv:2610.17999），归纳其将 SONIC 从 Unitree G1 迁移到 H2 的预对齐、动作筛选、蒸馏和低秩强化学习流程。
- **开源结论：** arXiv 指向官方项目页，但本次无法访问；未发现可确认的代码仓库，开放状态记为待核实。
- **关键页面：** [TRACE 实体页](../wiki/entities/paper-trace-cross-embodiment-bfm.md)、[论文来源](../sources/papers/trace_arxiv_2610_17999.md)、[项目页核查](../sources/sites/trace-bfm-project.md)。

---
date: 2026-10-18
op: ingest
---

将 DR-WM 论文与官方项目页归档，并整理到动作后果预测专题。官方项目页目前将代码和数据标为 coming soon，暂无可运行实现。

- 2026-10-19 | ingest | Joshua Bird 的 Low Cost Mocap 项目：确认 MIT 开源仓库含相机处理、ESP32 固件和 3D 文件；补充项目来源与动捕实体页，区分约 20 美元相机系统与约 50 美元无人机成本。

# 2026-10-19 ORA ingest

- **Op：** ingest
- **意图：** 收录 ORA 官方博客及其 robot-use agent harness，沉淀方法、仿真结果与部署边界。
- **开源核查：** 官方仓库已公开，Apache-2.0；主量化结果来自仿真，真机内容为初步演示。
- **关键页：** [ORA wiki 实体](../wiki/entities/ora-open-robot-use-agents.md)；关联更新 [LLM 机器人控制接口](../wiki/concepts/llm-robotics-control-interfaces.md)。

- **2026-10-20** — **ingest**：新增 Embodiment-Aware Inference-Time Search 论文来源与实体页，整理冻结 ReMoGen 的候选搜索、G1 可行性评测及代码核验边界，并关联动作重定向概念页。

# [2026-10-20] ingest | PointZero NeurIPS 2026 状态更新

- **意图：** 复核已收录的 PointZero 官方项目页与代码仓库，补充发表信息并厘清 robot-free 仅指预训练阶段。
- **开源结论：** 官方代码仓库 README 仍标注 Release coming soon；项目页数据集按钮仍为 Coming Soon。
- **关键页：** [paper-pointzero](../wiki/entities/paper-pointzero.md)

---
date: 2026-10-21
op: ingest
---

归档 Viggle JST-2 官方发布与评测资料，新增世界模型实体页，并从 Marble 页面建立交叉链接；记录模型/API 的闭源状态及其与机器人仿真之间尚未验证的边界。

---
date: 2026-10-23
op: ingest
---

收录 huashu-art-motion 编码代理动画 Skill。核对公开仓库 README、SKILL 与许可说明；记录其面向技术解释和传播的用途，以及代码动画不能替代机器人仿真验证的边界。

## 2026-10-24 | ingest

- **意图：** 收录 Datawhale Diy-LLM 中文课程与 Stanford CS336 Language Modeling from Scratch 的课程站、在线/离线阅读和官方讲义仓库入口。
- **开源结论：** 两个项目提供公开课程材料；Diy-LLM 最新 Release 提供 PDF，Stanford 讲义仓库包含可执行 Python 讲座与 PDF。仓库元数据未返回许可证信息，不推断授权条款。
- **关键页：** sources/courses/、sources/repos/、sources/sites/ 下的 Diy-LLM 与 Stanford CS336 归档。

# Ingest: Phoenix-MWA

意图：归档 Anyverse Dynamics 与中科院自动化所强化学习团队的机器人操作项目。
开源结论：官方预训练代码和模型入口已公开；训练数据/权重资源与部分后训练、数据处理和 RTC 部署代码存在额外准备或待发布边界。
关键页：Phoenix 项目页、官方代码仓库与 WAM 知识节点。

# [2026-10-26] ingest

- **意图：** 收录 STAMP 人形动作片段时空对齐与人体动作先验论文。
- **开源结论：** 项目页已核查；页面没有提供可核实的公开代码仓库链接，记录为源码链接未核实。
- **关键页：** `sources/papers/robot_motion_transition_human_motion_prior_arxiv_2610_30817.md`；`wiki/entities/paper-stamp.md`。

# 2026-11-08 ingest｜DINOv3 与世界模型技术谱系

按主论文核对 DINOv3、GAWM、DUET-DINO、PointWorld、FlowWM、RLA-WM、JEPA-WMs；DINO-WM 作为使用 DINOv2 的历史对照。复用已有实体，追加缺少的原文/代码/项目页归档。原文和源码尚未全部通读；PR 保持 Draft 直至复核通过。

# [2026-11-08] ingest | Recova 既有节点原文复核与真机编排补全

- **去重：** `wiki/entities/paper-recova.md` 已存在且 `project_id: recova`；本次不新建重复实体。
- **官方原文：** https://arxiv.org/html/2610.01178v1 （论文 §1–5、附录 A–E、Tables 1–5、Figures 1–10）；https://research.nvidia.com/labs/gear/recova/
- **更新节点：** `wiki/entities/paper-recova.md`，补 Appendix A Algorithm 1 对应的 Mermaid 真机调度时序（**论文伪代码，不是源码时序**）、四任务逐项 20-trial 成功率、异步 VLM 判定与过期回调防护、训练动作 offset/absolute 语义、技能计数原文差异。
- **更新原文归档：** `sources/papers/recova_arxiv_2610_01178.md`，逐条记录对照章节、实验统计和未开放源码边界。
- **数据核验：** Base 23.8% → DAgger 77.5% → 恢复 87.5%；摸牌任务四轮人工接管 7/8 → 0/7；不能将后者推广到所有真机任务。
- **论文内部统计差异：** §4.2 报告 25 recovery skills；附录 D 图 8–9 展示 24 项。本次不推测缺少的一项。
- **健康度目标：** 沿用原节点 `type/project_id/tags/status/summary/original/sources` 和九节阅读骨架，`original` 指向标有「通读：已完成」的现有来源；按仓库三项静态检查预期 3/3，需 Actions 最终复核。
- **发布边界：** 没有找到官方独立完整源码/权重/数据下载入口，不虚构源码调用或复现结果；不更新自动生成的 `catalog.md`、`log.md` 或站点导出物。

# 2026-11-08 ingest — ResearchBoost-Agent：可审计的 AI 训练 Infra 优化

- **意图：** 根据作者的《ResearchBoost: RSI starts with Infra》、官方 README/Methodology/Judge 及可运行脚本，收录“先冻结裁判再由 Agent 持续提效”的技术路线，并为机器人训练工程提供辨析。
- **材料：** `sources/blogs/researchboost-rsi-starts-with-infra-2026.md`；`sources/repos/researchboost-agent.md`。
- **主节点：** [ResearchBoost-Agent 双语深读](../wiki/entities/researchboost-agent.md)（一个项目一个实体，不为 Motus/NanoChat 重复建立 ResearchBoost 主节点）。
- **开源/源码结论：** GitHub 仓库公开了实际运行入口及评测组件；本次查到的根目录未列 `LICENSE`，未擅自认定宽松代码再分发许可。
- **核心核验：** 2026-11-08 官方 README 累计 20 项、7 个领域，9.59× 仅是 8×B200 上 Faster-Motus 7.468→0.779 秒/迭代；NanoChat 1.0744× 在 8×H20 的 FP8 训练更新中测得。必须区分 step time、estimated eval wall-clock 与完整训练。
- **流程可溯源：** 冻结数值 baseline σ → 逐轮 profile/time/check → FINAL.json → `judge.sh` 最终 B-C-B `check --final` → `accept.py` PASS/收益/哈希校验。节点含 Mermaid 项目流程图和真实源码入口时序图。
- **边界：** 尚未独立 GPU 复现；短轨迹验证、bubblewrap 回退安全和商业授权均已单独标注。研究 Infra 速度提升不等于完整 RSI。
- **图谱入链：** [递归自改进 RSI](../wiki/concepts/recursive-self-improvement.md) 添加反向入口。
- **变更边界：** 仅改 wiki、sources 与 log.d；未触碰 `catalog.md`、`log.md`、导出/站点统计文件。

# 2026-11-08 · Odyssey-3 官方首发原文深读与去重更新

- **Op：** ingest
- **意图：** 根据用户提供的 Odyssey @odysseyml 首发截图查找一手资料，避免与已有 Odyssey-3 项目节点重复。
- **原文：** 2026-09-15 [Introducing Odyssey-3](https://odyssey.systems/introducing-odyssey-3)；2026-10-08 [Meet Odyssey-3](https://odyssey.systems/meet-odyssey-3)。两份官方文章已逐节核对并标注阅读范围。
- **开源结论：** 官网有交互研究预览与开发者入口；未发现可公开下载的 Odyssey-3 权重、训练/推理源码和完整数据；不绘制虚构的源码时序图。
- **关键差异：** 共享的是世界模型的视觉表征，不是一个零适配的统一执行器；驾驶 20 小时为模拟训练，77% 是安全员接管之间行驶距离比，不是任务成功率。
- **节点：** [Odyssey-3](../wiki/entities/odyssey-3.md)；[首发归档](../sources/sites/odyssey-3-introducing.md)；[后续发布归档](../sources/sites/odyssey-3.md)。

# 2026-11-10 · ingest · FLORA 原文深读与开源状态修正

- **意图：** 用户指定 Sicong Leng、Yunze Man、Zhiqi Li 等的 `FLORA: An Open Model Suite for Scaling Robotics Data Engines`（arXiv:2610.31779v1）进行 ingest。
- **去重：** 保留已有唯一实体 `wiki/entities/flora-robotics-data-engine.md`；论文、项目主页和源码均挂到此实体，不创建重复论文节点。
- **原文：** 核对 arXiv v1 正文与 Appendix A–H，补分段/字幕 GRPO、Rewarder SFT、不同测试协议及成本与生产吞吐的估计口径。
- **开放状态：** NVIDIA 官方 Code 链接现指向 `NVlabs/Flora`；开源推理示例与三类模型权重，**不含训练代码/生产服务**。补 `sources/repos/flora-nvlabs.md` 源码实际入口与 `sequenceDiagram`。
- **输出：** 完整中英标题、方法流程图、对照表、边界/许可说明；同步更新 paper / site / repo 归档。
- **未复现：** 未下载 130 GB / 56 GB 权重，也未在 H100 或 512×GB300 集群运行实验。
- **原文：** <https://arxiv.org/html/2610.31779v1>
- **项目：** <https://research.nvidia.com/labs/gear/flora/>
- **代码：** <https://github.com/NVlabs/Flora>

# 2026-11-10 · ingest · RoboLineage（CoRL 2026）

- **意图：** 根据用户指定的《RoboLineage: Agent-Native Data Lifecycle Governance Across Robot Policy Iterations》（Qian Luo 等；arXiv:2606.22142）收录跨机器人策略轮次的数据生命周期治理。
- **去重：** 检索 Robotics_Notebooks 主分支代码与 PR，未发现同名实体；论文/官网/代码统一到 `wiki/entities/paper-robolineage.md` 一个稳定 `project_id: robolineage` 节点。
- **原文：** 通读 arXiv HTML §1–5 和 Appendix A–K，保留 25 tasks / 500 rollouts、12 对 session、八任务匹配训练和三轮复采的口径，并区分人工活跃时间与物理采集。
- **开放状态：** 官方 GitHub MIT 许可，已提供 Python runtime、ROS2 数据采集与 VSA、review、dataset lock、训练/评测适配、FastAPI/Vue frontend；mock replay、打包评分脚本和更广覆盖 adapter 尚在 README 待办。
- **输出：** 中英双标题、原始资料三层归档、方法 Mermaid flowchart、基于 `run.sh` / `UnifiedRuntime` / `PostRolloutReviewWorker` / `TrainingLifecycleRunner` / `PolicyEvaluationAgent` 的 Mermaid sequenceDiagram，并从数据飞轮已有页面入链。
- **核验边界：** 静态阅读官方源码与文档；未在真机、GPU 或 ROS2 完整 workstation 运行，不声称复现论文中定量结果。
- **论文：** <https://arxiv.org/html/2606.22142v1>
- **项目：** <https://robolineage.github.io/>
- **代码：** <https://github.com/robolineage/robolineage>

# 2026-11-11 ingest：LHNav / DASH / VLA Adaptation Chess / ExoStack

- 意图：为长时程人形导航、异步 VLA 控制、VLA 具身适配实验及 ROS 2 外骨骼软件栈建立四个独立实体节点；保留英文题名与中文机制摘要。
- 一手来源：arXiv:2611.12136、2611.11854、2611.12676、2611.12306；LHNav 项目站、Chess 官方代码/LeRobot 数据及 ExoStack 官方实现与技术文档。
- 关键结论：LHNav 现有仓仅展示网页；DASH 未核得官方代码；Chess 公开 MuJoCo/训练/评测和 261 条真实轨迹；ExoStack 公开 ROS 2/micro-ROS 代码但 PolyForm Noncommercial 限非商用。
- 边界：结果为原作者报告；未在本环境完成硬件或完整 CI 复现，待 PR GitHub Actions 检查。

# 2026-11-11 ingest | Ai2 ToyBlocks 原始研究深化与 Sim2Real 边界核实

- **意图：** 响应 ToyBlocks 单项目 ingest，发现 2026-11-09 已由 PR #3551 建立主节点，因而选择更新现有唯一实体、补充定量实验与真实源码执行链，避免重建同项目节点。
- **原文：** https://huggingface.co/blog/allenai/toyblocks （2026-11-06）；https://allenai.github.io/ToyBlocks/experiments.html；https://allenai.github.io/ToyBlocks/sim2real.html；https://github.com/allenai/ToyBlocks
- **开源结论：** 仿真/数据生成、模型训练/评测、官方模型与数据已公开；FR3 真机执行工具包与 system identification 指南截至本次核查仍标注 Coming soon，完整一体化 RL 训练尚待补充。
- **主要更新：** VLA 空间泛化的五种对照方法、智能体与世界模型定量结果、FR3 真机与仿真分母区分、代码核验的 datagen 与 eval 时序图。
- **主节点：** [ToyBlocks](../wiki/entities/toyblocks.md)
- **核验边界：** 通读官方发布原文与实验/Sim2Real 页面，核对运行入口代码；未执行 Isaac、GPU 训练或真机部署。

# 2026-11-11 — ingest LIFE：Learning to Live

- 来源：arXiv:2611.00180v2（2026-11-04）、AgibotTech/LIFE 官方 README、RoboBeats `gate.py`、SSS-Bench `pipeline.py` / `agent.py`，项目页以及前作 ReSight arXiv:2610.33487。
- 唯一主实体：[`wiki/entities/paper-life-learning-to-live.md`](../wiki/entities/paper-life-learning-to-live.md)。
- 区分论文完整系统主张（RoboBeats / Plastic Memory / EDM）与公开可运行部分（门控推理、模型适配器、SSS-Bench）。
- 按源码绘制因果事件流图和 `Gate.stream` → `select` → writer / reader 的 Mermaid 运行时序图；解释 100/10/90 分块、0.5/0.3 滞回、提问截断与 hard10 单次最多 10 帧回看。
- 从既有 Agent 记忆 wiki 建反向链接；未改动 catalog/log/export 派生文件。
- 论文数字属作者报告，未进行 CUDA benchmark 与实机复现；项目主页正文抓取未完成，具体边界记录在 sources/sites。

# 2026-11-12｜Ingest ReACT、RedirectMimic 与 BEHAVIOR 2026

- **意图：** 将三项 2026 年 11 月新研究编译进机器人知识库，区分人形非对称负载移动、全身动作局部编辑与百任务家务 VLA。
- **论文：** [ReACT](https://arxiv.org/abs/2611.13929)、[RedirectMimic](https://arxiv.org/abs/2611.14049)、[One Policy for 100 Household Tasks](https://arxiv.org/abs/2611.14432)。
- **核对与开放范围：** ReACT 和 RedirectMimic 当前未确认可运行作者源码；BEHAVIOR 提供 Apache-2.0 仓库、可下载权重与示范数据，但完整 agent 练习会话未公开。
- **主要结论：** ReACT 的 G1 导航 40/40 完成不等于全部运动误差更低；RedirectMimic 仅对局部动作奖励参考做删改；BEHAVIOR 纯 VLA 本地 Q-score 0.392，公开榜混合方案为 0.446。
- **编译形式：** 3 份论文通读归档、2 份项目站点归档、1 份官方源码归档、3 个含 Mermaid 方法流程的中英双语实体页；可运行 BEHAVIOR 源码另附时序图；回链挂至人形移动与 VLA 方法页。
- **说明：** 本次仅基于一手论文和公开源码静态核查，尚未在本环境执行本地仿真或全仓预检。

# 2026-11-12 · Glen Berseth CoRL 2026 Reading List（全文 ingest）

- **来源：** 博客全文已取得并通读（`sources/blogs/glen-berseth-corl-2026-reading-list-2026-11-10.md`，原文落盘 `sources/raw/`）；101 篇（P0 26 / P1 45 / P2 30）。
- **节点：** 101 篇逐条对应唯一详情节点：新建与升级的节点逐篇读论文并核对清单摘要；2 篇全文拿不到（FlashRFCL、Emergent Control via Prior-Based Free Energy Minimization），节点为 draft、未达 3/3。
- **概述页：** `wiki/overview/corl-2026-glen-berseth-reading-list.md` 重写为 101 篇索引，含跨论文观察与局限。
- **说明：** arXiv 号由标题精确匹配搜得；OpenReview 被验证页挡住，未逐篇核对；会场标注取自清单。

# 2026-11-13 ingest：SuperMimic 获奖更新与 RoboCap LiDAR-100

- **意图：** 为既有 SuperMimic 实体补充 CoRL 2026 获奖动态，另收录 BitRobot 真实作业头戴 LiDAR 数据集，不新建 SuperMimic 重复节点。
- **核验：** 原文以 SuperMimic arXiv v2、作者项目页、现有已读代码归档，以及 BitRobot Hugging Face 数据卡为主。获奖由 11 月 12 日媒体报道核对，用户提供的 Berkeley AI X 原帖受限，标记待直接复核。
- **开源边界：** SuperMimic 研究团队有训练/回放源码但完整实验资产未知；RoboCap LiDAR-100 是非商业许可数据集，下载须接受条件，目前 50 / 100 小时开放，没有独立可运行策略训练仓库。
- **涉及页面：** wiki/entities/paper-supermimic.md 与 wiki/entities/dataset-robocap-lidar-100.md；另给 wiki/entities/paper-robomocap.md 添加区别说明与反向链接。

# 2026-11-13 ingest｜VINO 与 Volt：3D 场景自监督与体积 Transformer

- **意图：** 将社群所称“3D DINO 时刻”的 VINO 表征学习工作与 ECCV 2026 Long Oral 的 Volt 骨干分开归档，避免把两篇论文及不同的训练协议合并。
- **原文：** arXiv:2611.07176v1（含补充材料）与 arXiv:2604.19609v2；对照两个作者项目主页与官方 GitHub/Volt 训练代码。
- **开源结论：** Volt 官方代码、训练脚本和 HF 权重已发布；VINO 仓库只有 README 占位，模型权重和完整代码待发布。
- **知识沉淀：** [VINO](../wiki/entities/paper-vino.md) 与 [Volt](../wiki/entities/paper-volt.md)，并补充具身感知三维表征的交叉引用。

# 2026-11-13 ingest：Mundo 相机 Plücker 射线与可编程仿真技术精读

- **意图：** 对已有 Mundo 节点增量深入整理，避免重复实体；厘清 Scene Program、MuJoCo、Proxy Video、Plücker Camera Encoder 与视频扩散 Transformer 的职责分界。
- **原始资料：** 2026-10-23 作者官方技术博客全文、项目主页全部技术板块、商汤 2026-11-04 官方新闻、Luo-Yihang/Mundo 仓库 README；数学定义参考 ICLR 2024 Cameras as Rays。
- **核心修订：** 增加 Plücker 射线定义与相机坐标约定、注入位置、无位姿参考图的 learned embedding；改绘项目 Mermaid 方法图，补输入/仿真/训练/性能边界。
- **开放结论：** 2026-11-13 官方仓库只有 README 与 assets，README 将代码/权重列为 11 月底待发布；不绘制虚构源码运行时序图。
- **质量：** 为原文归档补显式通读记录，wiki 增加 original 和统一阅读骨架；仓库完整预检需以 GitHub Actions 为准。

# [2026-11-13] ingest | 星尘智能 CoMind / Be My Eye

- **操作：** ingest。
- **意图：** 收录星尘智能 CoMind 项目页及 arXiv:2611.12295v1 论文，准确组织双机器人受限通信下的任务相关上下文共享，并对比零通信 CHORUS。
- **已核对：** arXiv HTML v1 §1–6 方法、公式、Tables 1–5、定性实验与限制；官网 CoMind 任务描述与公开资源入口。
- **新增主实体：** [CoMind / Be My Eye](../wiki/entities/paper-be-my-eye-comind.md)；包含双语标题、机制、论文方法 Mermaid 流程图、数值证据和开源边界。
- **来源：** [论文](../sources/papers/be-my-eye-arxiv-2611-12295.md)、[官网](../sources/sites/astribot-comind-project.md)、[官方代码核验](../sources/repos/astribot-comind-code-status.md)。
- **开源判定：** 截至本日官网未公开该方法可运行代码、权重与数据；不虚构源码时序图。
- **关联：** 新主实体引用 [CHORUS](../wiki/entities/paper-chorus.md) 作零通信对照；[人形多机协调](../wiki/concepts/humanoid-multi-robot-coordination.md) 增加 CoMind 入链。
- **通读补齐：** PR #3699 复核中逐篇通读 Swarm Sensors 2025、SPL 2401.15026v1、ARTEMIS 2512.09431v1、RoboCompiler-Humanoids 2610.14237v1 及 CoMind 2611.12295v1；前四篇集中为[一手论文全文核验记录](../sources/papers/humanoid-coordination-fulltext-review-2026-11-13.md)，协调概念页以 `original` 关联该记录与 CoMind 归档。
- **纠错：** 区分单机内管理与队级集中控制、SPL 定性赛事与 60 min 仿真、Swarm 控球百分点与缺员/丢包口径；CoMind 图中未来帧改为独立训练目标，补充 Process Score 权重定义。

# 2026-11-13 ingest：Flash ALOHA（高速双臂动觉示教）

- **Op：** ingest
- **意图：** 为机器人学习知识库归档 Stanford Flash ALOHA 的原始论文、项目页与源码，并写入独立、可导航的论文详情节点。
- **查重：** 已用 Flash ALOHA、flash-aloha 与 ALOHA Lightning 查询当前仓库内容和 PR；未发现同论文专属详情页，新建唯一节点 `wiki/entities/paper-flash-aloha.md`。
- **原文核验：** 官方论文 PDF 12 页（方法、Table I–III、附录）与官方站点，来源完整记录于 `sources/papers/flash_aloha_icra2026.md`。
- **开源结论：** 官方 GitHub 已开放 MIT 许可的采集、训练及真机推理框架；未确认原论文完整五任务示范与检查点公开下载；新仓库 Flow/RTC 能力不可反推为论文主结果。
- **关键技术：** 可反驱手持示教、编码器采集轨迹、基于运动学的操作员手部图像遮罩、ACT 动作块重叠 Sigmoid 平滑。节点内含原文流程 Mermaid 和结合真实源文件入口的运行时序 Mermaid。
- **评测读法：** 各 50 条示范、五项成功率 80%–100%；电池插入 5.70 s vs 主从 17.87 s；抛接球 1.85 s。把速度、数据采集与任务成功率的因果边界分开。
- **关联入口：** `wiki/tasks/bimanual-manipulation.md`，避免孤立页。
- **源码边界：** 已追踪 `scripts/run.py` → 采集/训练/推理入口及 `blend_actions`；论文 FK+相机标定手部遮罩的完整函数仍未定位，不宣称源码完全复核。
- **校验：** 当前 GitHub 连接仅允许读写线上仓库，无法在此运行 `make ci-preflight` 完整本地检查；最终以 GitHub Actions 的 Wiki Lint / Mermaid 检查为准。

# [2026-11-13] ingest | FlowDPG CoRL 2026

- **意图：** 把 Skild AI/CMU/UPenn FlowDPG 的长期双臂策略优化路线收录为唯一论文节点。
- **原文核验：** arXiv:2606.22303v2 正文与 Appendix A–D，官网 PDF、任务/扰动评测。
- **交叉链接：** 双臂协调操作任务；正文增加基于论文 Algorithm 1 的 Mermaid 主干流程图。
- **开源结论：** 项目页未列正式源码/权重/数据下载，因此不生成未经核对的运行时序图。
- **指标纠正：** Franka AirPods 92% 为端到端成功率；YAM 炒蛋 85.8% 为离线阶段评分；不混入 PNP 营销信息。

# [2026-11-14] ingest | sources/blogs/embodied-research-digest-2026-11-12.md — 13 篇论文去重收录

- 逐页分析用户上传的 5 页研究资讯，确定 13 篇论文的 arXiv 来源。
- 复用主分支已有 MetaWAM、BEHAVIOR 2026、DexSeed、Elastic-WAM 四个节点。
- 为其余九篇创建独立论文实体、来源与源码/项目归档、Mermaid 方法图，以及可由 README 对齐的概念时序图。
- **开源与健康边界：** 后续已核对九篇原文及相关附录、追加实验约束与源码入口；知识节点健康度按规则升级，源码与机器人实测仍未独立复现。URDF2Ontology 原站 PDF 获取受限，已明确标注使用论文文字镜像。

# 2026-11-14 ingest：2026 VLA 15 项目开源对比

- **意图：** 将用户附件《15 个开源项目，看透 2026 年 VLA 的三个关键转变》（深蓝具身智能，25 页）整理为 2026 开源生态选型对照，衔接 2025 VLA 资源索引。
- **入库：** 二手报道来源导读 1 份；按论文/项目唯一身份升级 10 个既有 wiki 实体，并新增 5 个独立 wiki 实体（TurboVLA、Hy-Embodied-0.5-VLA、StarVLA 框架论文、vla-evaluation-harness、verl-vla）；新增 9 个一手原文阅读归档与补充官方源码导航；新增跨项目 comparison 1 页。
- **核验：** 当前官方 `allenai` 仓库纠正评测工具归属；Qwen-VLA 只披露研究结果、尚未找到完整开放代码/权重；区分 TurboVLA 单次推理 32 Hz 与整环频率；区分 Hy-VLA 10k+ h 总数据与 2k+ h 公开数据。
- **边界：** 未重新通读 15 篇独立原始论文/源码，未运行 15 套基准。各实体只指向对应论文或项目的一手归档；StarVLA 框架和 α 为两篇不同论文，也不更改自动生成的目录/站点导出物。

# 2026-11-14 · ingest · 紧密螺旋橄榄球投掷

- **意图：** 收录爱荷华州立大学 Zaid Mahboob / Bowen Weng 关于 Unitree G1 精确旋转出球的 Humanoids 2026 论文，形成跨动态操作与接触控制的知识导航。
- **原文：** https://arxiv.org/html/2608.16642v2（已核对 §I–§V 方法、实验、局限）；作者项目页 https://www.zaidmahboob.com/projects/tight-spiral/
- **关键结论：** 两阶段抛投/随动；MPC 采样优化求解一次约 500 ms，真机使用离线 52 维查表；最高旋转效率 93.6%、最佳球头角 0.286 rad；不能误读为在线 250 Hz MPC 求解。
- **开放程度：** 论文和演示公开视频；未查到该论文的官方可运行控制器仓库及查表数据，故源码运行时序图明确不适用。
- **知识落点：** 建立唯一论文 wiki 实体、论文与项目页 sources 归档，加入 WBC 概念入链；包含原文数据/控制流程 Mermaid。
- **验证边界：** 按仓库来源与阅读骨架静态核对；通过 GitHub 连接器提交，当前容器无法访问 GitHub，未本地执行全量 make ci-preflight 或真机复现。

# 位置编码教程 ingest 记录

# 2026-11-16 | ingest | Agentic Robots 第一批独立节点

- 从官方固定清单快照建立 307 条批次台账，编号匹配仅作候选，后续继续语义查重。
- 完成 PAFF 独立详情页：通读 v4，区分评测协议，静态核查 CALVIN 代码及默认重标注阈值，补充流程图和运行时序图。
- 综述建立入链，VLA Stage 5 建立阅读入口。代码未全仓审计，实验未复现，其余条目仍待逐项验收。

# 2026-11-16 | ingest | Agentic Robots 第二批

- 完成 TidyBot、OLAF、LMPC 三个独立论文节点；先按编号、标题、官方入口查重，区分 TidyBot2 与 Disney Olaf。
- 核读固定版本正文与相关附录；核查 TidyBot 官方实现，获取并静态阅读 LMPC 官方教学 Colab，记录 OLAF 空 Code 链接。
- 三类适应分别作用于语言规则、视觉运动策略与语言模型。建立综述入链和 VLA Stage 5 导航。
- 台账总计 307，累计完成 4，剩余 303；实验均未运行，节点健康度以脚本为准。

# 2026-11-16 | ingest | Agentic Robots stub 与地址覆盖

- 用户调整为先建 stub，不以健康度为本阶段门槛。
- 补入此前漏计的 additional_papers.yaml 23 条，范围从 307 修正为 330。
- 新建 229 个 stub，复用 101 个既有节点，节点缺失 0；项目入口未确认与无 arXiv 编号分别列出，未猜测地址。
- 深读仍为 4 条，剩余 326 条；本次地址提取不冒充论文通读或复现。

# 2026-11-16 | ingest | Towards Agentic Robots v1 分类补核

- **意图：** 论文节点已存在，不重复建页；依据原文改善摘要式旧页，聚焦 Table 1 四域十七类、执行编排与评测边界。
- **一手材料：** [固定 v1 HTML](https://arxiv.org/html/2610.20442v1)、[固定 v1 PDF](https://arxiv.org/pdf/2610.20442v1) 与 [TeX 源包](https://arxiv.org/e-print/2610.20442v1)；PDF 第 10 页 Table 1 原表已核。
- **开放边界：** [作者 Awesome 索引](https://github.com/chonghanqin/Awesome-Agentic-Robots) 是动态文献资料库及生成/校验脚本，并非机器人训练/推理方法代码。
- **主要位置：** [唯一论文实体](../wiki/entities/paper-towards-agentic-robots-survey.md)、[v1 来源归档](../sources/papers/towards_agentic_robots_arxiv_2610_20442.md)、[VLA Stage 4](../roadmap/depth-vla.md)。
- **待完成：** e-print 源包在当前环境无法解包，全部引用未逐条审计；故来源归档为「通读：未完成」、实体保持 draft；健康度需以脚本实测为准。

# 2026-11-16 · ingest

- 完整读取具身运控lab《11.09–11.13 人形／四足 29 篇》原始上传 PDF，建立 29 篇独立论文阅读坐标。
- 以 arXiv 主身份去重：复用 15 篇已完成论文详情；升级已有 BlinkMimic 草稿 1 篇；新建 13 篇独立论文节点及其一手原文归档。
- 本次更新的 14 篇论文包含方法、实验、局限、Mermaid 主流程、双语标题、跨页面引用；BlinkMimic 与 FORGE 另基于实际可运行入口添加源码运行时序图。
- 本次 29 篇均有健康度 3/3 的结构化判据；最终以报告脚本与 CI 实测为准。
- [29 篇阅读坐标](../wiki/overview/humanoid_quadruped_2026_11_09_13_29_papers.md)

# 2026-11-18 · ingest ffmpeg-skill

- 将 kajisho5/ffmpeg-skill 按唯一非论文项目实体收录，归档 GitHub README/SKILL/关键源码与官方主页。
- 描述 42 个媒体处理工具、SPEC 从 argparse 派生的契约、MCP 默认展示核心 12 项及 probe→edit→check→verify 的工程闭环。
- 添加依据真实脚本入口的 Mermaid 数据流程图与 MCP 源码运行时序图；与 video-shotcraft 双向链接。
- 仓库 MIT 开源；源代码/文档已核对，未执行视频处理回归。该项目用于实验展示视频后处理而非机器人真机控制。

# 2026-11-18 · ingest Code World Model 2.0 / Invoker-Render

- 新增唯一项目实体 `wiki/entities/code-world-model-2-invoker-render.md`，并归档作者官网、GitHub 推理实现和 Hugging Face 权重三类一手资料。
- 记录 Proxy 对世界状态与相机控制、外观首帧、四步 DiT 因果 chunk、KV cache、8 GPU 重叠编解码及官方性能测量边界。
- 提供基于公开入口的 Mermaid 项目流程图、源码时序图；与 EdenLive 双向链接，并挂载 WAM 纵深路线的 Stage 1 世界基座对照。
- 仅核读官方 README、官网、模型卡、部署文档和入口源码；仓库的技术报告 PDF 尚未完整读取，训练方法与数据未开放，GPU 未实测。

# 2026-11-19 | ingest | 具身机器人 FDE 交付商业模式

- 原文：《FDE 火了，具身模型的交付会是一门什么样的生意？》（具身智能之心；用户提供全文）
- 新增博客归档、厂商一手来源归档、对比知识页与 Mermaid 交付流程。
- 用官方智元、破壳、Dexmal、PI、Dream Labs、Formic 与 SEC Agility 的公开资料区分已证实、厂商陈述和待验证。
- 关联既有 FDE 工程参考书页面；不创建重复企业项目实体。
- 验证边界：当前执行环境无法访问 GitHub 仓库网络克隆，尚未运行本地 make ci-preflight、健康度脚本或 Mermaid 解析器；以 GitHub Actions 为后续门禁，不声称已通过。

# [2026-11-19] ingest | AutoVLA-D 从推理学习轨迹几何条件

- **操作：** ingest
- **日期：** 2026-11-19
- **资料：** [AutoVLA-D arXiv:2611.12382](https://arxiv.org/abs/2611.12382) / [ShivaTeam/AutoVLA-D](https://github.com/ShivaTeam/AutoVLA-D)
- **意图：** 解释驾驶指令语义如何经离散局部运动 token 过渡到最终连续航点；明确 TrajCond 是中间条件而非最终离散输出。
- **结论：** 新增独立论文详情，论文和源码分层归档，Mermaid 方法图与核心代码运行时序图，关联 DriveTeach-VLA 和 VLA 纵深路线。记录论文分位区间与当前公开代码 min/max 的差异，以及 GRPO 在主仓之外的复现边界。

# [2026-11-19] ingest | NeuIDO 神经内禀动力学算子

- **操作：** ingest
- **日期：** 2026-11-19
- **意图：** 收录 ECCV 2026 NeuIDO；区分“共享弹塑性神经算子”与“无需物理先验”的错误解读。
- **一手资料：** [arXiv v1](https://arxiv.org/html/2609.24313v1)；[官方代码与项目页](https://github.com/JiajingLin/NeuIDO)。
- **开源结论：** 部分源码已公开；README 对权重、环境、数据生成、完整运行命令仍标 TODO；未实测。
- **关键节点：** [NeuIDO 论文详情](../wiki/entities/paper-neuido-neural-intrinsic-dynamics-operator.md)，并在 Real2Sim Stage 2 与可微仿真页建立入链。

# 2026-11-19 Ingest：HyWorldVLA 自动驾驶混合世界建模

- **操作：** ingest 论文 arXiv:2607.20988（Quanfu Yu、Xian Wu、Hao Xu、Liulong Ma，比亚迪汽车新技术研究院）。
- **一手资料：** https://arxiv.org/html/2607.20988v1
- **原文归档：** [sources/papers/hyworldvla_arxiv_2607_20988.md](../sources/papers/hyworldvla_arxiv_2607_20988.md)
- **唯一论文实体：** [wiki/entities/paper-hyworldvla.md](../wiki/entities/paper-hyworldvla.md)
- **核对：** 论文正文、实验表、附录失败案例与损失权重消融；把 NAVSIM v1 PDMS 与 v2 EPDMS 分开，并区分正常天气与 655 例雨雾噪声子集。
- **架构要点：** 预训练的像素与 latent 混合监督 → 联合微调阶段让未来 latent 条件化规划头；流程图只画原文方法，不冒充运行时源码图。
- **关联：** 在 S²-VLA 详情增加方法对照与入链；挂载至 VLA Stage 5「世界模型融合」。
- **开放性：** 论文未提供 HyWorldVLA 官方实现和权重链接，因此不编造运行时调用、训练日志或复现结论。
- **验证：** 待 GitHub Actions Wiki Lint / Tests / Search & Export Quality 检查；未在本运行环境完整 clone 仓库跑 make ci-preflight 或真实驾驶仿真。

# 2026-11-19 ingest — OpenAI Research Acceleration

- **意图：** 收录 OpenAI 官方 2026-11-18 `Research acceleration: The jagged frontier`；区别研究活动强度、任务零干预成功率与模型能力指标。
- **原文：** <https://openai.com/index/research-acceleration-jagged-frontier/>（官方研究报告；正文已核读；网页方法弹层及交互图表原始数据未全部导出）。
- **身份与开放边界：** 非 arXiv 单论文／非可运行代码项目；报告未公开可复现的内部数据与专用实现；无需虚构时序图。
- **关键节点：** 独立报告节点 [OpenAI 科研加速](../wiki/entities/openai-research-acceleration-jagged-frontier.md)，源记录 [官方报告归档](../sources/sites/openai-research-acceleration-jagged-frontier-2026.md)，与 RSI/Auto-Research 概念页交叉引用，挂载 RSI 纵深 Stage 5。
- **评估视角：** 人机混合研究闭环仍是现状；内部观察数据不能外推为独立因果或机器人研究测评。

## [2026-11-22] ingest | RoboWild | 复杂仿真家务的 VLM-RUA 评测

- **意图：** 将 XLANG Lab 的 RoboWild（38 场景、4480 交互物体、341 任务–本体组合）补入具身评测路线，厘清“最强 13.28%”不是实机成功率。
- **结果：** 新增独立论文详情、官网/README/源码核查归档、Mermaid 项目流与源码时序，挂载评测路线 Stage 3 与四层评测枢纽。
- **开源：** RoboWild + RoboWild-Harness 仿真与评价代码公开，Hugging Face 资产公开；未执行 GPU 复现。
- **原文核对：** 已通过官方 PDF 直链通读 2026-11-20 版 51 页论文（含附录 A–G）；确认 239 个等权 setup / 341 个本体评测实例、G1 在本文 ScaleBFM + Harness 方案下五模型均无成功、深度与 Coding Agent 消融结果；论文节点更新到 complete。
- **关键页：** [RoboWild](../wiki/entities/paper-robowild.md)

## [2026-11-22] ingest | comma.ai openpilot 工程栈

- 意图：归纳 openpilot、msgq、panda、opendbc 四个独立开源项目的真实模块边界与安全执行链。
- 开源与资料：四个 master 仓库公开，核对 README、核心执行入口与部分源码；未做实车/HIL 测试。
- 知识沉淀：四个实体页分别包含 Mermaid 数据流、源码运行时序和双语标题；挂载安全控制 Stage 4 工程对照案例。

## [2026-11-22] ingest | DTMR 论文深读与官方源码开放状态核实

- 对照 arXiv:2609.38617v1 全文，完善现有唯一实体节点；纠正 12.9% DTW 指标解释，区分 G1 全语料 1% 与 H1-2 9.5% 总时长变化。
- 项目页 Code 已正式指向 nicolejrkim/DTMR；核对 MuJoCo Warp GPU MPC 跟踪器、CLI、示例运行时序、配置和后处理指标，不将离线重定向误写为真机实时控制。
- 更新动作重定向纵深 Stage 3 关联，保留前端派生数据及 log.md 由部署流水线生成。

## [2026-11-22] ingest | sources/papers/pgx_arxiv_2611_26596.md — 收录 PGX 物理接地长时程人形移动操作

- 原文 arXiv:2611.26596v1 与官方项目页已核对，独立实体及 Mermaid 框架流程图已入库。
- 核心是谓词 FSM 编排、VLA 操作技能、WCL 世界系稳定；同权重 GR00T 咖啡 C0 35%→80%。
- 截至入库日官方项目页未列 PGX 可运行代码或权重，源码运行时序图标明不适用。
- 挂载移动操作 / VLA 纵深路线及移动操作任务页。

## [2026-11-22] lint | 推进 #3997 P1-1 与 27 条非历史信息型预警清理：补齐 paper-robochallenge-zero venue 键、easy-quadruped 源码复核升级 3/3、新建 Chaos Loop 选型 Query、锚定 6 处时间敏感措辞、收敛 15 个 missing_concept_pages 词条并补齐 karpo-browserbench 评测闭环回链（其余 3 处 crosslink 留待 PR #4081）

## [2026-11-22] ingest | 实验室路线（docs/lab-roadmaps.json）全量 54 个节点提升至 3/3 健康度：完成 ETH RSL / OpenDriveLab / RAI Institute / LeCAR Lab / TeleAI 共 19 个缺项节点的一手原文与开源入口通读核对、阅读骨架与 Mermaid 图表补齐

## [2026-11-22] ingest | sources/repos/cyberbeast-robot-mc.md — CyberBeast 关节电机全软件栈（5 项目 + 总览）

- 录入 robot_mc、OpenBootloader、JointSDK、JointROS、canable2-fw-n32 五个独立开源项目，附官方源码链接、控制流程与运行时序 Mermaid 图。
- 归档 Motor Wizard 官方手册调试入口；EXE 文件大小和现时下载状态未作事实认定。
- 形成 firmware 40 kHz FOC → CAN/CAN-FD C99 SDK → ROS 2 1 kHz tick 的分层认知，区分 CAN Bootloader Classic 500 kbit/s 升级链与 SLCAN USB2CAN 诊断链。
- 注册 CyberBeast 所属机构，链接力矩电机与人形硬件纵深路线，节点健康度由 GitHub CI 核验。

## [2026-11-22] ingest | 完成全站 30 条纵深路线最终轮 Batch 3（contact-manipulation / sim2real / loco-manipulation / model-architecture / cross-embodiment / rsi 共 107 节点通过 arXiv 全文与 Agent-Reach/GitHub 一手资料核读升至 3/3，全库 26/30 路线全满分、1589/1593 节点达 3/3；仅 HUST/OpenReview/IEEE 3 个唯一外部受阻论文按实保留 2/3 并标注阻塞原因）

## [2026-11-22] ingest | Route Health Sweep Batch 2 (Cluster A+B): 补齐 73 个纵深节点一手原文通读与完整骨架（含 Agent-Reach 读取 Holland 1962 / Goldberg 1988 / Mordatch SCA 2012 / Zenodo 元数据），闭环 depth-safe-control (29/29)、depth-rocket-landing (38/38)、depth-rl-locomotion (49/49)、depth-humanoid-soccer (30/30)、depth-humanoid-swarm-performance (32/32)，depth-humanoid-hardware-design 推进至 57/58（paper-humanoid-leg-generative-design-dynamics 因 HUST 超时及 CNKI 验证码诚实保留 2/3）

## [2026-11-21] ingest | 批次1纵深节点满分升级：通读开放一手原文并补齐阅读骨架，关闭 depth-bfm (55/55)、depth-humanoid-boxing (22/22)、depth-classical-control (49/49) 等路线共 71 节点达 3/3，如实记录 3D-IC 与 TRAMP 外部 403/付费墙阻塞

## [2026-11-22] ingest | company-nodes-health — 按公司路线批量补全 165/167 个未满分节点一手原文核读、阅读骨架与 Mermaid 图表至 3/3（其余 2 项因反爬/验证码门禁保留未完成记录）

## [2026-11-21] ingest | 力矩电机设计纵深 42 节点（Clusters A-D）原文通读与阅读骨架补齐，roadmap/depth-torque-motor-design.md 直连 70 节点全部 3/3

## [2026-11-21] ingest | 完成导航纵深 Stage 1–5 剩余节点（Stage 1/2/3/4/5 共 33 页）原文通读与代码入口核验，roadmap/depth-navigation.md 达成 69/70 满分（仅 paper-3d-ic 因 OpenReview 403 阻断保留 2/3）

## [2026-11-21] ingest | 完成 Issue #3490 导航纵深第 1 批 12 篇 3/3 提升（补全 KILVO/iCrowdNav/CommNav 最新开源实现与时序图、补齐 PythonRobotics/ORB-SLAM3/Habitat-Sim/SocialNav 全文归档，并如实记录 3D-IC OpenReview 403 阻塞）

## [2026-11-21] ingest | 完成 Issue #3490 第 4 批 12 篇深度通读与 health 补全（roadmap/depth-bfm；更新 GigaBrain-WBC-0.5 与 ScaleBFM 开源代码入口及 MetaMotivo/MaskedMimic/PHC/Perceptive-BFM 等原文指标与时序图）

## [2026-11-21] lint | 推进 #3997 P1-2 治理第 3 批（收尾清零）：核对最后 13 篇 arXiv 原文并补齐 NSDL/PAFF/Passive-Soft/Physical-Recipe/PreNav/Robo-SWE/Robot-Weight/SO101/SPARC/TCP4D/VIM/VINE/VOKE 对比小节

## [2026-11-21] ingest | Issue #3490 Batch 3 纵深节点健康度推进：全文通读与官方仓库/项目页核查 13 篇（PackLab/LIBERO-VPro/Uranus/ManiSkill-HAB/Mimicking-Bench/Barkour/Physics-Consistent HRC/Real2Sim 2511.04665/2508.11117/WM Eval 2606.15032/Motion Difficulty 2512.07248/TrackerLab/humanoid-policy-reward-functions），补齐源码运行时序图、实验结论与 original 归档，depth-embodied-eval.md 提升至 63/67 3/3

## [2026-11-21] ingest | TimesFM 3.0 官方零样本时序预测基础模型

- 原文：Google Research 2026-08-31 技术博客、google-research/timesfm 官方源码、Connected Sheets + BigQuery ML 官方帮助。
- 知识：新增 `wiki/entities/timesfm-3.md`，说明 330M 参数、CPM、交替时间/变量注意力、多变量预测与 9 分位数；补 `wiki/methods/model-predictive-control.md` 边界对照。
- 校对：TimesFM 2.5 为 200M；3.0 源代码 Apache-2.0，但下载权重是非商业/非生产许可；Google Sheets 依托 Connected Sheets + BigQuery ML。
- 关联 PR：#4045。

## [2026-11-21] ingest | 推进 #3490 第二批（depth-embodied-eval）：通读 MMHU/WorldEcho/WorldScore/PRM-as-a-Judge/SoftVTBench/HumanTracker/Bet4Sim2Real/XPolicyLab/RoboSynChallenge/H2R-Bench/CALVIN/RLBench/HumanoidBench 与 Evo-SOTA 原文及官方仓，将 14 个实体节点健康度升至 3/3；复核发现 H2R-Bench 已补齐 benchmark 与 evaluation/、PRM-as-a-Judge 迁至 YuyangLiu2003 且已发 RoboPulse++、WorldEcho 仓库链接仍返回 404

## [2026-11-21] lint | 推进 #3997 P1-2 治理第 2 批：核对 14 篇 arXiv 原文并补齐 Alloy/ARTS/BAMOS/Bicycle/BWM-Forge/CrowdMAPS/DexRemo/Elastra/FLUX-mimic/GenGenIK/HALO/Hibikino/MOSAIC/MoViC 对比小节

## [2026-11-21] lint | 推进 #3997 P1 治理第 1 批：补齐 paper-agnes-sys1 来源键（P1-1 清零）及 WorldFlow/HOVER/FloMo/DepthGS/PROOF/Cosmos Policy/ReCogDrive/Alpamayo-R1/WorldEngine/SGDrive/Envisioning/Zero-Shot 等 12 篇原文核对对比小节

## [2026-11-21] ingest | 推进 #3490（perceptive locomotion 批次）：完成 AME-1/2、InfiniteDiffusion、FastStair、Explicit Stair Geometry、ANYmal/Humanoid Parkour、RPL、DAgger、ResNet、YOLO 原文通读核对，将 depth-perceptive-locomotion 15 个 wiki 节点健康度升至 3/3（该路线 41/44）

## [2026-11-21] ingest | RoboScope Kit：补齐 PyPI 复现链、上游新增适配与评测纵深导航

- 复用 `wiki/entities/paper-roboscope-kit.md` 唯一论文实体（arXiv:2611.03850），保留已核对的全文、Mermaid 方法图与源码时序图。
- 对照 MINT 官方 README 与包指南，补充 `roboscope-kit` 安装、Policy/manifest 接口、quick-test、run cache、12 个当前 benchmark 条目的范围与限制。
- 将后续开源扩展与论文 v1 的 StarVLA 性能数字严格分开，挂载至具身模型测评 Stage 6；未进行 GPU 仿真复现。

## [2026-11-21] ingest | Zane Hub #398 机器人结构静刚度与末端变形预算

- 来源：sources/blogs/wechat_zanehub_robot_component_stiffness_deformation_2026-11-21.md；用户提供全文，微信原页未取得。
- 新增：wiki/concepts/robot-structural-static-stiffness-deformation-check.md，连接人形硬件设计纵深 Stage 2。
- 方法：以 TCP 工艺允许偏差为起点，分配连杆/关节/结合面静变形预算，经过梁手算、FEA、雅可比柔度映射、实测闭环。
- 勘误：按文章给定数值，I=1.12×10⁶ mm⁴、L=800 mm 挠度≈0.10884 mm、L=1200 mm≈0.36735 mm，均与原文不符；ISO 9283 静柔顺度与定位重复性分开对待。
- 证据边界：行业经验阈值、减速器参数、结合面 30% 与补偿 50% 等未逐条验证，不作为通用验收标准。

## [2026-11-21] ingest | ReLoco arXiv:2611.23916 — 单策略人形跌倒恢复与命令续行

- 通读论文 v1 和项目主页，归档行为条件双头 AMP、仿真精炼动作、真机实验与硬件保护统计。
- 创建独立论文知识节点并接入 Balance Recovery 与人形 RL Locomotion 进阶路线。
- 官方项目页有 MuJoCo 交互演示；截至入库日未发现公开训练/部署源码和权重，未虚构运行时序图。

## [2026-11-21] ingest | sources/papers/robokit_corl2026_spotlight.md — 补录 RoboKit arXiv:2611.20185 标准论文身份

- **意图：** 用户提供 arXiv 编号后复用已经完整收录的 RoboKit 实体，避免重复页面。
- **核对：** arXiv v1 正文、方法、实验、限制与附录 A–C；之前作者 PDF 的全篇归档和源码时序图保持不变。
- **开源结论：** 官方 [openrobokit/robokit](https://github.com/openrobokit/robokit) 可公开获取；性能仅引用论文作者实验。
- **导航：** 更新 RoboKit 实体标准 arXiv ID，并接入动作重定向纵深路线 Stage 1。

## [2026-11-21] ingest | sources/sites/noitom-robotics.md — 增补诺亦腾机器人公司路线，串联 Adam-U、ModalityNet 与联合研究，记录数据与源码开放范围

## [2026-11-21] ingest | sources/blogs/embodied_heart_itw_ego_plus_learnability_2026_11_20.md — 可学习性报道与 ITW-Ego Plus 五种选配模态一手核验

- 复用已有 ITW-Ego Plus 实体，补齐 Sol/Jupiter 模态流程、训练场景配方与 Mermaid 图，避免重复造页。
- 基于 2026-11-05 诺亦腾公告和官方 ModalityNet 技术文章校准 Body/Wrist/Exo/Touch/Sound；4D 重建仅部分 ITW 数据明确涉及。
- 当前未核验 Plus 原始样本、许可、准确时序指标及独立下游消融，节点继续保持 draft；挂载具身数据 Stage 1。

## [2026-11-21] ingest | SimReady Central 官方资料复核与资产质量边界

- 核对 2026-10-21 NVIDIA/Hugging Face 公告、2026-11-19 SimReady Central 官方文档、资产数据集和 GitHub 源码/验收规范。
- 复用已有 `wiki/entities/nvidia-simready-central.md`，更新 2,919 个 USD 资产、近月 1,731,953 下载及 CC BY 4.0 数据集许可口径；添加官方目录构建代码时序图和物理校验清单。
- 区分 ABB RobotStudio HyperReality 99% 宣传指标与 SimReady Central 资产库；未证实的趋势排名、3 吨货架与抽屉驱动传闻不当作事实。
- 同步更新 `sources/sites/`、`sources/repos/`、`sources/datasets/` 归档，并挂载 `roadmap/depth-real2sim.md` Stage 2。

## [2026-11-20] ingest | 推进 #3490：depth-perceptive-locomotion 10 个 2/3 节点补齐原文通读归档并升至 3/3

## [2026-11-20] ingest | Lumina 触觉 Scaling：异构传感器、事件数据与跨本体表征

- 意图：将 Lumina具身智能文章的六种传感路线、四层跨硬件差异和数据基础设施机会整理为工程选型对照。
- 来源：`sources/blogs/wechat_lumina_tactile_scaling_2026-11-20.md`（全文由用户提供；公众号网页本次独立打开失败，图片未核验）。
- 对照页：`wiki/comparisons/tactile-scaling-data-infrastructure-2026.md`；挂到触觉感知和 Contact-Rich Stage 3。
- 一手核验：Sparsh、T3、AnyTouch、FTP-1、LingBot-Touch、T-Rex、TacSL、Taccel、Tactile Genesis 的官方资料和统计口径。
- 结论：跨硬件的“共享表示”与“控制闭环”是两个分层问题；FTP-1 未见传感器实验仍要求 encoder 适配，Taccel 915 FPS 为 4096 环境总吞吐。复用已有独立论文节点，未重复建页；未运行代码或真机。

## [2026-11-20] ingest | 多模空间11月上半月具身智能风向标
- **意图：** 将九篇重点论文按问题组织为 VLA 技术地图和独立 canonical 实体，防止只列论文清单。
- **去重：** AnyCamVLA 与 A³-VLA 复用已存节点，另外七篇新建详情。
- **源码结论：** NS-VLA、DynaFLIP、PersonaDrive 的开源仓及脚本入口已静态核对。
- **未完成：** 公众号配图与全文、九篇论文必要实验和附录未完全核验，PR 保留草稿。

## [2026-11-20] ingest | sources/papers/kinetics_observer_arxiv_2406_13267.md — 收录 Kinetics Observer 紧耦合状态估计

- arXiv:2406.13267v2 论文独立实体：质心动力学、黏弹性接触、IMU/wrench 校正、可观性与真实机器人证据；包括基于原文的 Mermaid 流程图。
- 对照 jrl-umi3218/state-observation 官方 C++ 库的 KineticsObserver、单元测试和运行调用链，附真实源码时序图与单测/真机复现的边界。
- 与 State Estimation、SG-ODOM 和传统模型控制 Stage 4 互链；只提交源文档，不提交导出派生文件。健康度以 CI 实测为准。

## [2026-11-20] ingest | HandRSI arXiv:2611.22164：手部姿态估计 Harness + 权重联合递归改进

- 论文 HTML v1、官方代码运行入口与 HF 权重分别归档，建立独立 paper 实体、Mermaid 流程图及源码时序图。
- 记录 HOT3D/ARCTIC/HOI4D 结果、初轮人工放行、许可差异及加速层级；挂入 RSI Stage 4、WiLoR 互链。

## [2026-11-20] lint | 推进 #3490 第三批：通读核对 arXiv/官方项目页/仓库源码，将 roadmap/depth-motion-generation.md 全线 25 个待补节点升至 3/3（路线达标 40/40）

## [2026-11-20] lint | ASI 论文补齐方法/评测/对比三段式

- **基准：** 已合并的 PR #4006（`From AGI to ASI` 更新），GitHub Wiki Lint 实测节点健康度 **3/3**，所有工作流通过，但对 `wiki/entities/paper-from-agi-to-asi.md` 提示一条信息型预警「缺 对比」。
- **修复：** 在已包含“流程总览/核心机制”“评测与研究议程”的实体页中，新增 `## 与其他工作对比`，明确区别 DeepMind 宏观情景、具身 Scaling、MARL 和 RSI；以工作对象、可测证据和边界为对照维度。
- **原文与健康度：** 保留 `original` 及已通读归档，不更改主论文身份和页面状态；维持健康度 **3/3**，以本次 GitHub Actions 实测为准。
- **范围：** 不修改机器生成的派生文件，不增加重复论文实体。

## [2026-11-20] ingest | DeepMind Institute《What comes after AGI?》与 From AGI to ASI v3 核对

- **触发资料：** Tim Genewein (2026-11-19), [What comes after AGI?](https://institute.deepmind.com/essays/what-comes-after-agi/)；基于 [From AGI to ASI v3](https://arxiv.org/html/2606.12683)。
- **新增：** `sources/blogs/deepmind-institute-what-comes-after-agi-2026.md`，官方文章全文逐节核对、作者立场声明、四技术路径与认知工业化条件外推。
- **更新：** `sources/papers/agi_to_asi_arxiv_2606_12683.md`、`wiki/entities/paper-from-agi-to-asi.md`；同一 arXiv 论文不新建重复实体，补 v3 版本及 Evaluation Problem（Table 4 第七类瓶颈）。
- **结构核对：** 已有 `roadmap/depth-rsi.md` 含本论文链接，无需改动路线图；既有实体保留原 Mermaid 流程，并补文章对应的认知工业化/RSI 耦合 Mermaid。
- **原文状态：** 作者公开文章通读完成；技术报告 v3 全文 §1–7、Table 1–4、参考文献列表及 Appendix A 已通读，PDF 第 61 页的 Appendix B Glossary 已核对。未逐篇验证被引参考作品，不混同于本文原文未读；paper 归档标记已完成。
- **源码：** 本次为观点文章与理论综述，无可运行官方实现入口，源码运行时序图不适用。
- **验证：** 原 Wiki Lint 真实日志显示页面 1/3（缺阅读骨架、原文未通读），增补后须由新一轮 CI 实测健康度与其他检查确认，不捏造分数。

## [2026-11-20] ingest | 深蓝AI DNN/CNN/RNN/GAN：结构与对抗训练维度纠偏

- 收录用户提供的公众号完整文字，保留原始链接、读稿范围与网页图片未核验说明；不转载全文。
- 新增独立对比节点，复用 MLP/CNN/RNN/GAN 已有概念页，关联模型架构纵深路线 Stage 1。
- 以 LeCun/LSTM/Goodfellow 一手资料核对关键原理，区分「平移等变 / 不变」以及「DNN / MLP」。

## [2026-11-20] ingest | A Society of AI Scientists | 科研 Agent 社区观察与可执行审稿

- 意图：归档 alphaXiv Exclusive 论文的观察队列、公开数据、agent 审稿与可执行审计，对照机器人自动研究的外部验证边界。
- 开源结论：AgentAlphaXiv 数据集与 OpenResearch 代码已公开；平台当前源码并不能直接代表 35 个实验室在 12 周中的全部运行环境。
- 关键页：[独立论文实体](../wiki/entities/paper-society-of-ai-scientists.md) · [原文](../sources/papers/society_ai_scientists_alphaxiv_2611.md)。
- 待完成：论文 PDF 全文逐页及附录通读，独立全量实验复算和 PR CI 核验；因此先以草稿 PR 提交，健康度不虚报。

## [2026-11-20] ingest | WorldFlow v2：联合视频与 3D 运动预测原文核对及开源边界修正

- 复用唯一论文实体 `wiki/entities/paper-worldflow.md`，核对 arXiv:2610.39278v2 正文和附录，补全 Mermaid 方法流程与 WordFlowBench / RoboLab / 真机实证。
- 纠正原先把可访问 GitHub README 当成“模型代码已开源”的误判；移除无源码支撑的运行时序图，记录模型、权重、训练及评测工具尚待发布。
- 更新 papers/sites/repos 三层来源，并将其定位在 WAM 纵深 Stage 1 世界模型基座，不误归为联合 action 输出 WAM。

## [2026-11-20] ingest | 3D GameDev Arena 长时程游戏开发 Agent 三轨评测

- 来源：arXiv:2611.24073v1；108 个任务、14 类游戏、功能验收/代理试玩/真人 BT 偏好。
- 独立论文节点：wiki/entities/paper-3d-gamedev-arena.md；原文与试玩入口分别归档 sources/papers 和 sources/sites。
- 论文 HTML v1 正文及附录已核对。试玩 GitHub Pages 仅为重定向壳，未验证后端及官方可运行代码，保留核验缺口。
- 更新 Coding Agent 验证闭环的入链；避免将游戏体验指标误作机器人控制或真机安全指标。

## [2026-11-20] ingest | VLN 七基准的一手资料、框架与评测口径对照

- 意图：将 R2R、VLN-CE、R4R、CVDN、HANNA、SOON、REVERIE 与对应官方数据/代码入口统一成可复现的跨任务对照。
- 开源核查：R4R 提供数据生成器；CVDN/NDH、SOON 提供训练/评测脚本；VLN-CE 依赖历史 Habitat 环境；HANNA 数据和实验入口仍需另行检查访问性。
- 关键页：[七类 VLN 基准与评测协议](../wiki/comparisons/vln-seven-benchmarks-frameworks-evaluation.md)；导航纵深 Stage 4 已挂载。

## [2026-11-20] ingest | sources/blogs/wechat_shenlan_world_models_11_2026_11_20.md — 编译 11 项世界模型，补录 4 项独立实体与路线对照

- **意图：** 将深蓝具身智能的 11 项世界模型转为技术路线与源码可复现边界对照。
- **资料：** sources/blogs/wechat_shenlan_world_models_11_2026_11_20.md
- **结果：** 复用已存在的 7 项实体，针对缺项增加 HY-WorldPlay 1.5、Matrix-Game 4.0、Lyra 2.0、AlayaWorld 独立节点；补充对照与 WAM 纵深链接。
- **主要修正：** 区分发布年份、模型侧 FPS 与端到端时延、开源代码和权重许可。

## [2026-11-20] ingest | sources/sites/agility-robotics.md — 公司路线新增敏捷机器人（Agility Robotics，Research + Analysis）并建立公司与 Motor Cortex WBC 实体页

## [2026-11-20] ingest | A Circuit for In-Context Learning v1 原文审阅与已有唯一节点完善

- 复核 arXiv:2611.05344v1 正文与附录、官方 MIT 许可证、README 和单文件交互讲解入口。
- 复用 `wiki/entities/paper-a-circuit-for-in-context-learning.md`，不重复建页；补齐原文归档与结构化阅读状态。
- 区分读取头因果解释的部分效应、读取律 18 示例外推失败，以及仅适用于短句分类、未验证机器人策略的边界。
- 未运行 H100 全量复现实验；源码模块尚未逐文件审计。

## [2026-11-20] lint | 合并重复项目节点 linger-personal-agent 至 linger-simplexai 并补齐 claude-code-dynamic-workflows 对比落地

## [2026-11-20] lint | health-check — 修复 4 项高频 methods 缺查询落地与 11 项缺页概念误报候选，info 预警降至 734（不含 stale 为 500）

## [2026-11-20] ingest | sources/repos/linger-simplexai.md — Linger 本地个人 Agent

- 依据 SimplexAI 官方 GitHub README、架构/推理/Wiki/任务文档、LICENSE 与官网 FAQ，新增 Linger 唯一实体。
- 核对本机 Runtime、Computer Daemon、主动 Wiki/记忆、桌面工具与许可证限制；补 Mermaid 数据流程和源码运行时序图。
- 对照 Hermes Agent、Deep Agents、LLM Wiki 及 Agent 验证闭环；未把性能宣传当成独立实测，也未把桌面自动化视为机器人实时控制。

## [2026-11-19] ingest | sources/blogs/openathena-scaling-marins-open-llm-pretraining-2026-11-05.md — Marin 开放 LLM 预训练

- 通读 Will Held 原始演讲全文及问答，整理 Delphi / Iris / MoE / Datakit / Belay 体系，并建唯一 Marin 项目实体。
- 官方代码已公开并核对可运行 TinyStories 教学入口，补项目流程图及真实源码时序图；未执行训练。
- 核对 2026-11-17 官方 Mochi 发布口径：18.2T 已训练 token（不同于早期约 20T 规划）、57 个预训练 checkpoint，并保留 benchmark 污染与无安全后训练的限制。
- 与 AI Auto-Research 建立双向关联；纯文本 LLM 项目不硬挂机器人纵深路线。

## [2026-11-19] ingest | ODrive v3 固件 0.5.6 RTOS/双轴 FOC 实时控制

- 从比特Z 2026-09-24 技术文章提炼 RTOS 与硬实时控制的两层架构，文章与官方版本锁定源码分别归档。
- 官方 `fw-v0.5.6` 核对 TIM8 sampling、ControlLoop ISR、Axis 任务与 8 kHz/125 μs 控制周期、超期双轴卸力、单目标梯形轨迹。
- 新建 `wiki/entities/odrive-v3-firmware-rtos-foc.md`，包含 Mermaid 控制流与源码运行时序图。
- 与 FOC 概念页及力矩电机设计 Stage 3 交叉链接；不将该实现外推至 ODrive 新一代产品，也不把静态源码分析误写为 WCET 实测。

## [2026-11-19] ingest | Overleap 双自由度机械腿弹跳控制

- 意图：从比特Z《拆解 Overleap：机械狗腿的弹跳算法》与 Overleap 原始源码整理完整知识实体。
- 开源结论：Aaron de los Santos 的 Overleap 官方仓库公开 Arduino 库、ODrive 示例、装配与机械图。
- 关键判断：下蹲、收腿和抬腿使用 MCU 侧关节 PD；蹬地使用 200 ms sin² 期望足端力与 Jᵀ 力矩映射；FSR 只负责接触判断，并非足端力闭环。
- 修正细节：标明 200 ms 力曲线与代码中的 >400 ms 离地退出条件不匹配。
- 本次核心页：wiki/entities/overleap.md；关联 roadmap/depth-classical-control.md Stage 1。
- 验证边界：已做文档及官方源码静态核读，未进行实机复现。

## [2026-11-19] ingest | sources/papers/so_arena_arxiv_2611_15777.md — 收录 SO-Arena 双轴结构化决策评测与可靠性复现入口

- 基于 arXiv:2611.15777v1 与官方项目页，新增独立双语论文节点，记录 2,919 任务 / 11 领域 / 1,232 榜单任务与 C、S、N、R 分离指标。
- 官方代码 holdrain/SO-Arena、HF 数据集 SO-Arena/SO-Arena 已发布；摘要之外复核 data-lock、evaluator、adapter 及 reliability 实际入口，并补流程图与运行时序图。
- 将新论文链接到 AI Agent 评测页；不强行挂到仅讨论具身/运控的纵深路线，明确 benchmark 不等于机器人真机安全性。
- 本次为一手资料和源码静态核读，未声称执行本地 benchmark 或 GPU 复现。

## [2026-11-19] ingest | Co-Sight Harness Studio

- 意图：独立收录 ZTE-AICloud/Co-Sight 单仓中的 Co-Sight Harness 子项目，不把 Co-Sight 主平台与 Co-Knowledge 误归为 Harness。
- 开源结论：官方框架、Drive 编排插件、hooks 与多项配套组件源码公开；CLI/模型仍需外部依赖。源码与文档核对，未运行本地端到端验收。
- 关键页：[Co-Sight Harness](../wiki/entities/co-sight-harness.md)；[官方源码归档](../sources/repos/co-sight-harness.md)。
- 关键风险：框架默认全网卡监听且无登录鉴权；工作流门禁不代替机器人设备安全。

## [2026-11-19] ingest | embodiedglobal-embodied-ai-guide | 具身智能入门综述

- 意图：将 Embodied Global 2026 长文从「感知—认知—行动」闭环归纳到本站具身智能入门坐标。
- 来源：sources/blogs/embodiedglobal_embodied_ai_beginners_guide_2026_06_17.md；原文 2026-06-17 发布，已核对全文章节。
- 升格：wiki/overview/embodied-ai-beginners-guide-2026.md；对照本站模型架构、VLA、Sim2Real、Infra 和成长路线。
- 源码：文章无可运行实现；概念流程图不是源码运行图。
- 核验边界：商业预测、量产和市场数字按原文转述，未独立一手核实。

## [2026-11-19] ingest | Remotion + HyperFrames 官方源码与视频工程栈复核

- 复用既有 `wiki/entities/remotion.md` 和 `wiki/entities/hyperframes.md` 两个唯一实体，不重复建页。
- 核对 Remotion CLI/本地渲染入口与许可：免费许可适用个人、最多三名员工的营利组织、非营利组织及非商业评估。
- 核对 HyperFrames CLI/生产模块，纠正旧的 21 Skills 为截至 2026-11-19 官方列出的 9 个公开 Skills；分清开源核心与 Studio Pro。
- 两页增加源归档通读字段、项目流程与源码运行时序图、选型比较与双向链接，按脚本判据保留 3/3 健康度；CI 复核为准。

## [2026-11-19] ingest | 复核 DriveTeach-VLA（ECCV 2026 / arXiv:2607.01658）论文全文及官方代码

- 查重后复用 `wiki/entities/paper-driveteach-vla.md`，不新增同一论文的重复实体。
- 通读论文 §1–5 与 Appendix 0.A–0.C，修正 NAVSIM navtest PDMS 90.4（单次）/ N=12 selector 92.7（不可直接混比）、nuScenes L2 0.30 m 与基准差别、消融、延迟和检测器噪声边界。
- 核对 `data_engine/main.py`、`data_engine/src/pipelines/runner.py`、`dvd/README.md` 及训练 YAML，按真实调用链完善 Mermaid 数据引擎运行时序，明确论文 `2×4` 与当前 DVD YAML `16×28` 的差异；GRPO 仍依赖 Curious-VLA。
- 补全 `sources/papers/` 归档结构化「核对范围 / 通读」及实体 original / project_id，健康度以 CI 同一脚本检测为准。

## [2026-11-19] ingest | sources/papers/flomo_arxiv_2611_05062.md — 精读 FloMo 原文并升级三维场景流 WAM 节点

- 意图：检验“预测 3D 运动是否比预测未来视频更有利于动作学习”，区分视频预训练与未来预测目标。
- 原文：核对 arXiv:2611.05062v2 的方法、实验、局限与附录；保留同论文已有唯一实体 `wiki/entities/paper-flomo.md`。
- 开源：官方 pairlab/flomo 已提供训练、预处理、推理入口和 MA/VA/VMA 权重；运行与真机集成尚未独立复现。
- 关键结果：同条件真机 MA 91%、VMA 51%、VA 40%；OOD MA 18/30、VA/VMA 各 2/30；RoboTwin Easy 五任务 MA 63.6%。
- 落点：补原文与代码核验、Mermaid 主流程和源码时序，挂入 WAM 纵深路线 Stage 5。

## [2026-11-19] ingest | sources/sites/spirit-ai.md — 公司路线新增千寻智能（Spirit.AI，https://www.spirit-ai.com/blog/），新建 Spirit.AI 与 Legato 实体页并补齐 Spirit-v1.5 源码/权重/博文核查（MIT 代码与 Apache-2.0 HF 权重，uDAS 与 Spirit v1.6 闭源；Legato 仅开源 Kinetix 仿真）

## [2026-11-19] ingest | sources/papers/kagan_2025_cl1_biological_neural_platform.md — 收录 CL1 生物神经闭环平台论文并区分后续官方 SDK

- 新增唯一论文详情节点 `wiki/entities/paper-cl1-biological-neural-platform-kagan-2025.md`，记录期刊原文与后续 API 文档的时间与证据边界。
- Nature 原文订阅正文暂无法核实；已在 sources 标明「通读：未完成」，论文节点保持 draft，PR 同步设为草稿。
- 对接官方 cl-sdk 的可验证示例运行时序图；由强化学习方法页提供入链，明确生物闭环不等于机器人 RL。

## [2026-11-19] ingest | VGE | 冻结 VLM + 接地专家处理广义指代

- 意图：通读 arXiv:2611.22625v1 并为 VGE 建立唯一论文详情节点，挂载 VLA Stage 2。
- 机制：逐层联合注意力 + 文本条件实例评分；输出零/单/多目标框，不直接输出动作。
- 结果：RoboRefIt Test A/B 三指令平均 F1 95.02/89.67%；推理约 140 ms（不含预处理）。
- 开源：论文承诺发表后公开 VGE 代码，截至入库日尚无可核实的官方运行入口；不提供虚构源码时序图。
- 关键页：sources/papers/vge_arxiv_2611_22625.md；wiki/entities/paper-vge-2611-22625.md；roadmap/depth-vla.md。

## [2026-11-19] ingest | sources/blogs/claude_dev_chaos_loop_docs_launch_2026-11-17.md 与 claude_dev_dynamic_workflows_2026-06-02.md — 整理 Claude Docs 持续故障注入和 Claude Code 动态工作流

- 通读两篇 Claude Code 官方博客，分别形成 Chaos Loop、动态多 Agent Harness 独立方法页，绘制基于原文的 Mermaid 流程图。
- 区分内部工程自报指标、未经实测的成本估算和面向机器人的迁移建议；文章没有提供可复现实验源码，不虚构运行时序图。
- 在 RSI 纵深路线 Stage 3（验证锚）与 Stage 4（harness）中添加推荐读物；不把动态工作流误作权重级 RSI。

## [2026-11-19] ingest | 通读一手论文与官方仓库源码，将 depth-imitation-learning 与 depth-icl 两条纵深路线的 24 个缺分节点全部升至 3/3 满分健康度

## [2026-11-19] lint | 清理全局 lint 首批高性价比 Info（8 页 / 11 项）：4 类知识链枢纽回链清零、GM-100 重定向就绪度、pyplcopen 版本快照措辞与 Limit of RLVR 对比小节

## [2026-11-19] ingest | 机器人论文与工具六项核对

- 意图：以一手论文和官方源码整理 TCP4D、Heave-Ho、ForceHarness、robot-weight-debugging 四篇独立论文，更新既有 XMimic/XSoma 与 Alloy 实体。
- 开源结论：TCP4D / ForceHarness / 权重补丁代码可读；Heave-Ho 截至入库日代码尚未发布；Alloy v0.1.7 官方 release 已公开。
- 知识落点：四个新 wiki 论文实体，两个既有论文详情的版本补充，manipulation 任务入链及相关纵深路线。
- 边界：只做论文/源码静态核查，未复现论文数值，健康度与 CI 需 PR 流水线验收。

## [2026-11-19] ingest | 通读一手论文、官方手册与源码，将 motion-control、depth-robotics-diffusion-dit-flow、depth-teleoperation 三条路线的 19 个缺分节点全部升至 3/3 满分健康度

## [2026-11-19] ingest | Meta-Rewarding LLM v2：双偏好自训练与元评委漂移

- 精读 arXiv:2407.19594v2，建立独立论文实体并连接 Self-Rewarding 与 RSI Stage 3。
- 区分 Actor/Judge/Meta-Judge 与前两轮联合 DPO、后两轮仅 Actor DPO；补充原文算法 Mermaid 流程图。
- 原文报告 AlpacaEval 2 LC 22.92%→39.44%，但元评委高分、位置偏差加剧；未核实到论文专属官方可运行源码，不编造时序图。
- 论文已发表于 EMNLP 2025；注意正式发表与用户指定 arXiv v2 内容版本的区别。

## [2026-11-18] ingest | Claude Code 验证与代码评审两篇官方博客交叉核对

- 复用已有博客原始归档与 Coding Agent 验证闭环页，不创建重复来源或实体。
- 补齐 2026-10-15 评审文章的全文通读与版本核对字段，新增验证/评审证据与风险门禁对比页，并从已有概念和方法页双向关联。
- 两篇均为软件工程实践博客，无独立公开实现；机器人安全门禁是工程迁移建议，不当成原文事实。

## [2026-11-18] ingest | sources/papers/diplo-self-distilled-diffusion-planning_arxiv_2610_24847.md — 收录 DiPLO 自蒸馏扩散全身搬运规划

- 核对论文 arXiv:2610.24847v1 全文及项目页，建立唯一独立论文详情、Mermaid 数据/控制流程与 Loco-Manipulation Stage 4 导航。
- 区分仿真存活条件成功率 98.8%、G1 真机 10/10+5/5 和外部 mocap；网站源码不是训练推理代码，数据可得性待核实。

## [2026-11-18] ingest | HiGR v5（CIKM 2026）与 TGR 推荐技术栈

- 意图：将两篇独立腾讯推荐技术论文分别入库，保留 HiGR 与上层 TGR 架构之间的清晰身份与交叉引用。
- 一手核对：HiGR arXiv v5、TGR 技术报告、CIKM 2026 官方会议日程。
- 开源结论：两篇论文未列可核验的官方训练/推理源码；TRS 产品入口可检索，但官网完整页面此次访问超时，记录为未通读。
- 定位：跨域可借鉴 Transformer/Token/异步 Reason 工程分层，不将线上推荐指标误认为机器人控制结果；无匹配机器人纵深路线，不强行挂载。

## [2026-11-18] lint | entity-outgoing-batch-04 | 再修复 100 个实体关联出边

- 在 #3873、#3875 合并后的最新 main 上另选 100 个节点，与前三批共 300 个节点无重叠，覆盖 10 个主题、45 个零出边及 55 个单出边缺项。
- 新增 145 条带关系说明的 methods/tasks 出边（51 个不同目标），每页恰有 2 个不同且有效目标；移除对应 100 条冻结出边豁免。
- 全部 100 页的健康度、阅读与来源状态保持原值；全库实际历史出边缺项 3,079 → 2,979。
- 四批累计修复 400 个不同节点、新增 597 条出边、移除 400 条出边豁免；[本批逐页清单与复验方法](../docs/checklists/lint-outgoing-batch-04-100.md)。
- 验证：`make ci-preflight` 通过（lint 0 阻塞、搜索 40/40、导出 14/14）；45 项定向测试、100 页真实规则及基线负对照、3 类站点样本与 5 个目标跳转通过。

## [2026-11-18] ingest | Zero to SGLang 推理课程入库

- **意图：** 按课程官方中文 README 和 Part II 代码走读新增 Datawhale × RadixArk 的双语推理基础设施课程独立节点，关联现有 SGLang、Mini-SGLang 与模型架构纵深 Stage 5。
- **开源结论：** 课程 GitHub 提供开放的 Markdown 教材和代码走读；其参考可执行引擎分别在独立的 sgl-project/mini-sglang 和 sgl-project/sglang 仓库，并非 zero-to-sglang 仓库内的运行工程。
- **发布边界：** 官方 README 标示课程正文已完成、视频仍在发布；GitHub API 已核实 2026-11-12 发布中文 363 页、英文 426 页 PDF Release，PDF 为静态快照。
- **关键页：** wiki/entities/zero-to-sglang.md；sources/courses/zero-to-sglang.md；sources/repos/zero-to-sglang.md；sources/sites/zero-to-sglang.md。

## [2026-11-18] ingest | sources/papers/vino_arxiv_2611_07176.md — VINO 官方资料增量复核及跨域迁移评测

- 对照作者论文 HTML v1 的 §3–5 与 Appendix A.2/A.5，补录 3DMatch/3DLoMatch 冻结特征配准、NYUv2 单视角 RGB-D 与训练资源口径；明确配准后端与机器人部署边界。
- 核实 VINO 作者项目页、官方 GitHub：2026-11-18 仍为 README 占位，代码与权重待发布；复用唯一的论文节点，不重复创建。
- 将现有 VINO 节点接入模型架构纵深 Stage 2，与 Volt 和二维预训练骨干作区分。

## [2026-11-18] lint | entity-outgoing-batch-03 | 追加修复 100 个实体关联出边

- 在 PR #3875 同一分支同步 main 后，另选与前两批均无重叠的 100 个节点，覆盖 10 个技术主题及 54 个零出边、46 个单出边缺项。
- 新增 154 条带关系说明的 methods/tasks 出边（47 个不同目标），本批所有节点恰有 2 个不同且有效目标；移除对应 100 条冻结出边豁免。
- 本批及 PR 全部 200 页健康度、阅读与来源状态保持原值；本批使实际历史出边缺项 3,179 → 3,079。
- PR #3875 合计 200 个不同节点、285 条新增出边；三批累计 300 个不同节点、452 条新增出边、300 条出边豁免移除。
- [本批逐页清单与复验方法](../docs/checklists/lint-outgoing-batch-03-100.md)。
- 验证：`make ci-preflight` 通过（lint 0 阻塞、搜索 40/40、导出 14/14）；45 项定向测试、200 页真实规则与基线负对照、3 类站点样本及 5 个目标跳转通过。

## [2026-11-18] lint | entity-outgoing-batch-02 | 再修复 100 个实体关联出边

- 在首批 #3873 合并后的 main 上另选 100 个不同节点，覆盖 10 个技术主题与 0/1 出边两种缺项。
- 新增 131 条带关系说明的 methods/tasks 出边，所有样本恰有 2 个不同且有效目标，移除对应 100 条冻结出边豁免。
- 本批健康度、阅读和来源状态保持原值；全库实际历史出边缺项 3,279 → 3,179。
- 两批累计修复 200 个不同节点、增加 298 条出边；[本批清单与复验方法](../docs/checklists/lint-outgoing-batch-02-100.md)。

## [2026-11-18] lint | entity-outgoing-pilot | 100 个代表性节点关联出边试点

- 从全库 3,379 个实际历史缺项中按 10 个主题各选 10 页，覆盖 0/1 条出边与论文、工具、机器人、数据集和硬件接口等角色。
- 为 100 页补充 167 条有关系说明的 methods/tasks 出边；每页现有 2 个不同且有效的目标，同步移除这 100 条历史出边豁免。
- 全库实际缺项剩 3,279 个；样本健康度与原文通读状态保持原值，本批不承担存量资料补读。
- [完整样本、统计与复验方法](../docs/checklists/lint-outgoing-pilot-100.md)。

## [2026-11-17] ingest | Deepu Talla 访谈同源材料去重合并

- 同一 YouTube 视频 `rarKwkLN-Bg` 已在 PR #3867 收录；PR #3870 只补充更细分段的访谈观点、Microduck 案例、System 0–3 分层和五项工程基本功。
- 保留唯一视频资料：`sources/videos/nvidia-robotics-chatgpt-moment-deepu-talla.md`；保留唯一知识实体：`wiki/entities/nvidia-robotics-chatgpt-moment-deepu-talla-2026.md`。
- NVIDIA Cosmos 补充到唯一实体的入链；不再新增 `titv-nvidia-deepu-talla-physical-ai-2026.md` 同源资料和实体，避免重复。
- 倍数、可靠性和时间预测均与独立实验结果分开标注。

## [2026-11-17] ingest | NVIDIA 公司路线更新至 2026-11-17：补入 GR00T 2、cuRoboV2-Fast、Lab 3.0 与 DexBench，复核历史工具节点及原文边界；其余未通读论文保留真实健康度。

## [2026-11-17] ingest | sources/papers/mosu2_arxiv_2611_11148.md — 归档 MOSU2 多模态长程户外导航；新增独立论文实体、Mermaid 流程与导航纵深入链

- 一手依据：arXiv:2611.11148v1 全文与官方项目页；实测数字只采用 v1，不混用网页其他稿次。
- 系统：GPS 路由 + MTG/CVAE + SAM3 可通行性 + 可检索的几何/速度地图 + 按需两阶段 VLM 选轨迹。
- 工程边界：官方 GitHub 现为静态站点，不含可运行导航代码；高层云端 VLM 延迟为主要部署风险。

## [2026-11-17] ingest | VTAM v2 与 DexTacWAM 视触觉 WAM 更新

- 意图：依据两篇 arXiv 论文与作者官方实现重新核对触觉世界建模、部署推理与开源边界，更新已有节点而不重复建页。
- 论文：VTAM arXiv:2603.23481v2（2026-11-05）更新了标题、UniVTAC 八任务、Hanoi 跨 Xense 传感器与延迟消融；DexTacWAM arXiv:2609.24976v1 保留已有完整原文和源码归档。
- 开放核查：VTAM 训练/推理代码与 xArm+GelSight 数据已发布，UniVTAC 仿真配置及任务 checkpoint 仍未开放；DexTacWAM 三阶段实现与 Stage 1 权重/数据发布，任务 Stage 2/3 checkpoint 不提供。
- 知识结构：升级 VTAM 唯一论文实体，补项目流程与真实源码运行时序；DexTacWAM 与 VTAM 双向对照；接入 WAM / 接触操作 Stage 阅读链。
- 风险边界：区分 VTAM 三任务成功率（90%）与 DexTacWAM 六任务综合评分（70.6），不跨任务排名；两仓均存在非商业许可组件。

## [2026-11-17] ingest | Karpo Decision 全尺寸结构化决策模型、源码路径与许可核对
- 归档 MachinePulse 的 0.8B–31B 六款公开模型及 `Decider` / System One 可运行推理入口，建立 [Karpo Decision](../wiki/entities/karpo-decision.md) 实体并交叉关联 AI Agent 评测。
- 核心机制：共享状态下对 `choice` / `noul` / `score` 候选一次性概率读出；Qwen3.5 和 Gemma 4 的编码/注意力路径需区分。
- 开放边界：推理仓 Apache-2.0，模型卡报告权重许可在 2026-11-12 后变更；性能是 Decision Index 口径，不推断真机控制结果。

## [2026-11-17] ingest | sources/sites/robbyant-embodied-ai-challenge-2026.md — 归档蚂蚁灵波具身大模型挑战赛官方赛题并关联 VLA / RoboTwin 测评链

- 依据阿里云天池主办方赛事介绍与赛题资料，新增独立竞赛知识实体和赛事流程 Mermaid；关联 LingBot-VLA 2.0、RoboTwin 2.0 及具身模型测评纵深 Stage 3。
- 官方协议的关键区别：Aloha-AgileX 的 50 个任务每任务仅允许 50 条 clean 数据用于后训练，clean 和 randomized 两种 setting 下每任务分别评测 100 次；与 LingBot-VLA 2.0 README 通用后训练示例严格区分。
- 开源结论：模型代码、6B 权重及 RoboTwin 2.0 数据/仿真均公开；赛事本身未确认独立开源的统一评测程序和决赛任务包，不虚构源码运行时序图或官方最终获奖名单。

## [2026-11-17] ingest | sources/repos/weknora.md — 归档腾讯 WeKnora 开源知识管理平台

- 核对官方产品介绍、总体架构、RAG Pipeline、服务启动与解析入口。
- 建立 WeKnora 唯一项目实体，补充 RAG/Agent/Wiki 流程图、Go/Redis/docreader 实际文档入库时序图及相关知识互链。
- 源码已开源；未运行本地部署或效果测试，不把通用知识平台作为机器人控制算法。
- 对照现有 30 条纵深路线，本项目属于通用研发知识基础设施，暂无合适的目标导向 Stage，不强行挂载。

## [2026-11-17] ingest | mjviser 原位更新官方版本、源码时序与交互边界

- 已有 `wiki/entities/mjviser.md` 与 `sources/repos/mjviser.md`，避免重复实体。
- 核对官方 README、0.0.17 依赖、CLI / Viewer / 场景与交互源码；新增主干 Mermaid 和运行时序图，修正刚体拖拽现状。
- 仿真/训练分层：mjviser 是 Apache-2.0 MuJoCo 网页查看器，mjlab 承担 RL 工作流；本次未实机/本地启动验证。

## [2026-11-17] ingest | sources/repos/copilotkit-openbot.md — 收录 CopilotKit OpenBot AI Agent 工作空间

- **入库目标：** 归档 OpenBot 官方 README、架构、工作流、部署与许可，整理为独立项目节点并增加跨页关联。
- **名称去重：** CopilotKit/OpenBot 是 Agent 工作空间，不是 Intel Labs 的 ob-f/OpenBot 智能手机轮式机器人，也不与 CopilotKit/OpenMuse 合并。
- **开源边界：** 应用模板 MIT；CopilotKit Intelligence 独立许可，服务端计算组件依赖已发布的运行镜像。
- **技术边界：** AG-UI 编排、MCP 工具、Computer 与审批审计适用于上层自动化，不替代 ROS 2 或机器人实时运动控制。

## [2026-11-17] ingest | 七项机器人研究与产品动态
- 论文：沿用已有 PEAR/RURL 深度节点，新增 Agentic-Teacher、WATO、TouchTwin、Racing the Clock 四篇独立论文实体与来源。
- 产品：XHAND 1 Pro-U 独立实体与新闻稿。
- CloudRobo-WM1.2：官方发布资料 source-only，技术全文、可执行源码和榜单尚待核实，不虚报通读。
- 纵深：WAM、VLA、具身数据、导航、接触操作路线挂载。
- 未在本执行环境开展真机、仿真复现；健康度以 Wiki Lint 的 report_health.py 为准。

## [2026-11-17] lint | 复核来源时效性与缺页概念：按提交快照分流 169 条来源提示，实质同步八页并保留 161 条待核对；补机器人本体概念及回链，仅排除 complete 非概念标签，保留真实缺页回归检查。

- 全量复检同时修复 ARP 的否定式 SOTA 误报；真实断言保留回归覆盖，不改论文结论或日期。

## [2026-11-17] lint | 继续修复全量 Wiki Lint：补齐 6 处评测/感知知识链回链、5 处来源元数据，并为 RLCR 增加 CTDE 对比实践入口；保留历史 stub 的原文未核读与内容缺项，不以模板或状态变更代替核读。

## [2026-11-16] lint | PCB 设计路线全部 41 节点达到 3/3；以 NIST、ADI、TI、Autodesk、RLS、igus 与 NVIDIA 官方原文补齐剩余证据并修正工程口径

## [2026-11-16] lint | PCB 设计路线：补全阅读骨架、复核官方 README 与版本边界；保留未获原文节点的缺项，不改健康度规则

## [2026-11-17] structural | 抽取 scripts/utils/markdown 公共工具、合并交叉链接测试并校准全量覆盖率、收紧 canonical-facts 与建立历史 Lint 基线

## [2026-11-17] structural | 清理 exports 遗留产物、归档历史文档、将 catalog.md 移出 Git 并启用 log.md 季度自动归档

## [2026-11-16] ingest | sources/blogs/wechat-robotics-29-papers-2026-11-16.md | 29 篇具身论文导览线索与两篇重点原文

- 意图：归档用户提供的 8 页论文动态，不用转载摘要代替 29 篇论文核读。
- 完成：通读 ActionCodec2、ManiLadder 一手论文并分别生成独立实体/流程图；ActionCodec2 额外核读官方可运行评测入口和时序图；新增对应 VLA、具身评测路线链接。
- 去重：JustRUA 和 STVLA 已有独立实体节点，不复制。
- 边界：其余线索仅供后续逐篇核实；ManiLadder 项目站点读取失败，源码/数据是否开放待查；本次不声称完成 29 篇全文核读和实验复现。

## [2026-11-16] ingest | OVBench 与 S-EMBER：流式视频理解与第一视角情景记忆

- **意图：** 独立归档两篇原文，建立在线视频问答及第一视角有证据记忆检索两条知识节点，并挂载具身模型评测 Stage 1。
- **开源结论：** OVBench 官方已有训练/评测脚本与 ViT 特征 PMB（KV-cache 版本尚待）；S-EMBER 已有 MCQ/grounding 评测脚本和 HF 数据。
- **判断边界：** 在线时空问答与时间证据检索相关却不同；都不能代替机器人动作成功率。
- **待办：** 论文所有附录完整核读、模型评测复现、健康度与 CI 实测。PR 保持草稿。

## [2026-11-16] ingest | InternW0-Δ 官方权重上线与源码运行时序核对

- 原位完善 `wiki/entities/paper-internw0-delta.md`，避免按同篇 arXiv 论文重建重复实体。
- 复核 Base/LIBERO/RoboTwin/RoboDojo 四个公开 HF checkpoint，修正旧的 forthcoming 表述，明确 MIT 代码、Apache-2.0 模型与第三方数据许可边界。
- 阅读训练与 RTC ROS2 部署源码、补 Mermaid 源码时序图，挂载 WAM 纵深路线 Stage 3；未实际运行训练或真机。

## [2026-11-16] ingest | MACaQ | 逐层 MXFP4 量化的精度、时间与推理代价

- 意图：依据 arXiv:2611.10649v2 完整方法与实验收录 MACaQ，并核实多后端官方实现。
- 开源结论：论文、vLLM/llama.cpp/Diffusers/PyTorch 代码与部分 HF 模型产物可访问；未实际复现。
- 知识定位：新增 [MACaQ 独立论文详情](../wiki/entities/paper-macaq.md)，挂载模型架构纵深路线 Stage 5；与机器人 VLAQuantBench 闭环量化评测对照，避免把语言模型 PPL 误写成真机成功率。
- 一手资料：https://arxiv.org/abs/2611.10649v2；https://github.com/huawei-csl/MACaQ；https://huawei-csl.com/blog/macaq/；https://huggingface.co/collections/huawei-csl/macaq

## [2026-11-16] ingest | sources/papers/opennwm_arxiv_2611_01844.md — 核对 OpenNWM arXiv v1 公式与实验、源码和 Go2 部署边界

- 更新已存在的论文实体 wiki/entities/paper-opennwm.md（不重复建页）：UNLA 物理监督、CDiT、真实动作 CEM、v1 Table 1–5 与 NavAnywhere Table 6。
- 更新来源归档（paper/repo/site），核查根 run.py、LAM/NWM 训练评测入口、Hugging Face 模型与数据。
- 明确 Office-Go2 ATE 2.89 / RPE 0.65 是离线评测；Go2 EDU + Jetson 仅有部署计划，尚无验证的闭环节点。
- 将节点挂接至导航纵深路线 Stage 5。

## [2026-11-16] ingest | sources/blogs/raschka_understanding_positional_embeddings_2026_11_15.md — 通读 Raschka 教程，归纳现代 LLM 位置编码

- 原文为 2026-11-15 发布的技术博客；全文 §1–§7 已核对，非独立论文或新项目。
- 新建位置编码概念页（GPT-2 absolute、sinusoidal、RoPE、Partial RoPE、NoPE、相对偏置），并为 Transformer 与模型架构纵深 Stage 1 建立交叉引用。
- 源码核对了原文所引用的 rasbt/LLMs-from-scratch Qwen3 notebook，绘制基于真实调用关系的运行时序图；未执行模型。
- 对长上下文泛化与行业趋势的结论保留文章采样边界，区分数学性质与模型表现。

## [2026-11-16] ingest | sources/papers/cortex_arxiv_2610_20399.md — Cortex：上下文到 FFN 参数记忆的巩固

- 依据 arXiv:2610.20399v1 整理 prospective rehearsal、FFN 参数扩展及无上下文蒸馏，独立建立中文/英文论文详情。
- 明确 CR/RR 与 50 次连续事实编辑口径，区分 LLM 文本记忆巩固、外部 agent 记忆及机器人连续学习。
- 官方 zjunlp/Cortex 仓库截至核验日只有 LICENSE，暂无可阅读的可运行源码；不编造源码时序图。
- 将差异对照链接到 Agent 记忆概念页与 RSI 纵深路线；未触碰部署派生文件。

## [2026-11-16] ingest | sources/papers/vbot_exo2ego_arxiv_2611_16417.md — 归档 VBot Exo2Ego Policy 并挂载具身数据纵深路线

- 依据 arXiv:2611.16417v1 全文核对双机 EGO/EXO 采集、MiniMax-H3 视频生成、训练期相机射线与语义/手部监督。
- 归纳完整系统与 EgoX 对照，明确训练配方差异、部分覆盖率下降、未提供真机策略成功率。
- 当前论文未列官方可运行源码入口；暂不绘制未经核实的运行时序图。

## [2026-11-16] lint | 全路线复核：基于 7801 个 wiki 文件及上次审查后的新增 26/修改 483 个节点，修正先修与图文分支、基础控制结论及模块边界，补 VLA 复现入口；逐页记录见 docs/quality/roadmap-audit-2026-11-16.md。

## [2026-11-16] ingest | 近期机器人学习论文六项补录

- 意图：补齐 SALA、EgoX、Reins、ACE-CaP-0、STVLA、JustRUA 的论文实体和方法/评测/局限，建立主题互链并接入 VLA 与跨本体纵深路线。
- 查重：CoMind / Be My Eye（arXiv:2611.12295）已有实体和来源档案，本次避免重复入库。
- 代码：EgoX 可运行仿真训练/评测但缺数据及权重；JustRUA 开源评测适配器但有外部依赖；其余论文尚无完整公开可运行入口。
- 关键页：wiki/entities/paper-sala-latent-actions.md、wiki/entities/paper-egox-mxt-plus.md、wiki/entities/paper-justrua-robot-use-agents.md。

## [2026-11-16] ingest | sources/repos/googlechrome-modern-web-guidance.md — 收录跨浏览器现代 Web Agent Skill

- 意图：为知识库前端工程和编码代理工作流补充一手 Web 平台实践指南；不将其误归类为机器人算法框架。
- 开放结论：技能分发仓 GoogleChrome/modern-web-guidance、实际源码仓 modern-web-guidance-src 均公开；CLI 支持 search / retrieve / legacy-scan，扫描覆盖有限。
- 关键页：[Modern Web Guidance](../wiki/entities/modern-web-guidance.md)、[项目页核查](../sources/sites/modern-web-guidance.md)。
- 纵深路线：已核对 roadmap/README.md 的三十条路线，无对应的机器人纵深 Stage，不强行挂载。

## [2026-11-16] ingest | PACE 人机交互人格适配原论文与项目资料

- 来源：arXiv:2607.15579v5（Humanoids 2026）；麦考瑞大学、英伟达新加坡。
- 知识：互动问答 → PersonaSpec → 动态 prompt 编译 → Ameca 语音/面部表情同步；对比静态人格并保留实验统计非显著结果。
- 资源核验：lipzh5/PACE GitHub 有公开项目资料，README 所列 Python 启动脚本不在仓库，暂不具备公开源码级复现入口。
- 关键页面：wiki/entities/paper-pace-persona-adaptation.md；wiki/methods/humanoid-voice-interaction.md。

## [2026-11-15] ingest | AesFormer（ICML 2026）：七维美学规划与结构化照片编辑

- 意图：核对 arXiv:2605.22126 的 APR 任务、VCMP/AesRecon、SFT+GRPO-A、动作条件编辑器及评测，并归档到独立论文实体。
- 开源：AesFormer 训练/推理/评测代码公开，权重由官方仓库链接；AesRecon 图片需签协议邮件申请，基础模型与依赖须自行配置。
- 知识节点：wiki/entities/paper-aesformer.md；交叉引用：wiki/methods/grpo.md。
- 校验范围：arXiv v1 全文；官方 README；Stage 1 规划与 Stage 2 编辑脚本；未在本机运行模型训练/推理。

## [2026-11-15] ingest | 补齐本次 lint PR 涉及节点的原文通读、结构与工程实践；新增 StarVLA-α 独立论文实体，复核硬件官方运行入口及论文评测边界，使 41 个 wiki 节点健康度全部达到 3/3，并挂载 4 条健康度 3/3 的既有纵深路线。

## [2026-11-15] lint | 全量健康检查与预警修复

- 基线：`main` 的 `3f90b97e0f9b57ad197287d9cb0167367689a4e6`，包含本日首页路线预览与星标统计更新。
- 基线报告：17 处来源时效问题、29 条信息型预警。缺页候选在报告中只展示前 15 项，本次展开并逐项核对全部 23 项。
- 来源复核：为 H-JEPA/HWM、LEGO/LEGS 与感知链、LeWM/LpWM/INTACT、LG-DKGP/MPC、ODEWorld/V-JEPA 2、OVO、Faster-WAM、RoboHarness、SPU 和 StarVLA 等关联补齐版本、协议与接口边界。
- 历史入口：复核超过 180 天的 5 页，归档 myCobot、Fourier、Kinova、Poppy 的官方说明；区分文档与 SDK 源码、本体与 ROS 配置、软硬件许可及 CAD LFS 获取方式。
- 结构与措辞：将 NavWAM 已有对照表升为正式对比章节，补充 7 处主题枢纽回链；将 4 处“最新”表述绑定到发布配置或明确版本。
- 检查器：登记已有实体/缩写及命令字段的语义归属；科学湿实验预测基准留在 AI Agent 评测链，带机器人标签的跨域基准继续要求具身枢纽；增加正反回归用例。
- 验证：完整 Python 测试 596 项与 843 项子测试通过，覆盖率 63.16%；同步上游后，前端 138 项和受影响的 13 项 Python 测试通过。完整门禁及逐页健康度见 PR 验证记录。

## [2026-11-14] ingest | roadmap/depth-vla.md — 补齐 VLA 纵深路线全部 80 个 Stage wiki 节点原文证据与满分健康度

## [2026-11-14] ingest | sources/blogs/wechat_guyueju_perceptive_nmpc_convex_decomposition_2026-11-14.md — 感知型 NMPC 高程图凸分解与 Go2 源码核对

- **意图：** 消化用户提供的古月居技术解读 PDF，建立可溯源的感知型 NMPC / 凸踏面几何约束工程入口。
- **开源核验：** Grandia 等原始方法见 IEEE T-RO 2023（ANYmal）；Go2 适配补丁和 ROS 2/MuJoCo 工具见 Khaledsweid1/ocs2-perceptive-mpc-go2（公开）。
- **重要边界：** Go2 工程的落足区域约束默认关闭，指南说明尚不能完整爬完楼梯；模拟射线高程图非真机感知，附件参数来自文章与仓库文档，未独立复测。
- **沉淀：** [Go2 感知型 NMPC 实体](../wiki/entities/ocs2-perceptive-mpc-go2.md)，并接入 [感知越障纵深路线](../roadmap/depth-perceptive-locomotion.md)。

## [2026-11-14] ingest | 完成 PR #3732（Issue #3487）涉及全部 63 个知识节点的原文与官方源码通读、Mermaid 项目流程图与源码运行时序图补齐，修正 SRU 404 仓库链接为 leggedrobotics/sru-pytorch-spatial-learning，63 页节点健康度全部升至 3/3

## [2026-11-14] structural | wiki — 完成 #3487 全量 277 页候选语义核验：回填 55 页、排除 217 页、待定 5 页；保留首批 8 页并恢复原 updated，核验记录见 docs/quality/issue-3487-code-backfill.md

## [2026-11-13] lint | wiki/entities — 回填第一批 8 个实体页官方 GitHub 仓库 code 字段 (#3487)

## [2026-11-14] ingest | sources/papers/science_prediction_universe_arxiv_2611_15049.md — 收录 UniPat AI Science Prediction Universe；归档论文、官方站点与仓库，新增独立论文详情节点并互链 AI Agent 评测。

## [2026-11-14] ingest | sources/blogs/ultralytics_yolo26_ocr_python_2026_11_12.md — 入库 YOLO26+PP-OCRv6 OCR 与 HierText 数据集

- 以 Ultralytics 英文原文、官方 OCR Guide 和 `ultralytics/solutions/ocr.py` 为依据，新增逐词识别、旋转矫正、阅读顺序、PDF 工程入口及源码时序图。
- 整理 HierText 的分层标注与评测协议，并单独记录 Signature 签名定位数据集；将作者基准明确标识为非独立复测。
- 用户的 bit.ly 短链无法直接解析，原样保留并给出独立核实的官方对应资料，不冒认重定向结果。
- 官方框架代码已公开，PP-OCRv6 模型由 assets 下载；在线 Demo 上传页面，应注意隐私与许可证。

## [2026-11-14] ingest | 2026 BEHAVIOR Challenge 赛制、评测协议与源码核对

- **意图：** 按 CoRL 2026 官方一手资料补全已有 2026 BEHAVIOR Challenge 来源归档，增加独立赛事节点，不重复建设 BEHAVIOR-1K 基准或 HiPhys-RSI 冠军论文。
- **归档：** sources/sites/behavior-challenge.md 与 sources/repos/behavior-1k-challenge.md。
- **wiki：** wiki/entities/behavior-challenge-2026.md；评测选型 hub 添加内部入口。
- **开源结论：** 赛事环境与基线开放、示范数据公开；冠军方案并不包含在赛事代码库。官网要求 v3.9.3，但本次源码接口只成功读到 main，版本对齐与端到端复现待办。

## [2026-11-14] ingest | AHEADance 官方项目页复核与复现边界

- 意图：用户再次提出 AHEADance 入库；按 arXiv:2611.13370 在库内查重，复用现有唯一论文实体，不重复创建节点。
- 来源：arXiv v1 完整论文和 UCL Humanoid Robotics Lab 官方项目主页（https://ihcr.github.io/AHEADance/）。
- 结论：ListenAhead 与 Commit Forcing 的既有论文分析可保留；补录官方网页真机视频、麦克风演示、MuJoCo 回放和云端 Live 的开放范围。
- 开源：项目页仍为 “Code soon”，未发布可复现的 AHEADance 研究训练/推理源码或评测入口，源码运行时序图不适用。
- 更新：`sources/sites/aheadance_project.md`、`sources/papers/aheadance_arxiv_2611_13370.md` 与 `wiki/entities/paper-aheadance.md`；流程图补全机器人测量状态仅反馈 SONIC 跟踪层。

## [2026-11-14] ingest | LG-DKGP 连续性感知世界模型规划：原文、代码、时序与指标口径

- 新增论文原文归档 sources/papers/lg_dkgp_arxiv_2611_16645.md、官方开源仓库归档 sources/repos/lg-dkgp.md，以及独立论文实体 wiki/entities/paper-lg-dkgp.md。
- 核对 arXiv:2611.16645v1 全文、附录及官方 CLI/训练/规划/评测源码：门控历史注意力 → 神经均值 + GP 残差 → 专家动作增量探索尺度 → 冻结 LeWM + CEM。
- 主文 Macro 成功率 87.07%；明确区分 PushT/Cube 原生成功与 LIBERO/RoboTwin 局部目标口径，附上流程图和真实代码运行时序图。
- 开源状态：源码与评测配方已发布；模型权重、大型训练数据、原生仿真 fixtures 分别获取。当前环境未实际训练或复现模拟器成绩。

## [2026-11-14] ingest | sources/sites/steel-skild-ai.md — 复核 Steel 的完整异构基准与复现实验边界

- 2026-11-09 Skild AI × NVIDIA 官方发布的 Steel 已于先前 ingest 建立唯一实体；本次复用原页，不重复建节点。
- 核对并补录 5/10/15/20 种混合任务吞吐表、20 类任务 matched-batch 12.0×、约 62% 显存节约。
- 澄清 `reuse_kinematics=True` 仅用于单任务对照，异构批次因 joint velocity limits 未启用；补充 PhysX 6.1 与 6.0 的版本敏感性、5 ms 物理步和单卡评测条件。
- 更新 Newton 求解器目录和 Steel 双向知识链接；性能数字均源于作者发布，未独立运行基准。

## [2026-11-14] ingest | ATLAS × SuperMimic：真机在线模型学习与 BPTT，核对网站代码开放边界

- 在已合并的 ATLAS 项目实体上补充与 SuperMimic 的方法关系：同为通过学习的可微动力学反传优化，但 ATLAS 从真实尝试学习末端残差、冻结 ScaleBFM 全身跟踪基座，不复用 SuperMimic 代码的证据。
- 复核 ATLAS 网站源码仓库，其 tools 是网页概念动画与视频渲染辅助脚本，不是 ATLAS 的可复现模型训练、部署与真机控制源码。
- 已读官方技术博客与项目页；现有节点避免重复，保留 Mermaid 机制图并细化在线模型学习与 BPTT 数据流。

## [2026-11-14] lint | 全量健康检查与预警修复

- 基线：`main` 的 `57c0b29c3c29190e93b4337d2186d72a9e2b3d1e`，包含 RTC、轮式双臂 URDF 的合入更新，以及随后机器人更新的星标统计。
- 来源复核：区分训练期/推理期 RTC 源码分支；修正生成式世界模型页的 W0 资源发布状态；补充 UnifoLM 与 W0 的视频/动作接口对照，以及 EgoXperience 与 Xperience 2 的重建/场景数据边界。复核后更新四页日期并保留来源链接。
- 结构与回链：补齐 DexTacWAM、Flash ALOHA 的对比章节及综述预印本发布元数据；给 VINO、Volt 和智驾/具身对照页补充相应知识链枢纽回链。
- 检查器：ATLAS 候选混合多个已有同名实体，登记已覆盖的不同语义；回归测试同时确认真正缺页的机制仍会被报告。
- 搜索：CLF 定义页标题补齐通用缩写，恢复「CLF Lyapunov 衰减 稳定性」查询的前五命中，不修改搜索权重或回归门槛。
- 验证：完整 Python 测试 571 项及 843 项子测试通过，覆盖率 63.16%；前端 130 项测试通过。完整门禁与最终计数见 PR 验证记录。

## [2026-11-13] ingest | sources/sites/lightwheel_robofinals_rl_env.md — 核验光轮 RoboFinals RL-Env 官方公告，建立独立训练环境节点并与 RoboFinals 评测双向关联（官网产品全文待补核）

## [2026-11-13] ingest | 关联 #3490：通读并核对 WAM 纵深路线的主要原文与公开运行入口，补齐 51 个节点的原文证据、评测口径及运行图，62 个 Stage 节点全部达到 3/3；更新全局健康度待办清单。

## [2026-11-14] ingest | RTC / arXiv:2506.07339 | 对齐论文与官方仿真源码、补充流程与时序图

- 资料：arXiv v2（NeurIPS 2025）、PI 官方研究说明、Physical-Intelligence/real-time-chunking-kinetix README 及核心训练/评测代码。
- 去重：复用既有 `wiki/entities/paper-real-time-chunking.md`（论文 ID 2506.07339），不创建重复实体。
- 澄清：PI research URL 是项目介绍页；官方可运行源码范围为 Kinetix 仿真，不等于 π₀.₅ 真机 RTC 部署仓。
- 修改：论文页补 `project_id` 和中英全称、Mermaid 论文流程图，修订为代码可追溯的 Kinetix 运行时序图；补充论文、仿真仓来源核对记录。
- 验证边界：使用论文 v2 HTML、官方博客和仓库源码静态核对，未本地执行完整多 GPU 复现；PR 依赖 GitHub Actions 进行完整 lint。

## [2026-11-14] ingest | sources/blogs/zane-hub-wheeled-dual-arm-urdf-workflow-2026-11-14.md — 收录轮式双臂人形 URDF 工程工具链

- 来源：Zane Zhang / Zane Hub 2026-11-14 工程文章，用户上传 20 页 PDF，逐页核对。
- 新建：wiki/queries/wheeled-dual-arm-humanoid-urdf-workflow.md；从 wiki/concepts/urdf-robot-description.md 建立入链。
- 内容：七阶段工作流、Mermaid 流程图、工具链和验收表；标注惯性 rpy、单位换算、mimic 与格式转换的原文边界。

## [2026-11-13] ingest | 第 86 期组队学习 21 项来源核对

- 输入：用户提供课程表截图，按图片顺序逐项匹配。21 条课程对应 17 个候选项目 / 书籍身份。
- 查重：Agentic RL Lab 与 Easy-Vibe 已有 sources/wiki 节点，未复制；llm-algo-leetcode 四方向及 Hello-Agents 入门/进阶分别合并项目身份。
- 核验：建立总来源表与 13 个新资料归档（含沐曦官方活动课表、黄佳《企业本体论》配套代码、范冰 FDE 公开书）。
- 阻塞：Nailongbot 和图片所称「AI for Science 科学模型与科研智能体实战」尚无准确对应的一手课程仓库；不借用无关仓库充数。
- 阅读：已通读沐曦官方活动介绍；其他教程仅读 README / 目录，未完整精读各章节和源码，归档标注通读未完成。
- 健康度：本轮未新增或修改 wiki 实体页，不影响现有节点分数；因未完成全面深读，不标记任何新节点为 3/3。
- 复核后续：逐章通读、源码入口确认、Mermaid 主流程及运行时序、去重后的 wiki 节点建档、ci-preflight/health 报告。本 PR 应保持草稿，直到满足门禁。

## [2026-11-13] ingest | Whole-Body Dexterity 全文深读与原有节点升级

- 复用 2026-10 已合入的论文身份（arXiv:2610.22184），不重复建节点。
- 核对 arXiv v1 正文及附录 A–G，补录 251 万稳定抓取、27.4 万轨迹、8.73 亿帧、PPO 专家与扩散/逆动力学蒸馏链路。
- 补充 Unitree G1 真机 45/50、模拟 EMA 对比、RTX 5090 推理延迟与实验边界。
- 作者项目页截至 2026-11-13 未列可运行源码/数据下载入口，源码运行时序图不适用。
- 更新 `wiki/entities/paper-whole-body-dexterity.md` 的原文通读、图谱主流程和健康度字段。

## [2026-11-13] ingest | TRABot 原文 v2 复核并修正规划参数

- 论文：Talk, Render, Act: Integrating Social Gesture and Digital Face with Synchronized Speech for Conversational Humanoid Robot（arXiv:2610.06153v2）。
- 查重：已有 wiki/entities/paper-trabot-talk-render-act.md 与对应来源，故原位升级，不重复创建论文节点。
- 阅读：完成 arXiv HTML v2 摘要、I–V 全文章节、式 1–14、算法、表 I–VI、参考文献；对照官方项目页。
- 核验：机构为阿联酋大学计算与人工智能学院；最小加加速度桥接速度 **1 rad/s**，纠正原记录误写的 **11 rad/s**。
- 内容：补英中论文标题、机构标签、全文通读状态与 original 映射、12 组对话中的具体样例、规划代价/原子执行机制与五条件排序口径；Mermaid 主流程图按论文重新核对。
- 开源：官方项目仍标注 Code (soon)，无 TRABot 代码入口；未编造源码运行时序。
- 变更：wiki/entities/paper-trabot-talk-render-act.md；sources/papers/trabot_arxiv_2610_06153.md；sources/sites/trabot-project.md；log.d/2026-11-13-194700-ingest-trabot-v2-source-review.md。不触碰 catalog.md、log.md 与生成文件。

## [2026-11-13] ingest | sources/blogs/atlas_humanoid_early_release.md — 收录 ATLAS System 2 / System 0 与在线残差学习架构

- 已通读项目官方 Early release 技术长文及主页的机制、实验、部署与局限。
- 创建 ATLAS 唯一项目实体、Mermaid 控制流图，链接到 Loco-Manipulation 技术地图。
- 截至 2026-11-13，项目页未列 ATLAS 可运行源码、权重或论文 PDF；不虚构源码时序或 arXiv 身份。

## [2026-11-13] ingest | Embodied First Exam 论文协议、九套件与失败阶段深化

- 复用已入库的 arXiv:2611.12177 唯一实体 `wiki/entities/embodied-first-exam.md`，不重复建立同论文第二个节点。
- 根据论文 v1（正文 §1–5、附录 A–K 的既有通读记录，以及本次 Figure 3、§2–3、Appendix I–J 原文复核），补充三种控制路径、三类模式、九个原创套件与 Mermaid 对应数据流。
- 澄清 456 核心 Standard 33.6%、391 配对任务 Standard 30% vs Privileged 77%、391 受控消融 28% → 45% / 53% / 68%，避免跨分母误读。
- 补充阶段失败、Realtime 配对、harness 差异、真机演示边界；按 2026-11-13 官方 README 更新代码仓扩增至 36 benchmarks / 1,211 tasks 与论文 35/1,156 区分。
- 保留原先逐文件核对的官方源码时序图；资料归档更新 `sources/papers/`、`sources/repos/` 与 `sources/sites/`。

## [2026-11-13] ingest | sources/papers/ropedia_xperience_2_arxiv_2611_17060.md — 收录 Ropedia Xperience 2 环境与行为数据基座

- 通读 arXiv v1 正文及重要附录，归档官方项目、数据集、配准源码，并建立统一论文实体节点。
- 区分 1,589 个保留扫描与 Hugging Face 首批 126 个场景的分批上传状态；区分采集清单与可下载内容。
- 通过源码确认仅配准工具已公开可执行，并补主干 Mermaid 与配准运行时序图。

## [2026-11-13] ingest | sources/papers/dextacwam_arxiv_2609_24976.md — DexTacWAM 视触觉世界—动作模型

- 核对论文 `1–6 与附录 A.1–A.11，单一 wiki 实体收录跨指触觉压缩、预测式接触世界模型、六任务实验及局限。
- 核实代码及权重开放边界：三阶段训练与离线推理代码、Stage 1 触觉编码器与数据公开；作者任务专属 Stage 2/3 checkpoint 未发布；部署脚本在 deploy 分支。
- 为节点添加方法 Mermaid 流程图和源码训练时序图，并从触觉感知概念页建立入链。

## [2026-11-13] ingest | TipSim 触觉仿真平台：官方技术文档、闭环示例与许可证核实

- 收录华威科 TIP 的 MuJoCo Loong/Universa 触觉仿真，补充双语实体介绍、Mermaid 接触/反馈流程、Python 示例运行时序。
- 核对核心触觉插件只提供预编译二进制，示例与场景 MIT；双指力阈值抓取不代表真机 Sim2Real 已验证。
- 关联 [触觉感知](../wiki/concepts/tactile-sensing.md)；保持本次 PR 仅源文件，不触碰自动生成的 catalog/log 和前端统计。

## [2026-11-13] ingest | Kimodo-Policy 官方资源与源码闭环核验

- 意图：复用现有 `wiki/entities/paper-kimodo-policy.md`，避免重复项目节点；补全论文全体作者署名。
- 一手来源：arXiv:2611.11554v1 全文（既有归档已通读）；本次增量核对官方 GitHub README、训练/评测核心入口与真机部署脚本。
- 开源结论：官方发布了预训练、HumanoidArena、SIMPLE、真机 G1 + Dex3 数据/检查点/部署流程；BiGym 2.0、HumanoidToolBench 仍为计划项。
- 写回：`wiki/entities/paper-kimodo-policy.md`、`sources/papers/kimodo_policy_primary.md`、`sources/repos/kimodo-policy-official.md`；区分模型生成、SONIC 跟踪、两套仿真环境与实测成绩；记录未独立复现。

## [2026-11-13] lint | 基于 main@7dd940a4 全量清理 Wiki 阻塞及信息预警，补齐论文阅读结构并复核来源更新

- 基线：`main@7dd940a4dd73021b9a80b10d36f6d71a36b8cca1`；全库 7,715 个 wiki 页面。
- 原始明细：64 条阻塞预警、197 条信息预警。旧摘要把 82 条缺结论合为一个桶，因此显示的信息预警总数为 116；本次按真实明细处理。
- 补齐 82 页结论（总判 + 三条操作要点）；90 页整理已有评测/对比标题，未把摘要页改为已通读。补 3 页来源元数据、2 页数据集维度和 2 页评测枢纽回链。
- 7 个超过 180 天的页面已核对官方 README，固定版本和未运行边界写在对应 wiki 页。
- `exec` 是执行符号，RTC 已由动作块方法和两个论文实体覆盖；书名号内的资料标题和明确否认用词不是时效性性能断言。识别修复附真阳性保留测试，未降低阈值或关闭规则。

### 64 个来源更新预警的复核记录

核对范围为库内来源归档的更新摘录、修订说明与对应 wiki 的既有定义/职责及交叉引用。以下日期表示此次维护复核，不表示重新通读了所有被引论文或运行了实现；既有 `original`、`status` 和原文阅读记录均保留。除表中明确纠正的内容外，保留页面定义，更新 `updated` 为复核日。

| 页面 | 触发来源 | 复核结论 / 保留边界 |
|---|---|---|
| [wiki/methods/action-chunking.md](../wiki/methods/action-chunking.md) | [behavior_2026_solution_arxiv_2611_14432.md](../sources/papers/behavior_2026_solution_arxiv_2611_14432.md) | 动作块与仿真 tick / 控制时钟分离；不把挑战赛 Q-score 当完整成功率 |
| [wiki/concepts/contact-rich-manipulation.md](../wiki/concepts/contact-rich-manipulation.md) | [chord_nvidia_video_to_data_2026.md](../sources/papers/chord_nvidia_video_to_data_2026.md) | 接触力旋量奖励、物理重定向及 Isaac Lab 的职责保持区分 |
| [wiki/tasks/manipulation.md](../wiki/tasks/manipulation.md) | [chord_nvidia_video_to_data_2026.md](../sources/papers/chord_nvidia_video_to_data_2026.md) | 接触力旋量奖励、物理重定向及 Isaac Lab 的职责保持区分 |
| [wiki/methods/spider-physics-informed-dexterous-retargeting.md](../wiki/methods/spider-physics-informed-dexterous-retargeting.md) | [chord_nvidia_video_to_data_2026.md](../sources/papers/chord_nvidia_video_to_data_2026.md) | 接触力旋量奖励、物理重定向及 Isaac Lab 的职责保持区分 |
| [wiki/queries/dexterous-manipulation-data-pipeline.md](../wiki/queries/dexterous-manipulation-data-pipeline.md) | [chord_nvidia_video_to_data_2026.md](../sources/papers/chord_nvidia_video_to_data_2026.md) | 接触力旋量奖励、物理重定向及 Isaac Lab 的职责保持区分 |
| [wiki/entities/isaac-lab.md](../wiki/entities/isaac-lab.md) | [chord_nvidia_video_to_data_2026.md](../sources/papers/chord_nvidia_video_to_data_2026.md) | 接触力旋量奖励、物理重定向及 Isaac Lab 的职责保持区分 |
| [wiki/entities/cosmos-3.md](../wiki/entities/cosmos-3.md) | [cosmos_interactive_world_simulation_tech_report_2026.md](../sources/papers/cosmos_interactive_world_simulation_tech_report_2026.md) | Cosmos 3 基座与交互世界仿真后训练报告分别引用 |
| [wiki/entities/sharpa-robotics.md](../wiki/entities/sharpa-robotics.md) | [dex-x_arxiv_2609_07747.md](../sources/papers/dex-x_arxiv_2609_07747.md) | 保留 Sharpa 硬件角色；Dex-X 的视觉触觉策略不等于公司全部技术栈 |
| [wiki/methods/egoscale.md](../wiki/methods/egoscale.md) | [ego2robot_arxiv_2608_02580.md](../sources/papers/ego2robot_arxiv_2608_02580.md) | Ego2Robot 合成、EgoScale 预训练、EgoVerse 数据生态与 RoboTwin 平台分别定位 |
| [wiki/entities/robotwin.md](../wiki/entities/robotwin.md) | [ego2robot_arxiv_2608_02580.md](../sources/papers/ego2robot_arxiv_2608_02580.md) | Ego2Robot 合成、EgoScale 预训练、EgoVerse 数据生态与 RoboTwin 平台分别定位 |
| [wiki/entities/paper-egoverse.md](../wiki/entities/paper-egoverse.md) | [ego2robot_arxiv_2608_02580.md](../sources/papers/ego2robot_arxiv_2608_02580.md) | Ego2Robot 合成、EgoScale 预训练、EgoVerse 数据生态与 RoboTwin 平台分别定位 |
| [wiki/concepts/motion-retargeting.md](../wiki/concepts/motion-retargeting.md) | [egoscale_arxiv_2602_16710.md](../sources/papers/egoscale_arxiv_2602_16710.md) | 人类腕/手标签仍需目标机器人重定向与对齐，不能把 HumanNet 规模当可执行动作规模 |
| [wiki/entities/humannet.md](../wiki/entities/humannet.md) | [egoscale_arxiv_2602_16710.md](../sources/papers/egoscale_arxiv_2602_16710.md) | 人类腕/手标签仍需目标机器人重定向与对齐，不能把 HumanNet 规模当可执行动作规模 |
| [wiki/entities/painode-332-eurekanvidia.md](../wiki/entities/painode-332-eurekanvidia.md) | [eureka-llm-reward-design_arxiv_2310_12931.md](../sources/papers/eureka-llm-reward-design_arxiv_2310_12931.md) | Eureka 迭代奖励代码；官方说明与配置默认值不同，保留实体页的复现边界 |
| [wiki/entities/paper-expo.md](../wiki/entities/paper-expo.md) | [expo_ft_arxiv_2605_25477.md](../sources/papers/expo_ft_arxiv_2605_25477.md) | EXPO 是方法前作；EXPO-FT 的 VLA 真机结论不回填为前作通用保证 |
| [wiki/entities/paper-dreamwam.md](../wiki/entities/paper-dreamwam.md) | [fact_arxiv_2608_10232.md](../sources/papers/fact_arxiv_2608_10232.md) | FACT 的因果动作后果预测与 DreamWAM 的多模态联合预测分别定位 |
| [wiki/overview/robot-world-models-action-consequence-technology-map.md](../wiki/overview/robot-world-models-action-consequence-technology-map.md) | [fact_arxiv_2608_10232.md](../sources/papers/fact_arxiv_2608_10232.md) | FACT 的因果动作后果预测与 DreamWAM 的多模态联合预测分别定位 |
| [wiki/overview/embodied-station-2026-10-14-41-papers-technology-map.md](../wiki/overview/embodied-station-2026-10-14-41-papers-technology-map.md) | [freebar_barrier_guidance_async_vla_arxiv_2610_14997.md](../sources/papers/freebar_barrier_guidance_async_vla_arxiv_2610_14997.md) | FreeBar 属推理期动作块边界约束，保留专题导航定位 |
| [wiki/concepts/tactile-sensing.md](../wiki/concepts/tactile-sensing.md) | [ftp-1_arxiv_2606_13102.md](../sources/papers/ftp-1_arxiv_2606_13102.md) | FTP-1 为跨触觉传感器策略；触觉测量不直接等于跨硬件控制接口 |
| [wiki/tasks/loco-manipulation.md](../wiki/tasks/loco-manipulation.md) | [grail_arxiv_2606_05160.md](../sources/papers/grail_arxiv_2606_05160.md) | GRAIL 的数字合成、SONIC tracker、视觉策略与约 22k 公开数据集逐层区分 |
| [wiki/methods/sonic-motion-tracking.md](../wiki/methods/sonic-motion-tracking.md) | [grail_arxiv_2606_05160.md](../sources/papers/grail_arxiv_2606_05160.md) | GRAIL 的数字合成、SONIC tracker、视觉策略与约 22k 公开数据集逐层区分 |
| [wiki/entities/paper-viral-humanoid-visual-sim2real.md](../wiki/entities/paper-viral-humanoid-visual-sim2real.md) | [grail_arxiv_2606_05160.md](../sources/papers/grail_arxiv_2606_05160.md) | GRAIL 的数字合成、SONIC tracker、视觉策略与约 22k 公开数据集逐层区分 |
| [wiki/entities/paper-hrl-stack-34-gr00t_n1.md](../wiki/entities/paper-hrl-stack-34-gr00t_n1.md) | [grail_arxiv_2606_05160.md](../sources/papers/grail_arxiv_2606_05160.md) | GRAIL 的数字合成、SONIC tracker、视觉策略与约 22k 公开数据集逐层区分 |
| [wiki/entities/grail-locomanipulation-dataset.md](../wiki/entities/grail-locomanipulation-dataset.md) | [grail_arxiv_2606_05160.md](../sources/papers/grail_arxiv_2606_05160.md) | GRAIL 的数字合成、SONIC tracker、视觉策略与约 22k 公开数据集逐层区分 |
| [wiki/overview/loco-manip-161-category-03-visuomotor.md](../wiki/overview/loco-manip-161-category-03-visuomotor.md) | [grail_arxiv_2606_05160.md](../sources/papers/grail_arxiv_2606_05160.md) | GRAIL 的数字合成、SONIC tracker、视觉策略与约 22k 公开数据集逐层区分 |
| [wiki/overview/motion-cerebellum-category-08-real-tasks.md](../wiki/overview/motion-cerebellum-category-08-real-tasks.md) | [grail_arxiv_2606_05160.md](../sources/papers/grail_arxiv_2606_05160.md) | GRAIL 的数字合成、SONIC tracker、视觉策略与约 22k 公开数据集逐层区分 |
| [wiki/overview/loco-manip-contact-category-03-generative-data.md](../wiki/overview/loco-manip-contact-category-03-generative-data.md) | [grail_arxiv_2606_05160.md](../sources/papers/grail_arxiv_2606_05160.md) | GRAIL 的数字合成、SONIC tracker、视觉策略与约 22k 公开数据集逐层区分 |
| [wiki/entities/article-videodb-jepa-world-models.md](../wiki/entities/article-videodb-jepa-world-models.md) | [hierarchical_planning_latent_world_models_arxiv_2604_03208.md](../sources/papers/hierarchical_planning_latent_world_models_arxiv_2604_03208.md) | HWM 是分层规划扩展，不把 JEPA 表征本身等同控制策略 |
| [wiki/entities/paper-vjepa2.md](../wiki/entities/paper-vjepa2.md) | [hierarchical_planning_latent_world_models_arxiv_2604_03208.md](../sources/papers/hierarchical_planning_latent_world_models_arxiv_2604_03208.md) | HWM 是分层规划扩展，不把 JEPA 表征本身等同控制策略 |
| [wiki/entities/behavior-1k.md](../wiki/entities/behavior-1k.md) | [hiphys-rsi-scaling-by-structure-2026.md](../sources/papers/hiphys-rsi-scaling-by-structure-2026.md) | BEHAVIOR 是评测环境；HiPhys-RSI 冠军方案不等于基准自带能力 |
| [wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md](../wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md) | [hire_hindsight_reward_editing_arxiv_2609_27068.md](../sources/papers/hire_hindsight_reward_editing_arxiv_2609_27068.md) | HiRE 是奖励编辑方法，专题页保留其与其他奖励路线的导航关系 |
| [wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md](../wiki/overview/paper-notebook-category-04-loco-manipulation-and-wbc.md) | [humanoid_pnb_meshmimic.md](../sources/papers/humanoid_pnb_meshmimic.md) | MeshMimic 对接运动地形重建，分类索引不承担论文独立实验结论 |
| [wiki/overview/hub-embodied-eval-benchmark.md](../wiki/overview/hub-embodied-eval-benchmark.md) | [imitator_game_arxiv_2608_22301.md](../sources/papers/imitator_game_arxiv_2608_22301.md) | Imitator Game 按场景失配分级；LeRobot 仍是实现基础设施 |
| [wiki/entities/lerobot.md](../wiki/entities/lerobot.md) | [imitator_game_arxiv_2608_22301.md](../sources/papers/imitator_game_arxiv_2608_22301.md) | Imitator Game 按场景失配分级；LeRobot 仍是实现基础设施 |
| [wiki/entities/molmo-er.md](../wiki/entities/molmo-er.md) | [molmoact2_arxiv_2605_02881.md](../sources/papers/molmoact2_arxiv_2605_02881.md) | Molmo2-ER 为推理骨干，MolmoAct2 另含连续动作专家 |
| [wiki/methods/crisp-real2sim.md](../wiki/methods/crisp-real2sim.md) | [prism_real2sim2real_arxiv_2609_38172.md](../sources/papers/prism_real2sim2real_arxiv_2609_38172.md) | PRISM 接触锚定 Real2Sim、CRISP 场景重建与 DAgger 数据聚合分别定位 |
| [wiki/methods/dagger.md](../wiki/methods/dagger.md) | [prism_real2sim2real_arxiv_2609_38172.md](../sources/papers/prism_real2sim2real_arxiv_2609_38172.md) | PRISM 接触锚定 Real2Sim、CRISP 场景重建与 DAgger 数据聚合分别定位 |
| [wiki/entities/videomimic.md](../wiki/entities/videomimic.md) | [prism_real2sim2real_arxiv_2609_38172.md](../sources/papers/prism_real2sim2real_arxiv_2609_38172.md) | PRISM 接触锚定 Real2Sim、CRISP 场景重建与 DAgger 数据聚合分别定位 |
| [wiki/methods/grpo.md](../wiki/methods/grpo.md) | [r3_robotic_reasoner_arxiv_2608_26053.md](../sources/papers/r3_robotic_reasoner_arxiv_2608_26053.md) | R³ 的单步量表 RL 接口与 GRPO 组内相对优化机制保持区分 |
| [wiki/concepts/motion-control-policy-evaluation-metrics.md](../wiki/concepts/motion-control-policy-evaluation-metrics.md) | [react_arxiv_2611_13929.md](../sources/papers/react_arxiv_2611_13929.md) | 负载导航成功与分方向速度/航向误差均须报告 |
| [wiki/methods/diffusion-policy.md](../wiki/methods/diffusion-policy.md) | [robofft_arxiv_2609_32236.md](../sources/papers/robofft_arxiv_2609_32236.md) | RoboFFT 的 matching-loss 比率为代理，不当作 PPO 精确似然比 |
| [wiki/methods/ppo.md](../wiki/methods/ppo.md) | [robofft_arxiv_2609_32236.md](../sources/papers/robofft_arxiv_2609_32236.md) | RoboFFT 的 matching-loss 比率为代理，不当作 PPO 精确似然比 |
| [wiki/methods/reinforcement-learning.md](../wiki/methods/reinforcement-learning.md) | [sdpg_visual_rl_arxiv_2605_26478.md](../sources/papers/sdpg_visual_rl_arxiv_2605_26478.md) | SDPG 的物理辅助环境与渲染环境分开计数；不改变其他视觉蒸馏框架定位 |
| [wiki/entities/genesis-sim.md](../wiki/entities/genesis-sim.md) | [sdpg_visual_rl_arxiv_2605_26478.md](../sources/papers/sdpg_visual_rl_arxiv_2605_26478.md) | SDPG 的物理辅助环境与渲染环境分开计数；不改变其他视觉蒸馏框架定位 |
| [wiki/methods/policy-optimization.md](../wiki/methods/policy-optimization.md) | [sdpg_visual_rl_arxiv_2605_26478.md](../sources/papers/sdpg_visual_rl_arxiv_2605_26478.md) | SDPG 的物理辅助环境与渲染环境分开计数；不改变其他视觉蒸馏框架定位 |
| [wiki/tasks/locomotion.md](../wiki/tasks/locomotion.md) | [sdpg_visual_rl_arxiv_2605_26478.md](../sources/papers/sdpg_visual_rl_arxiv_2605_26478.md) | SDPG 的物理辅助环境与渲染环境分开计数；不改变其他视觉蒸馏框架定位 |
| [wiki/entities/gr00t-visual-sim2real.md](../wiki/entities/gr00t-visual-sim2real.md) | [sdpg_visual_rl_arxiv_2605_26478.md](../sources/papers/sdpg_visual_rl_arxiv_2605_26478.md) | SDPG 的物理辅助环境与渲染环境分开计数；不改变其他视觉蒸馏框架定位 |
| [wiki/methods/sumo.md](../wiki/methods/sumo.md) | [smpc2rl_arxiv_2608_12063.md](../sources/papers/smpc2rl_arxiv_2608_12063.md) | SMPC-to-RL 是 MPC 专家到 RL 策略的训练方向，与 Sumo 运行时层级相区分；judo 非官方实现 |
| [wiki/comparisons/mpc-vs-rl.md](../wiki/comparisons/mpc-vs-rl.md) | [smpc2rl_arxiv_2608_12063.md](../sources/papers/smpc2rl_arxiv_2608_12063.md) | SMPC-to-RL 是 MPC 专家到 RL 策略的训练方向，与 Sumo 运行时层级相区分；judo 非官方实现 |
| [wiki/comparisons/online-vs-offline-rl.md](../wiki/comparisons/online-vs-offline-rl.md) | [smpc2rl_arxiv_2608_12063.md](../sources/papers/smpc2rl_arxiv_2608_12063.md) | SMPC-to-RL 是 MPC 专家到 RL 策略的训练方向，与 Sumo 运行时层级相区分；judo 非官方实现 |
| [wiki/entities/mjlab.md](../wiki/entities/mjlab.md) | [smpc2rl_arxiv_2608_12063.md](../sources/papers/smpc2rl_arxiv_2608_12063.md) | SMPC-to-RL 是 MPC 专家到 RL 策略的训练方向，与 Sumo 运行时层级相区分；judo 非官方实现 |
| [wiki/methods/model-predictive-control.md](../wiki/methods/model-predictive-control.md) | [smpc2rl_arxiv_2608_12063.md](../sources/papers/smpc2rl_arxiv_2608_12063.md) | SMPC-to-RL 是 MPC 专家到 RL 策略的训练方向，与 Sumo 运行时层级相区分；judo 非官方实现 |
| [wiki/entities/paper-notebook-dpl-depth-only-perceptive-humanoid-locomotion-vi.md](../wiki/entities/paper-notebook-dpl-depth-only-perceptive-humanoid-locomotion-vi.md) | [solo_arxiv_2608_26583.md](../sources/papers/solo_arxiv_2608_26583.md) | SOLO 的 QR / TA-MSE、DPL / AME / SSR 的感知路线分开，保留教师学生边界 |
| [wiki/concepts/privileged-training.md](../wiki/concepts/privileged-training.md) | [solo_arxiv_2608_26583.md](../sources/papers/solo_arxiv_2608_26583.md) | SOLO 的 QR / TA-MSE、DPL / AME / SSR 的感知路线分开，保留教师学生边界 |
| [wiki/entities/paper-ame-attention-based-map-encoding.md](../wiki/entities/paper-ame-attention-based-map-encoding.md) | [solo_arxiv_2608_26583.md](../sources/papers/solo_arxiv_2608_26583.md) | SOLO 的 QR / TA-MSE、DPL / AME / SSR 的感知路线分开，保留教师学生边界 |
| [wiki/tasks/stair-obstacle-perceptive-locomotion.md](../wiki/tasks/stair-obstacle-perceptive-locomotion.md) | [solo_arxiv_2608_26583.md](../sources/papers/solo_arxiv_2608_26583.md) | SOLO 的 QR / TA-MSE、DPL / AME / SSR 的感知路线分开，保留教师学生边界 |
| [wiki/entities/paper-ssr-humanoid-open-world-traversal.md](../wiki/entities/paper-ssr-humanoid-open-world-traversal.md) | [solo_arxiv_2608_26583.md](../sources/papers/solo_arxiv_2608_26583.md) | SOLO 的 QR / TA-MSE、DPL / AME / SSR 的感知路线分开，保留教师学生边界 |
| [wiki/overview/embodied-station-2026-10-28-59-papers-technology-map.md](../wiki/overview/embodied-station-2026-10-28-59-papers-technology-map.md) | [tapestry_arxiv_2610_36823.md](../sources/papers/tapestry_arxiv_2610_36823.md) | TAPESTRY 的视频训练先验与部署动作单步路径分开 |
| [wiki/methods/star-vla.md](../wiki/methods/star-vla.md) | [thaw_vla_arxiv_2609_24682.md](../sources/papers/thaw_vla_arxiv_2609_24682.md) | THAW-VLA 为训练期特征蒸馏，部署不调用教师；收敛 StarVLA 的普遍领先措辞 |
| [wiki/entities/paper-gift-intermediate-feature-training.md](../wiki/entities/paper-gift-intermediate-feature-training.md) | [thaw_vla_arxiv_2609_24682.md](../sources/papers/thaw_vla_arxiv_2609_24682.md) | THAW-VLA 为训练期特征蒸馏，部署不调用教师；收敛 StarVLA 的普遍领先措辞 |
| [wiki/concepts/progress-reward-modeling.md](../wiki/concepts/progress-reward-modeling.md) | [topreward_arxiv_2602_19313.md](../sources/papers/topreward_arxiv_2602_19313.md) | TOPReward 的 VLM 进度信号不等于成败保证；OXE 是数据集，AWR 是策略更新方法 |
| [wiki/concepts/open-x-embodiment.md](../wiki/concepts/open-x-embodiment.md) | [topreward_arxiv_2602_19313.md](../sources/papers/topreward_arxiv_2602_19313.md) | TOPReward 的 VLM 进度信号不等于成败保证；OXE 是数据集，AWR 是策略更新方法 |
| [wiki/methods/awr.md](../wiki/methods/awr.md) | [topreward_arxiv_2602_19313.md](../sources/papers/topreward_arxiv_2602_19313.md) | TOPReward 的 VLM 进度信号不等于成败保证；OXE 是数据集，AWR 是策略更新方法 |
| [wiki/entities/aloha.md](../wiki/entities/aloha.md) | [transformer_transformer_arxiv_2607_25798.md](../sources/papers/transformer_transformer_arxiv_2607_25798.md) | 纠正 ALOHA2 为论文所写 ALOHA，补原始作者与单次真机比较限制 |

## [2026-11-12] ingest | THAW-VLA | 论文 v3 复核与实体健康度补全

- 原因：同一论文已经入库，不创建重复实体；将全文通读状态、项目身份和源码核对补全至既有 THAW-VLA 节点。
- 核查：arXiv:2609.24682v3、作者项目页、GitHub README 与预计算 / 蒸馏 / 评测源码、Hugging Face 模型列表。
- 结论：Cosmos3-Nano 特征离线缓存，训练使用余弦对齐损失且部署去掉辅助头；权重可公开浏览，但 README 仍写 private。
- 产物：wiki/entities/paper-thaw-vla.md，及论文/项目页/源码三份 sources 归档；保留 Mermaid 主流程图并修订源码运行时序图。

## [2026-11-12] ingest | Cartwheel Control 三机型全身跟踪控制策略

- 通读官方模型卡、README 和核心仿真/策略运行源码，新增一手来源与知识实体；附 Mermaid 主流程及已核实的源码运行时序图。
- 明确三款机器人模型、50 Hz 控制 / 200 Hz 物理步进，以及无训练开源、无真机验证、无仿真随机化等边界。

## [2026-11-12] ingest | Cosmos 3 Interactive World Simulation 官方报告复核

- 复用已存在的 Cosmos Interactive World Simulation 唯一实体，不拆分四个 checkpoint 或重复建页。
- 重新核对 NVIDIA 2026-11-04 技术报告的时序因果注意力、四阶段后训练和原表关键指标：Bimanual 1× GB300 48.1 FPS；Sim-Auto 7 路 RGB + LiDAR 16× GB200 16.7 FPS。
- 新增 NVIDIA Cosmos Lab 官方项目页一手来源归档，修正旧报告日期和“双向 / 流式”“实时 / 30 FPS”“视觉生成 / 动作策略”的混淆；保留现有项目 Mermaid 流程图与源码时序图。
- 权重与框架已公开，但完整端到端闭环、传感器适配、模型安全评估仍需开发者自己完成。随后补充通读报告正文 §1–9、附录 A–D 的可读取文本与图表文字转写，并新增 Isaac Lab 策略闭环实验的正反例；唯一 wiki 节点补充指向原文归档的 `original` 字段，按三项结构健康规则核算为 3/3。PDF 高分辨率图片未独立目视审计，GPU 实验未复现。

## [2026-11-12] ingest | sources/papers/fast-motion-retargeting-pipeline-humanoid_arxiv_2610_28470.md — 精读 FRDV 并核对 2.0 开源实现

- 升级已有 FRDV 唯一论文实体，补全文中约束 SQP／RTI、可选 CI-iLQR 及两张 Mermaid 图，纠正过时的 pixi 运行命令与模块路径。
- 新增官方项目页和代码仓库来源归档；2.0 README 确认**公开的是运动学阶段，动力学验证阶段未开源**。
- 明确区分论文 1.0（9,156 FPS）与 2.0 Apple M1 热运行 8 workers（503,288.6 FPS）不同设置和测量口径；未宣称本地复现。

## [2026-11-12] ingest | CoRL 2026 Glen Berseth 阅读清单 101 篇：博客全文归档 + 概述页 + 逐篇详情节点（新建 / 升级到 3/3，2 篇 draft）

## [2026-11-12] ingest | sources/repos/mingrammer-diagrams.md — 收录 Diagrams（Python Diagram as Code）

- 意图：将 Diagrams 作为可版本管理的 Python 云系统架构可视化工具单独收录，并与 Mermaid.js 关联。
- 开源结论：官方 MIT 开源，源码及预置图标、CLI 均可公开获取；仅绘图，不控制实际云资源或生成 Terraform/CloudFormation。
- 证据：通读 README、pyproject.toml、Diagram/Node/Edge/Cluster 实现、CLI 入口与官方 Installation/Diagram/Cluster/Edge/Playground 文档。
- 知识沉淀：[Diagrams](../wiki/entities/mingrammer-diagrams.md)；附 Mermaid 流程及源代码时序，纠正“17 家云”与 Python 版本要求混淆。

## [2026-11-12] ingest | NVIDIA Holoscan SDK 5.0 — 实时多传感器与 Physical AI 运行时

- **意图：** 将 NVIDIA Holoscan 平台 SDK 与此前已收录的 Sensor Bridge 区分，新增带原文/源码证据的完整技术节点。
- **一手资料：** NVIDIA 官方 SDK 5.0 文档、2026-10-21 发布的 `v5.0.0` release、`slow_fast_fusion.cpp` 与 `camera_imu_control.cpp`。
- **开源结论：** SDK 本体与可运行 C++ / Python 示例已公开；HSB 是另一个独立接入项目；未在本地编译或做真实机器人时延实测。
- **关键页：** [Holoscan SDK](../wiki/entities/nvidia-holoscan-sdk.md)、[官方文档归档](../sources/sites/nvidia-holoscan-sdk-5.md)、[源码归档](../sources/repos/nvidia_holoscan_sdk.md)。

## [2026-11-12] ingest | sources/repos/nutlope-hallmark.md — 收录 Hallmark 前端设计 Agent Skill 与源码检查链

- 意图：将用户提供的 Nutlope/hallmark 原始仓库转换为可交叉引用的工具实体知识页。
- 核对：通读官方 README 与 SKILL.md，核查主题选择和 sloplint 的源码入口；官网确认安装命令 `npx skills add nutlope/hallmark -g`。
- 开源结论：MIT；技能文档、50 个主题目录和 Node.js 检查脚本公开；浏览器渲染检查另需 Chrome 与 puppeteer-core。
- 写回：新增 [Hallmark](../wiki/entities/hallmark.md)，关联 [MengTo/Skills](../wiki/entities/mengto-skills.md)。
- 验证边界：已核对源码与官方文档，未实际安装执行，也未运行完整知识库 CI。

## [2026-11-12] ingest | 4D-HOF 与三篇基础文献

- 意图：补足 4D 手物体交互中，视觉初始化、条件流匹配、物体 6D 位姿、手部 MANO 之间的技术联系。
- 主要结果：新增 4D-HOF 主知识实体（论文/官网/实现统一身份）与双语阅读、Mermaid 流程图和真实源码推理时序；归档 Albergo et al.、WiLoR、FoundationPose 三篇一手文献。
- 开源边界：TAMU-Visual-AI/4D-HOF 当前公开 HO3D 的推理/评估，不等于完整训练与原始视频初始化。
- 关联：复用已有 WiLoR 方法页，新增与 4D-HOF 的互链。
- 通读边界：已完成 4D-HOF v1 全文；本次补读 WiLoR CVPR 2025 正文和 supplementary，并结合官方 demo.py 补全来源、流程和实验边界。Stochastic Interpolants 与 FoundationPose 尚属初步核验，未标记已通读。

## [2026-11-12] ingest | Eureka ICLR 2024 官方论文与源码复核

- 在已有 [Eureka 论文实体](../wiki/entities/paper-eureka-llm-reward-design.md) 上增补基于源码的 Mermaid 运行时序图，没有新增重复项目节点。
- 复核 [arXiv 2310.12931v2](https://arxiv.org/html/2310.12931v2) 与 [官方项目页](https://eureka-research.github.io/)，新增 [官方仓库源码档案](../sources/repos/eureka-research-eureka.md) 和 [项目页归档](../sources/sites/eureka-research.md)。
- 源码快照 eureka-research/Eureka@9eee428；核实 MIT 许可、`eureka.py` → `train.py` → `rl_games` 的训练/反思路径；发现 README 和 config.yaml 的 sample/iteration 默认值不一致。
- 论文原文通读结果沿用 2026-11-08 的详细归档；本次源码检查只读与本流程相关的文件，未运行训练或视频。

## [2026-11-12] ingest | Runway Praxis-1 视频预训练到机器人控制

- 通读 Runway 官方技术报告，核对视频预训练、具身中训练、视频—动作联合训练、本体适配与 benchmark 的评测边界。
- 已核查 Runway 官方仓库公开权重、推理、MuJoCo 仿真、策略服务及微调入口；记录源码运行时序并补充 WAM 概念交叉引用。
- 新建独立项目节点，与历史同名 Praxis（第一视角全身操作）及 OAT 的 praxis-vla 明确区分。

## [2026-11-12] ingest | sources/papers/ego2robot_learning_survey_arxiv_2611_13610.md — 通读 Ego2Robot Learning 综述并建立独立实体

- 对 2026-11-10 的 arXiv:2611.13610v1 通读四范式、三类数据资源、五项开放挑战并绘制论文分类 Mermaid 图；区分库内 arXiv:2608.02580 同名合成方法。
- 项目页公开、论文 HTML 可读；GitHub 入口当前 404，未找到本综述自身可运行源码，因此源码时序图不适用。
- 关联既有 Ego2Robot 论文页避免孤儿节点；未修改派生 catalog / log / exports。

## [2026-11-12] ingest | sources/papers/rethinking_figures_arxiv_2611_08486.md — 通读 NTU 科研图示生成式 AI 评价研究

- 核对 arXiv:2611.08486v1 官方全文及 NTU AI4X 图集，收录 200 篇论文、371 张引导图/框架图的样本设计与保真评估。
- 记录 369 张修订图通过研究流程中的保真核查、清晰度模型评分均值 +0.428、美学协调评分均值 −0.111；明确单模型评价不能代表人类读者测试或 AI 使用的因果效应。
- 图集及 CSV 公开，但项目页未列出可核实的研究可运行源码仓库；不虚构源码时序图。
- 写回：wiki/entities/paper-rethinking-ai-scientific-figures.md，并从 AI Auto-Research 页面建立站内入链。

## [2026-11-11] ingest | sources/papers/self_refine_arxiv_2303_17651.md — 收录 Self-Refine 自反馈迭代改进

- 核对 arXiv v2 原文与附录，七任务评测及数学推理的自评瓶颈。
- 核实官网 Code 指向 madaan/self-refine，源码已开源；记录缩写生成与对话生成的真实入口以及论文算法和脚本停止条件的差异。
- 升格唯一论文实体节点并与 RSI 现有知识页互链，加入 Mermaid 主流程图和源码时序图。

## [2026-11-11] ingest | sources/papers/tcsexam_arxiv_2611_07962.md — TCSExam v1 论文、项目与评测源码

- 意图：归档理论计算机科学研究的四赛道可核验 benchmark，避免把指数误作准确率或机器人能力。
- 来源：arXiv v1 全文、Moonshot AI 项目/交互结果、MoonshotAI/TCSExam 官方源代码与运行手册。
- 开源：代码、任务与部分验证资源可用；部分独立 backend/固定镜像/来源材料需单独准备；许可按内容区分。
- 沉淀：[TCSExam 统一论文实体](../wiki/entities/paper-tcsexam.md)；关联 AI Agent 评测。
- 验证：静态阅读/接口核对；未执行模型或 Docker 复现。

## [2026-11-11] ingest | 具身智能小站 11-11（53 篇）与 43 个论文节点

## [2026-11-11] ingest | sources/sites/huawei-harmonyos-developer-doc-center.md — 收录华为 HarmonyOS 5.0+ 开发者文档中心

- 将官方文档入口、开发者知识 MCP、llms.txt 两级索引编译为单个双语实体节点；附 Mermaid 文档获取流程。
- 已通读文档中心主页、MCP 与 llms.txt 官方指南、一级索引；未扩大宣称至整个 API 文档库。
- 开源边界：网页与索引公开；知识 MCP 为远程服务，未找到可核验的官方服务端源码运行入口，源码时序图不适用。
- 与 Agentic Coding 工程实践建立双向交叉引用，强调按 API 版本检索、留存出处和运行验证。

## [2026-11-11] ingest | ReflexArc 论文全文精读与现有节点升级

- 在原有唯一实体上升级 ReflexArc（arXiv:2611.04424v1）的完整方法、训练、真机冲击实验与消融对照，避免同论文重复建页。
- 核对官方项目页当前代码待发布；不编造 ARC 源码运行时序图。
- 新增原文通读字段与主流程 / ARC-Gym Mermaid 图，区分 Fig. 6 的箱体冲量与 Table I 的 100 ms 冲量口径。
- 对应节点：wiki/entities/paper-reflexarc.md；原文：sources/papers/reflexarc_arxiv_2611_04424.md。

## [2026-11-11] ingest | zju-foundations-of-llms | 浙江大学《大模型基础》教材

- 官方仓库：https://github.com/ZJU-LLMs/Foundations-of-LLMs；中文/英文 PDF 与六章分章、Paper List 已定位。
- 新增 wiki/entities/zju-foundations-of-llms.md，并从 Transformer 概念页引入；来源归档拆分为书籍 PDF 与官方 README。
- 原文核对：官方 README 已通读；对照中文教材 PDF 的可检索全文镜像完成六章 §1.1–§6.5 的分节阅读，记录公式/图示/案例及版本差异边界。教材与当前官方 GitHub PDF 的二进制版本一致性仍未独立确认，英文版尚未逐节对照。
- 教材资料仓库不提供明确运行入口，Mermaid 绘制教材知识路径，源码时序图不适用。

## [2026-11-11] ingest | TacitVLA（arXiv:2610.17540）— 推理监督与动作时延解耦

- 意图：通读论文 v1 正文与 Appendix A–D，补充 VLA 的 attention-mask、流匹配动作专家和异步子任务账本知识。
- 一手来源：sources/papers/tacitvla_arxiv_2610_17540.md、sources/sites/tacitvla.md。
- 编译实体：wiki/entities/paper-tacitvla.md；从 wiki/methods/vla.md 建立入链。
- 官方开源状态：截至 2026-11-11 项目页仍标“Code soon”，无可核实运行入口；未编造源码时序。
- 证据边界：区分 RoboMME Counting-only 89.5% 与 16-task 训练 82.7%；最多 3.4× 为动作就绪延迟对比，不是全系统控制周期。

## [2026-11-11] ingest | sources/repos/code-review-skill.md — 模块化代码审查技能

- **目标：** 引入 Code Review Skill 的四阶段审查、语言指南按需读取、严重性等级和差异分析器机制。
- **一手资料：** 通读官方 SKILL.md，检查 README、官方 GitHub Pages、Python 分析器与其测试；源码与站点皆 MIT 开放。
- **成果：** 新增 [独立 wiki 节点](../wiki/entities/code-review-skill.md)、[原始仓库归档](../sources/repos/code-review-skill.md)、[项目站归档](../sources/sites/code-review-skill.md)，并与相邻工程技能节点互链；提供 Mermaid 项目流程图与基于 Python 真实调用的运行时序图。
- **限制：** 不把启发式脚本描述成完整漏洞扫描；未执行上游单元测试或真实 PR 的效果 benchmark。

## [2026-11-11] ingest | 激光雷达SLAM综述（新机器视觉转载）：综述归档 + 概念页 + 19 个算法独立节点

## [2026-11-11] ingest | PartyMimic 单卡分钟级全身运动跟踪

- 意图：将 RoboParty Lab 的 PartyMimic 一手项目资料、全套训练数据配方、统一评测与 SuperMimic 消融落地为独立知识节点。
- 开源：官方已发布训练/数据/ONNX/评测代码与 G1、RP1 checkpoint；原始数据及派生物受到源数据许可证约束。
- 写回：`sources/sites/partymimic.md`、`sources/repos/partymimic.md`、`wiki/entities/partymimic.md`；含 Mermaid 数据流及源码时序图。
- 边界：数据为官方结果，未进行单卡或真机复现；修正转载中 RP1 16 mm 的近似数与 `sparse5` 输入数目口径。

## [2026-11-11] ingest | A²-MEAS 论文已有深读节点：补核官方开源、实现时序和复现边界

- 目的：对照 arXiv:2611.09124v1、官方项目页和 `AutoLab-SAI-SJTU/A2MEAS` `evaluation` 分支，避免重复创建已有主节点。
- 开源结论：已公开 BEHAVIOR / RoboTwin 推理和评测实现、权重与一个 Adapter；完整纠错配置、部分任务特化策略及完整训练基础设施未在当前发布版开放。
- 关键区别：论文完整 BEHAVIOR Q-score 0.6411 / SR 45.40%；官方 README 发布版复现表 0.5287 / 32.40%，本次未执行 GPU benchmark。
- 产物：补充项目主页与源码两份来源归档，更新论文归档及唯一主节点，新增基于实际源码入口的 Mermaid 运行时序图。

## [2026-11-11] ingest | 生成动作与人形闭环关联综述

- 意图：去重补齐 Motion Graphs / MDM / A-MDM / AAMDM / Kimodo-Policy，并建立非 roadmap 对照节点。
- 开源核查：MDM、A-MDM 与 Kimodo-Policy 有官方可运行仓库；Motion Graphs、AAMDM 官方项目页本次未确认可运行代码。
- 阅读状态：后续已通读五篇论文正文及适用附录，复核实验表、代码入口和关键边界；本次七个修改 wiki 页健康度均为 3/3；未在本地复现训练或真机。

## [2026-11-11] ingest | SeeBFM：接入地形扫描后训练与双深度相机视觉感知路线

- **意图：** 以作者项目页、官方完整演示脚本/字幕和浏览器源代码为一手材料，将 SeeBFM 纳入 BFM 技术谱系；记录高度扫描条件化、地形无关 backward 提示、独立深度学生估计器、仿真穿越率与失败模式。
- **开源结论：** `capfab/seebfm-media` 公开的是项目网页、视频及预录仿真动作的 3D 回放，并非控制策略训练、推理或 G1 部署源码。
- **关键边界：** 作者主页标注 under review，尚未取得完整论文/附录；现有官方演示为仿真，G1 搭载深度学生的真实测试待完成。因此知识节点保持 `draft`、原文通读标为未完成，PR 保持 Draft。
- **关键页：** `wiki/entities/paper-seebfm.md`，`sources/papers/seebfm_preprint_2026.md`；从行为基础模型概念页增加交叉引用。

## [2026-11-11] ingest | sources/sites/qiayuan-liao-homepage.md — 通读 Qiayuan Liao 学术主页并升级研究索引

- **意图：** 复用既有 `wiki/entities/qiayuan-liao.md`，把六项研究、五个软件项目映射至已有实体而不重复建页。
- **核验：** 主页全文、Workhorse 与 SuperMimic 等官方页面，whole_body_tracking 与 mjlab README；发现 Walking in Narrow Spaces 署名/入口误链及 Berkeley Humanoid 源码 404。
- **开源结论：** whole_body_tracking 已归档；legged_control 主页标注 retired 但仓库未设置 archived；mjlab 公开维护；Workhorse 代码暂未发布。
- **关键页：** `wiki/entities/qiayuan-liao.md`；`sources/sites/qiayuan-liao-homepage.md`。补足完整阅读骨架、研究导航 Mermaid 图与单页原文追溯；人物索引不伪造跨项目源码时序图。

## [2026-11-11] ingest | sources/sites/corl-2026-program.md — 扩充 CoRL 2026 官方议程、Keynotes 与论文检索导航

- 对照大会 Program、首页、Keynotes、报告说明与 OpenReview 会议入口，增补场地、日期、报告类型、三场 keynote 和论文 ingest 追溯路径。
- 复用已存在的 CoRL Program 来源与 SuperMimic 关联，未新建重复会议项目或未核实的论文占位节点。
- Accepted Papers 动态正文与 Cvent 场次表未取得；直播公告不等于单篇论文可验证的录播 URL。此次仅修改 sources/ 和 log.d/，不触及 wiki 与派生文件。

## [2026-11-10] lint | 全量 lint 清零：29 个 wiki 实体补齐对比/评测/结论/重定向就绪度章节与评测基准、感知栈枢纽回链；flex-rsi 陈旧声明误报改写；gr00t-2 补 venue；缺页概念巡检豁免 openpi / start；review 22 个陈旧页面（补论文回链、修正 action-chunking 子步骤口径与 CRISP 同名页描述）

## [2026-11-11] ingest | Zane Hub #389《如何对机器人关节模组进行轻量化设计》：新增 robot-joint-module-lightweight-design 概念页，一手核对 Sandia 散热数字、国标与 CSG 对 CSF，标出未核实数字与槽满率自相矛盾

## [2026-11-11] ingest | sources/papers/dexroam_arxiv_2609_35761.md — DexRoam 全文精读并提升既有论文实体健康度

- 复核 arXiv 全文与附录、CoRL 2026 项目页、采集对齐与双 VLA 训练代码的实际入口。
- 不新建重复节点，升级既有 DexRoam 论文实体，补原文通读标记、Mermaid 主流程图、源码运行时序图与工程实践。
- 核实 29%→56% / 32%→57% 的五任务成功率及人机混合示教边界，公开 HF 50 条 episode 仅为单任务示例数据。

## [2026-11-11] ingest | CamoDreamer / arXiv:2607.17768v2 — 上下文解耦伪装图像生成

- 阅读作者 arXiv HTML v2 全文含附录，归档 CCB、CDAS、FACB、LAKE-RED 结果、下游 ZoomNet 验证及方法局限。
- 核对作者 GitHub 主分支仅 README 与展示图片，Oral 标记来自作者 README；未发布可运行代码，不虚构源码时序图。
- 新增论文唯一实体与主干 Mermaid 流程图，并从目标检测方法页建立入链；未修改由 bot 维护的 catalog.md/log.md。

## [2026-11-10] ingest | ResGAC（arXiv:2610.09479）全文复核与原节点升级

- **对象：** Precise SE(3) End-Effector Tracking in Whole-Body Humanoid Control / 全身人形控制中的精确 SE(3) 末端执行器跟踪。
- **去重：** 保留唯一实体 `wiki/entities/paper-resgac-precise-se3-end-effector-tracking.md`；论文、官网继续指向同一实体，不新建重复项目节点。
- **一手来源：** https://arxiv.org/html/2610.09479v1 ，已核读正文摘要、I–V（方法公式 1–30、实验 Tables I–IV 和结论）；https://github.com/ResGAC/ResGAC-website 中 README.md/index.html/details.html 已核查站点展示和补充参数。
- **修订：** 补通读标记与原文 original 字段；中英双标题、阅读骨架、GAC 无实测 wrench 的边界、控制公式、六阶段训练和真实实验条件、重画 Mermaid 算法流程图；修正被误标为原文直引的改写摘要。
- **开源状态：** 官网注明 Code 将在论文被接收后开放；当前只有项目展示网页源文件，无可核验的控制器训练/部署入口，源码 sequenceDiagram 暂不适用。
- **工程与评测边界：** 记录 G1 50 Hz、FastSAC、4096 环境、L40S、OptiTrack 外部定位/配准、插孔 18/20 与 H0 对不同轴不同程度改善。
- **派生文件：** 未修改 `catalog.md`、`log.md` 或前端导出文件，等待主分支自动生成。

## [2026-11-10] ingest | sources/papers/agenticgenplan_coding_agents_generalized_tamp_arxiv_2609_30233.md — 通读 AgenticGenPlan 原文，核对官方项目与 RoboCode 运行入口，建立英中双语 TAMP 泛化节点、Mermaid 主流程图和源码时序图；澄清“同一环境内不同实例”不等于跨模拟器迁移。

- 一手材料：arXiv:2609.30233v1 官方 HTML、agenticgentamp.github.io、tomsilver/robocode@d3d42dd、作者 Hugging Face 运行数据。
- 开源状态：研究代码和运行数据已公开；作者论文 held-out seed 仍需按需申请。
- 规模核对：28 环境、980 个程序、98,000 次评估；16 个 planner 环境子集为 47% 对 Astra 95%；动态 3D 仍有明显弱项。
- 防重复：论文、项目站点与官方源码归同一 wiki 项目实体；仅新增源文件与 log.d，不提交 catalog.md / log.md。

## [2026-11-10] ingest | PlanContext | 原文精读与导航意图成对评测基准

- **意图：** 收录 arXiv:2611.03067v1，建立独立论文实体并把六类驾驶情境、三要素成功判定、NAS/CPS/NCS/LAS/MCS 与 23 模型结论连到已有 OneVL。
- **来源：** sources/papers/plancontext_arxiv_2611_03067.md；sources/sites/plancontext-project.md。
- **原文：** 阅读 arXiv v1 正文与附录 A–E，来源归档记录了章节/表格/实验及局限。
- **开源：** 官方项目页截至 2026-11-10 标注 Code / Datasets Coming soon；不虚构代码入口或源码运行时序图。
- **落点：** wiki/entities/paper-plancontext.md；补 OneVL 交叉引用。避免修改 main 自动维护的 catalog.md / log.md。

## [2026-11-10] ingest | sources/repos/velxio.md — 核验 Velxio 开源嵌入式固件与 SPICE 协同仿真平台

- 意图：将真实固件编译、CPU 指令仿真、ngspice-WASM 数模联合求解的架构整理为独立知识节点。
- 一手资料：Velxio 官网与年付定价、官方 README / 架构文档 / 模拟器实现、作者在 Espressif Developer Portal 的技术文章。
- 事实校正：正确拼写 Velxio；官网 35 板/6 类与源码 README 40 板/5 CPU 家族统计不一致；Linux 不是 ISA。
- 开源结论：AGPL-3.0 核心开源，STM32 / 树莓派 Linux 等商业板卡后端及 AI Agent / 账号功能由私有 overlay 提供。
- 沉淀：新增 [Velxio](../wiki/entities/velxio.md)，并与 [Wokwi](../wiki/entities/wokwi.md) 双向互链。

## [2026-11-10] ingest | sources/papers/tf2_steering_tactile_forecast_arxiv_2611_01483.md — TF²-Steering 原文精读与唯一节点升级

- **意图：** 将 11-04 周报的索引级 TF²-Steering 升级为直接阅读 arXiv v1 正文、附录的可检索知识节点；复用原项目身份，不新增重复实体。
- **原文核验：** 阅读作者官方 HTML 全文，核对触觉预测/反馈双专家、Flow Matching 速度场修正、stop-gradient、异步 chunk 执行、三阶段训练和 Tables 1–7。
- **结果与口径：** 六项仿真成功率 68.3%（较 FTP-1 摘要报告 +10.7 pp）；六项真机 PSR 71.5%（较 T-Rex +9.1 pp，非二值成功率）。
- **开源结论（2026-11-10）：** 项目主页暂未提供可访问的 Code / 官方 GitHub 地址；已补方法 Mermaid 流程图，源码时序图不适用，不编造代码入口。
- **修改：** 同步更新 sources/papers、sources/sites 与 wiki/entities/paper-tf2-steering-tactile-forecast.md；不改 catalog.md / log.md 等派生文件。

## [2026-11-10] ingest | 具身智能小站 11-10（22 篇：A²-MEAS 深读 + TPS-A* / 被动折叠四旋翼 + 19 篇扫读）：新建 20 个独立论文详情节点（逐篇通读核对小站说法），复用 2 个既有节点，新增来源归档与阅读坐标页

## [2026-11-10] ingest | sources/papers/dgm_arxiv_2505_22954v3.md | DGM arXiv v3

- **意图：** 将 Darwin Gödel Machine（Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents）整理为 Robotics_Notebooks 可检索的独立中英双语知识实体，复用已有源码归档。
- **资料：** arXiv:2505.22954v3 正文与附录 A–J；Sakana AI 官方介绍；jennyzzt/dgm 官方 README、DGM_outer.py、self_improve_step.py 及源码入口。
- **主要结论：** 固定基础模型、探索可修改的编码 Agent 代码；用外部编程基准评估与开放式 archive 保留暂时退步的候选。SWE-bench Verified 20.0%→50.0%，完整 Polyglot 14.2%→30.7%；50 题搜索集的 38.0% 与全量复评须分开读。
- **开源结论：** 官方仓库 Apache-2.0，Docker / Python 主入口与示例代码已发布。未实测完整进化，需付费模型调用，并须隔离不可信代码。
- **写回：** wiki/entities/paper-darwin-godel-machine.md；sources/papers/dgm_arxiv_2505_22954v3.md；sources/sites/dgm-sakana-ai.md；更新 sources/repos/darwin-godel-machine.md 与 roadmap/depth-rsi.md。
- **图表：** 论文方法 flowchart、对照源码调用路径的 sequenceDiagram。
- **局限：** 论文目标劫持案例与高算力成本；未实证机器人真机或基础模型权重自我更新。

## [2026-11-10] ingest | Guava（arXiv:2606.18363v3）：机器人语义 Harness 与 4B 蒸馏

- **意图：** 区分大模型逐步语义工具调用和底层实时控制，补足带失败恢复的数据蒸馏路线。
- **一手核对：** 通读论文 v3 正文/附录，核对项目页、发布仓库 README、评测主链、数据采集与恢复分支、SFT/GRPO 入口。
- **技术核心：** GPT-5.4 → 共享工具 Harness + MuJoCo 轨迹/恢复分支 → Qwen3.5-4B SFT；GRPO 仅附加于选定长任务。
- **数据口径：** 2,268 条成功及恢复轨迹（1,554 主轨迹/714 分支）；仿真 15×30 次 87.1%，真机 9×10 次 90.0%（论文作者报告）。
- **源码与权重：** 官方仿真执行、轨迹、训练及评测脚本公开；SAM3 权重与真实机器人条件另需具备。仓库根目录未发现顶层 LICENSE，授权不可臆断。
- **新增实体：** [Guava：机器人操作 Harness 中的前沿 VLM 蒸馏](../wiki/entities/paper-guava-harness.md)（含原文 Mermaid 流程图与源码运行时序图）。
- **站内关联：** [LLM 机器人控制接口](../wiki/concepts/llm-robotics-control-interfaces.md) 入链，区别于同名 3D 渲染 GUAVA。
- **测试边界：** 本次经 GitHub 静态源码核对；未启动 MuJoCo/模型推理或进行真机复现，完整 CI 以 PR Actions 为准。

## [2026-11-10] ingest | LingBot-Touch | 异构触觉 JEPA 表征与官方推理实现

- 意图：为 Robbyant LingBot-Touch 建立唯一项目实体，关联 LingBot 家族。
- 已核验：官网 4,983h / 三模态、26 真机 + 12 仿真任务、GitHub 推理入口/输入契约、Tiny/Small 冻结 encoder 和 Apache-2.0。
- 开源边界：当前发行提供推理链、样例及权重，不能等同于完整预训练开源。
- 复核完成：通过临时 GitHub Actions 按 SHA-256 校验并逐页阅读官方 PDF **27 页**（正文 §1–6、Appendix A–C、References）；技术报告和评测表已进入 sources/papers 与主实体。
- 关键纠正：论文 Table 2 的 26 项真机结果基于 **4K 预训练 Tiny**，§4 frozen probing 是 **5K Small**，§5.4 单独对比 **1K–5K Tiny**；多源预训练在固定总时长下存在空间精度 trade-off。
- 节点健康度：新实体与 Robbyant 公司总览均补齐原文核验锚点，目标按报告脚本验证 **3/3**；未做本地 GPU/真机数值复现。
- 主节点：[LingBot-Touch](../wiki/entities/paper-lingbot-touch.md)；[论文记录](../sources/papers/lingbot-touch-technical-report.md)。

## [2026-11-10] ingest | sources/papers/horizonstream_arxiv_2605_23889.md — 归档 NeurIPS 2026 HorizonStream 全文与流式三维重建源码

- 以 arXiv v2 正文 §1–5、Appendix A–E、官方主页及 GitHub 推理实现为一手依据，补充 Geometric Linear Attention、Gated Window Attention、Metric Readout Tokens 的方法图及真实推理时序图。
- 同一项目合并论文、项目页与代码为唯一 wiki 实体；从视觉/3D 感知研究地图加回链。
- 实验区分 48 帧训练→万帧推理、无/有回环 ATE/FPS、训练重叠和总内存口径；训练代码暂未公开，许可证 CC BY-NC-SA 4.0。

## [2026-11-10] ingest | EmbodiedEvo 原文与源码深读复核

- **意图：** 不重复创建既有 EmbodiedEvo 节点；按仓库最新规范登记 arXiv v2 全文通读及附录 A–F，修正 Playground / Test 只读边界，核对实际代码入口与仿真调用。
- **一手来源：** https://arxiv.org/html/2610.18867v2 ; https://lab.einsia.ai/embodiedevo/ ; https://github.com/Einsia/EmbodiedEvo
- **关键补充：** 区分 full 2,100 与配对 210 episode；+41.3%（相对无经验）与 -53.0%（相对原始历史）的不同成本口径；跨任务与未成功 Playground 的正面与局限案例。
- **开源边界：** 源码可访问，但项目级许可证未选定；仿真资源/GPU/模型服务另配。未本地复现 RoboDojo-Sim。
- **落地节点：** [EmbodiedEvo](../wiki/entities/paper-embodied-evo-coding-agent.md)。

## [2026-11-10] ingest | EgoWild2Dex | 既有节点论文全文与附录精读升级

- **意图：** 避免与历史 PR #2296 的 EgoWild2Dex 同项目节点重复，沿用唯一论文实体，补充原始文献通读证据。
- **原文：** arXiv:2609.23755v1（§I–VI，Algorithm 1，附录 A–D，图表）与 https://mmlab.hk/egowild2dex/。
- **研究事实：** EgoWild 538.9 h、179,049 episodes；GeoFormer 五级 8 参数单应变换，学 robot→ego，使用逆变换对齐 ego；三阶段 FM VLA（task-specific ego、robot-view glove、expert robot + 10 条 DAgger）。
- **工程边界：** 96.7% 是三任务各 10 次共 29/30 次真机成功；图像对齐 21.9× 不是控制策略加速；每任务 robot <1 h 不包含人类数据与 A100 训练算力。
- **开放核查：** 2026-11-10 项目页无可下载的 EgoWild/GeoFormer 官方实现与模型资源，论文仅承诺待发布，源码时序图不适用。
- **回写：** [EgoWild2Dex](../wiki/entities/paper-egowild2dex.md)；[论文源归档](../sources/papers/egowild2dex_arxiv_2609_23755.md)；[项目页源归档](../sources/sites/mmlab-egowild2dex.md)。

## [2026-11-10] ingest | Flow-GRPO（arXiv:2505.05470）：流匹配模型在线 GRPO

- 通读 arXiv v5 正文与附录，归档 ODE→SDE、GRPO、训练/推理步数和实验边界。
- 核对官网与已公开的训练代码；新增论文实体，附 Mermaid 方法流程图和基于真实源码的训练时序图。
- 与 Diffusion Policy 双向关联；GenEval 63%→95% 和 OCR 59%→92% 为图像生成评测，不能误读为机器人真机结果。

## [2026-11-10] ingest | 收录 Steel、ToolDex、Agentic Robotics 与 ODEAction，复用 GR00T 2 和 AMP1

- 意图：核对六项原始论文/项目，统一收录一手资料、作者实验指标、Mermaid 流程与开源状态。
- 去重：GR00T 2 与 Apple AMP1 已存在完整项目实体，本次只补充相关节点交叉链接。
- 新入库：Steel/Newton 求解器、ToolDex 工具示范数据集、智能体机器人对照研究、ODEAction 连续动作演化。
- 源码核验：Steel 的 SolverSteel 在 Newton 主仓已公开；其余新研究的官方完整代码/数据开放范围已标注证据边界。
- 指标边界：Steel 最高 13× 是异构仿真吞吐峰值；ODEAction 61.8% 是 EBench 成功率相对提升。均未在本地重测。

## [2026-11-10] ingest | sources/papers/fah_arxiv_2611_07711.md — 收录 FAH 手部 2D/3D 对应点估计、网格拟合与接触/世界系扩展

- 原始资料：arXiv:2611.07711v1、官方项目主页和 noahcao/FAH 开源实现；分别归档至 sources/papers、sources/sites、sources/repos。
- 知识沉淀：wiki/entities/paper-fah.md；使用论文主干流程图及可执行推理路径的源码时序图。
- 开源结论：代码 Apache-2.0；FAH 模型权重研究/非商用许可；MANO/SAM 等外部资产另行授权。
- 指标边界：HOT3D PA-MPJPE 4.31 mm、65 FPS 含网格拟合，不包含部位/接触全功能；与世界系模式速度区别记录。

## [2026-11-10] ingest | Zane Hub #388《机器人动力学仿真和动力学标定有什么区别和联系》：新增 robot-dynamics-simulation-vs-calibration 概念页，自算质心偏差与差分噪声量级，标出未核的标准条款与笔误疑点

## [2026-11-09] ingest | AdaPolicy 原论文深读与已发布源码复核

- 意图：将二手摘要级 AdaPolicy 节点升级为论文一手证据驱动的工程笔记，不重复建实体。
- 来源：arXiv:2610.34280v1（含附录）、作者项目主页与 teee000/AdaPolicy 源码。
- 开源结论：MetaWorld / RoboTwin 2.0 仿真训练评测代码已发布；未核实完整 LIBERO 与真机部署源代码。
- 知识产物：补 next-chunk 梯度机制、实验/消融与训练成本、Mermaid 流程及已核验入口的源码运行时序图；改进 wiki/entities/paper-adapolicy.md。

## [2026-11-09] ingest | HWAM arXiv:2610.12026v2 增量深读与节点校核

- 已有 HWAM 主节点，未重复建页；论文 v2 全文 §1–5 与附录 A–D 已核对，并更新来源、三路径 Mermaid 主流程与原文通读状态。
- 明确 ALOHA Clothes Folding 的 86.00 为 0–100 分任务完成评分（含部分完成得分），不是成功率。
- 2026-11-09 直接核对官方项目页：Code / Dataset / Video 均为 “Coming soon”，与 arXiv v2 摘要中的 available 描述存在落差；无可运行官方 HWAM 源码入口，不伪造源码运行时序图。

## [2026-11-09] ingest | LLM 推理开源贡献职业分层与六项目一手文档核验

- 意图：按学生、1–2 年工程师和资深跨行工程师给出可验证的推理引擎 / kernel 开源贡献入口。
- 一手资料：Nano-vLLM、Mini-SGLang、FlashInfer、vLLM、SGLang、TensorRT-LLM 官方源码、README 与贡献指南，全部为公开仓库。
- 写回：一页分阶段 wiki 路线与六个互不重复的项目详情节点（新建 Nano-vLLM、Mini-SGLang、FlashInfer、SGLang、TensorRT-LLM；升级已有 vLLM 清单节点），项目均含流程图与基于代码入口的时序图，补充既有软件工程知识页入链。
- 证据边界：尚未定位用户所述视频原稿；“10 月 24 日 Discord 组队”和“更容易合并”的表述不作为已证实事实；官方性能数字不冒充本站实测。
- 节点健康度：本 PR 涉及的 8 个 Wiki 页按仓库健康度三项规则逐项检查均为 3/3；完整 CI 仍待检查，以最终 Actions 结果为准。

## [2026-11-09] ingest | 具身智能小站 11-09（29 篇：SLA 深读 + Flex-RSI / 美团 MR0 跟进 + 26 篇扫读）：新建 19 个独立论文详情节点（逐篇通读核对小站说法），复用 10 个既有节点，新增来源归档与阅读坐标页

## [2026-11-09] ingest | CamX / CamUVA | 跨本体机器人数据与 rollout 展示

- **意图：** 收录 https://camx-anonymous.github.io/ 的 CamX 数据集、CamUVA 评测回放与公开源码。
- **核验：** 区分许可裁剪后的 17.1K h 网站数据与未裁剪的 25.85K h 原论文附录统计；核对 LeRobot v3 单 episode 样本与视频同步工具。
- **开源结论：** 数据索引、样本/视频处理工具及回放网页可见；完整训练集、论文全文和 CamUVA 策略权重/训练代码尚未核实。
- **知识页：** [CamX / CamUVA](../wiki/entities/project-camx-camuva.md)（draft，待原文通读）。

## [2026-11-09] ingest | 三项评测基准

- 意图：按一手论文、官方项目页与可运行开源入口，新增 DeliverBench、SelfImprove-Bench、AmbiHTMLBench 三个唯一知识节点并建立到 Agent 评测概念页的双向链接。
- 开源核查：三项主仓均公开；DeliverBench 当前只有三项完整公开案例，SelfImprove-Bench 包含受限第三方依赖，AmbiHTMLBench 的主观原始数据需另行准备。
- 限制：论文全文/附录尚未完成逐节通读，保留 Draft PR 与每项 2/3 健康度；仓库 CI 和端到端复现实验待执行。

## [2026-11-09] ingest | MIIB：多模态信息交互瓶颈，通读论文与补充材料并核验开源状态

- 一手原文：arXiv:2611.07718v1（正文 I–VII 与 3 页 supplementary Appendix A–C）；作者在 arXiv 声明已获 IEEE TPAMI 接收，未核实正式刊出卷期。
- 核心方法：模态专属、联合融合、协同表征经双视图自监督与熵/对齐代理目标共同优化；新增论文方法 Mermaid 流程图。
- 开源状态：2026-11-09 核验 wanglingzaishi/MIIB 公开空仓，无可运行实现；源码运行时序图不适用。
- 知识节点：wiki/entities/paper-miib-multimodal-information-bottleneck.md，与 arXiv:2609.14421 信息论综述实体双向关联。

## [2026-11-09] ingest | arXiv:2610.33026v1 — 通读曼德拉效应多智能体虚假信念传播、持续与缓解论文；归档 540 条合成断言与三拓扑的实验设计，记录搜索污染主通道和定向纠错结果；源码及数据按作者声明尚待发布。

## [2026-11-09] ingest | senlanke 具身运控lab 11.02-11.06 前沿论文动态（人形/四足 30 + Manipulation 23）：新建 27 个独立论文详情节点（逐篇通读核对周报说法），复用 26 篇既有节点，新增两份周报来源归档与技术地图

## [2026-11-08] ingest | sources/papers/open-x-embodiment-arxiv-2310-08864-v9.md — 通读 OXE v9，升级既有 RT-X 节点

- 官方项目主页是 `robotics-transformer-x.github.io`，源码为 `google-deepmind/open_x_embodiment`；复用既有论文实体，避免重复。
- 区分 60 个数据集/22 种机器人本体/1M+ 轨迹与论文实验实际使用的 9 种机械臂混合数据。
- 基于论文 IV–V 和官方 RT-1-X 离线推理示例复核 Mermaid 主流程/源码时序；修正 61% 与 62% 的泛化评测口径。
- 开放范围：数据读取示例和 RT-1-X checkpoint / 推理示例；不把 RT-2-X 55B 全链路训练或真机部署写成开源。

## [2026-11-08] ingest | sources/repos/facefusion.md — 官方 FaceFusion 3.9.2 源码主链与技术文档收录

- 目标：新增 FaceFusion 独立实体；中英双标题、CLI/job/processor 主干与官方源码运行时序图，同时关联已有 Deep-Live-Cam 节点。
- 开源结论：公开可浏览、可安装的 Python 源码；根 LICENSE.md 只有 OpenRAIL-AS 标识，模型和商用权限需逐项确认；无未经验证的性能承诺。
- 主要核对：官方 README、Introduction、Installation、Processors 文档、headless/image/video 工作流、face_swapper/core.py 与软件许可证标识。
- 涉及关键页：wiki/entities/facefusion.md、sources/repos/facefusion.md、sources/sites/facefusion-official.md。

## [2026-11-08] ingest | EmbodiedRSI 假设引导的机器人持续学习

- **意图：** 收录哥伦比亚大学与普林斯顿大学 EmbodiedRSI 一手研究；澄清“机器人自己教自己”的模型/执行框架边界。
- **论文：** arXiv:2610.10498v1（公开 12 页全文已读）；整理 VOI 选择、Hypothesis Graph、Code-Skill 联合试验、PPO 记忆 Manager、仿真与 SO-101 真机结果。
- **可复现性：** 官方项目仓库只开放网站源码，未发现 EmbodiedRSI 可运行训练/部署实现；论文引用但未附的附录 A–I 另列公开材料缺口。
- **知识节点：** [EmbodiedRSI](../wiki/entities/paper-embodiedrsi.md)；从 [RoboDoctor](../wiki/entities/paper-robodoctor.md) 加入站内关联。
- **关键事实：** 冻结基础 VLA；真机零样本对象是仿真演化后的 harness；眼镜精密抓取任务出现回退。

## [2026-11-08] ingest | Vork Agent 原生 3D/VFX 渲染引擎 — 核对官方项目、源码、许可证与性能边界

- 意图：为知识库补充 agent-native GPU 渲染工具，并与 Sand3D / 3D-as-Code 和机器人仿真描述建立对照。
- 一手证据：voyager.so/vork、官方技术解读、voyager-so/vork 的 README、CLI/Shot 源码、Benchmark RESULTS 与 LICENSE。
- 核心判断：Vork 是 source-available、非开放源码许可证；L4 百万鸟 96.942 秒不计 setup/warmup，Blender 耗时为外推数据，不能宣传为 24 倍等画质提速；不是机器人动力学仿真器。
- 输出：新增 wiki/entities/vork.md 及 sites/repos 溯源归档、原文 Mermaid 流程图和基于真实 CLI 的运行时序图，链接至 3D-as-Code 选型页。

## [2026-11-08] ingest | Open Code Review 官方核验与既有节点升级

- **意图：** 收录 Alibaba 官方项目仓库、npm 包 `@alibaba-group/open-code-review` 与 `ocr` CLI，复核现有知识节点，不重复新建实体。
- **开源结论：** `alibaba/open-code-review` 公开核心 Go CLI 与 npm 包装，Apache-2.0；官方支持工作区/分支/commit 评审、`scan`、Delegation Mode、MCP 与 CI 集成。
- **一手通读：** 核对 README、官方快速开始与 CLI 文档，并审阅 `review_cmd.go`、`agent.go`、`selection.go`、`grouping.go` 关键调用链。
- **节点修订：** `wiki/entities/open-code-review.md` 增加 project_id、original 归档、英中双标题，校正分组机制并更新 Mermaid 主流程/源码时序图。
- **范围：** 只改知识源文件，不提交 `catalog.md`、`log.md` 或其他自动生成文件；未实际安装或运行 OCR。

## [2026-11-08] ingest | dreamscale-labs | 云端机器人基础模型推理平台

- **意图：** 收录 “I want to run a model on a robot to do something” 的官方产品与技术实现，区别机器人模型和云端推理服务。
- **一手来源：** Dreamscale 官网、Quick start / Models / Python SDK / SO-101 / YAM 文档，以及 2026-07-24、2026-10-16 两篇官方完整技术文章。
- **开放结论：** CLI/SDK 文档公开；HUD 评测适配代码 MIT 开放；商业平台 serving 后端未被官方仓库完整公开，不将其误标为全部开源。
- **工程关注：** 模型与 embodiment 观测/动作合同、动作块与控制频率、云推理和本地安全层分离；官方展示与独立复现口径分开。
- **输出：** Dreamscale Labs 主节点、网站归档、技术博客归档、HUD 源码审阅、频率解耦交叉引用；包含原文流程图和开源代码运行时序图。
- **验证边界：** 未启动收费推理会话、仿真或实体机器人；以 PR CI 检查页面健康度与 Mermaid 语法。

## [2026-11-08] ingest | Blueprint3D、Pascal Editor、MengTo/Skills 三个独立项目入库

- 为 2D 平面图转 3D 建筑、语义建筑编辑/MCP 与设计 Agent Skills 分别编译项目实体、官方源码归档及双语术语。
- 三页均依据官方 README 绘制 Mermaid 主干图、依据可查源码绘制运行时序图；源代码未在本环境运行。
- 关键复现边界：Blueprint3D 所查版本缺 checkpoint 及 LICENSE；Pascal 本地与托管 MCP 访问隔离；MengTo 的 176 技能是版本快照。

## [2026-11-08] ingest | Sand3D：十万级 3D-as-code 资产与可关节导出

- **意图：** 核验「106,094 件 3D 模型均可由代码修改、可商用」的说法；建立 Sandflow Sand3D 唯一技术实体。
- **一手来源：** Sandflow 在 Hugging Face 的 2026-11-05 完整技术发布稿、官方项目主页、官方 GitHub 源码与五份数据卡。
- **结果：** 归档数目、程序结构、F-score 与许可证，编写 Mermaid 资产处理流程图与基于 `washing_machine.py → sand3d API → assembly.export → urdf/usd writer` 的源码时序图。
- **重要边界：** 库 Apache-2.0；数据 CC BY 4.0 且原资产许可保留。URDF/USD 尚未提供完整动力学（质量、碰撞、驱动），不可等同于已验证的仿真就绪物理资产。
- **收录：** [Sand3D 节点](../wiki/entities/sand3d.md)；未更改 `catalog.md` 或 `log.md`。

## [2026-11-08] ingest | sources/papers/stabilizing_rl_with_llms_arxiv_2512_01374.md — 通读 Qwen MiniRL 论文并关联策略优化

- 通读 arXiv:2512.01374v3 全文 13 页（含附录），收录 MiniRL、IS 校正、clipping、MoE R2/R3 与数学推理实验。
- 纠正误述：本文讨论 LLM 本身的强化学习稳定性，不涉及 LLM 环境反馈校验或机器人奖励塑造。
- 官方复现状态：论文/arXiv 条目未列出可确认的作者官方训练代码入口；不虚构源码运行时序图。
- 关键页面：wiki/entities/paper-stabilizing-rl-with-llms.md；反向关联 wiki/methods/policy-optimization.md。

## [2026-11-08] ingest | arXiv:2601.08237 — 语言化奖励与多智能体协作展望

- 通读 arXiv v1 §1–8 和图 1–3 图注；归档 `sources/papers/end_of_reward_engineering_arxiv_2601_08237.md`，编译双语论文实体并补 Mermaid 机制流程。
- 区分离线 LLM 奖励生成与运行时语言协作；将 EUREKA、CARD、RLVR 既有证据与第 6 节尚未执行的实验设计分开。
- 本文未提供专用可运行源码，源码时序图标为不适用；从 EUREKA 节点添加可追溯互链。

## [2026-11-08] ingest | 深蓝具身君《机器人强化学习最近三年在做什么》：新增综述概念页，并为 Eureka / DrEureka / VLA 泛化实证 / SimpleVLA-RL / 物体中心残差 RL 各建论文节点，VLA-MBPO 节点升级到 3/3，逐篇核对文章说法

## [2026-11-08] ingest | DAWN 对比学习投影头与深度鲁棒性机制复核

- **意图：** 对现有 DAWN 项目条目补齐 Projection Head 的理论动机、正负样本构造及与随机潜状态的关系。
- **一手资料：** arXiv:2609.29092v1 全文、<https://dawn-parkour.github.io/>、<https://github.com/DocyNoah/dawn-parkour> 的世界模型和 runner 关键训练源码。
- **开源结论：** 官方已发布训练/仿真回放；未见独立预训练权重或真机通信部署入口。
- **关键更新：** 论文式 (9) 与实现的 BT×BT 双向 CE 负样本组织差异；对应方法流程图、源码训练时序图与消融数字。
- **知识页：** [DAWN](../wiki/entities/paper-dawn.md)。

## [2026-11-08] ingest | 具身小站《MIT模拟CSP的力控参数整定浅析》：新增 mit-mode-csp-emulation-tuning 方法页，公式自洽核对并标出与既有整定页的顺序差异

## [2026-11-08] ingest | Deep-Live-Cam — 单图实时换脸源码、官方版本与许可核验

- 意图：将 Deep-Live-Cam 作为视觉/数字人合成工具纳入单一 wiki 实体，附真实 Live 调用链与 Mermaid 图。
- 一手来源：GitHub README 和主分支摄像头、换脸、后处理源码；官网首页与官方 Releases。
- 开源结论：Python / ONNX Runtime 主分支公开、AGPL-3.0；官方 Ultimate 版本另有订阅制功能，模型授权不等于源码授权。
- 事实校正：2026-11-08 GitHub API 为 97,780 Stars；GitHub main 直接输出 Qt 预览，OBS 可窗口采集；官方 Release 宣布虚拟摄像头但不能混同开源 main 的已验证功能。
- 入库：`sources/repos/deep-live-cam.md`、`sources/sites/deep-live-cam-official.md`、`wiki/entities/deep-live-cam.md`，并从 ARGUS 鉴伪页面建立反向关联。

## [2026-11-08] ingest | FlexGlue 现有实体补充 COLMAP ONNX 官方集成

- 去重：沿用已合并 PR #3418 建立的 `paper-flexglue`，不新增项目节点。
- 核对一手资料：arXiv:2610.32976v1、cvg/FlexGlue 推理源码、COLMAP 官方 Features 文档、`src/colmap/feature/flexglue.{h,cc}`。
- 补充 COLMAP ONNX 描述器与匹配器的 C++ 源码运行时序图、构建条件、关键参数与复现边界。
- 修正 PyTorch 默认关闭与 COLMAP 默认开启共视过滤的差异；COLMAP 官方文档的 35% 早停实验仅适用于指定 ETH3D 配置。
- 原论文全文及定性图尚未完成逐段逐图核查；来源标为未完成，PR 为草稿；未在当前环境运行 `make ci-preflight` 或执行模型。

## [2026-11-08] ingest | Zane Hub #386《人形机器人的机械臂有哪几种构型》：新增 humanoid-arm-configuration 概念页，核对宇树 G1 / 傅利叶 GR-2 官方规格

## [2026-11-08] ingest | 视觉三维一手研究资源归档

- **目的：** 用户给出十张课程目录，提取真正的原始论文、官方项目、源码和技术文档，明确排除 3D视觉工坊自身课程。
- **去重：** 原始链接 90 条；按 63 个主题/项目组归并；已存在核心节点（vins-fusion, orb-slam3, lio-sam, 3dgs）不另建重复 sources 档案。
- **本次变更：** 原始清单 + 新增 36 份源码归档、31 份论文归档和 12 份文档/项目归档。
- **原文状态：** 仅完成 URL 来源身份和部分 README 核对，大量论文全文/实验/局限/代码入口未完成；不标记完成或伪造 Mermaid。
- **PR：** https://github.com/ImChong/Robotics_Notebooks/pull/3497（草稿）

## [2026-11-08] ingest | 深读 NVIDIA cuRoboV2-Fast 论文与源码并修正性能对比口径

- 复用既有 arXiv:2611.06913 节点，不重复建项目；论文、官网、官方代码三层一手归档复核。
- 核验 RTX 5090/3kg 全链路规划 4.43 ms、14.7×，Jetson 25.99 ms，YAM 真机周期 5.93 s/缩短 34.6%；明确规划时延不等于真实周期。
- 补 B-spline、GPU BiRRT、TimeOpt 原理、项目 Mermaid 流程图和基于官方 Python/TimeOpt 入口的 Mermaid 时序图。
- 官方 NVlabs/curobo 已开源（Apache-2.0）；仅静态读码，未本地复现性能。

## [2026-11-08] ingest | 视觉感知与三维重建一手研究入口归档

- 去重保存 90 条外部一手链接，归入 63 个项目/方法身份组；未新增重复 wiki 实体。
- 新增 59 份来源初编，复用 4 份既有来源记录；补一张跨主题研究地图。
- 资料仍需逐篇通读，项目实体 Mermaid 流程及源码真实时序尚未全部完成，PR 保持 draft。
- 注意 OED 分类纠错、FeatDepth 作者代码身份、GaussianMesh 中文题名对应待核。

- [总表](../sources/sites/vision3d_primary_sources_20261108.md)
- [研究地图](../wiki/overview/vision-3d-perception-reconstruction.md)

## [2026-11-07] structural | roadmap/depth-*.md — 30 条纵深路线精简标题为「中文（English）」（去掉「路线（纵深）：如果目标是」前缀与流程链），长摘要改为首屏导读三要点（为谁 / 路径 / 纵深路线）

## [2026-11-07] ingest | OpenRewrite 官方源码与文档：LST / Recipe / Visitor / 调度器

- 意图：将自动化源码重构框架 OpenRewrite 收录到机器人软件工程知识图，提供可核对的源代码执行路径。
- 一手资料：openrewrite/rewrite 官方 README 与关键源码、docs.openrewrite.org 原始技术文档；源码快照 08560af3c93b7a87d1cd798a97c1b212e303b16e。
- 结论：核心 Apache-2.0 与部分 Source Available/商业配方应区分；RecipeScheduler 按 scan / generate / edit 周期处理源码，Wiki 附主 Mermaid flowchart 与非 RPC 源码 sequenceDiagram。
- 关键页：[OpenRewrite](../wiki/entities/openrewrite.md)；未运行重构示例或整库验证。

## [2026-11-07] ingest | sources/papers/robodoctor_arxiv_2610_35604.md — RoboDoctor 原文通读、保留抓握的局部修正与重复节点合并

- 意图：根据 <https://robodoctor-rsi.github.io/> 对照论文 Figure 4(b) 的瓶口/杯口偏移，一手通读后更新 RoboDoctor 的**单一论文/项目实体**，补 Mermaid 项目流程与已核实代码时序。
- 原文：<https://arxiv.org/html/2610.35604v1>，2026-10-26 v1；代码 <https://github.com/THUSI-Lab/RoboDoctor>（核对 main `e3ba0ea`）。
- 实质：保留稳定抓握，Doctor 横向一步修正，WAM 后续倾倒；局部对齐与任务成功分别核验。RoboDojo-Sim 54 项，Warmup 49.74→Refinement 57.10；但跨回合经验增益未经受控隔离。
- 开源：正式 Python 运行时、入口和测试已公开；模型权重与 RoboDojo 仿真环境独立获取。删除早期重复 wiki 节点并登记旧页面别名；未复制网页/论文原文，未触及 catalog.md/log.md。

## [2026-11-07] ingest | sources/papers/behavior_spatial_arxiv_2610_34051.md — BEHAVIOR-Spatial 三层空间诊断基准入库

- 通读 arXiv:2610.34051v2 原文及附录；按 Understand（VQA）/ Ground（IR）/ Act（Action）编译单一 Wiki 实体。
- 核实官方评测仓库、HF 数据和运行协议：源码包含 VQA/IR 打分、OmniGibson rollout harness，不含论文 VLA 的训练源码与权重。
- 完成官方评测代码时序图与论文项目流程图；从 Manipulation 任务页建立入链，避免孤儿节点。
- 主要诊断：held-out frozen-VLM soft 41.6%，prompt-probe soft 16.1%，wrong-target 70.0%；跨切片指标不可直接解释为因果降幅。

## [2026-11-07] ingest | Recova（arXiv:2610.01178）论文与项目主页入库

- 通读论文 v1 的方法、实验、附录 A–E，并对照作者项目页与 NVIDIA 项目镜像。
- 新增 `sources/papers/recova_arxiv_2610_01178.md`、`sources/sites/recova.md` 与 `wiki/entities/paper-recova.md`。
- 方法图包含数字孪生、双策略、VLM 失败监控与 DAgger 人工接管分流；真机四任务均值 23.8% → 77.5% → 87.5%。
- 官方网页展示恢复代码片段，但未见公开的完整训练/部署仓库，故不伪造源码运行时序图。
- 在 `wiki/tasks/manipulation.md` 增加相关研究入口。

## [2026-11-07] structural | media/site-demo.gif — 按最新图谱重录 README 演示 GIF（7400 节点），修复节点过小时悬停 / 点击两步被跳过

- **触发：** 用户要求更新首页 GIF
- **脚本：** [`scripts/record_readme_demo.cjs`](scripts/record_readme_demo.cjs)（88 frames / 1.45 MB；图谱 **7400** 节点 / **61087** 边）
- **修复：** 选悬停 / 点击节点不再要求直径 ≥ 7 / 8px（7400 节点适配全屏后最大约 5px，两步原本被静默跳过）；只悬停视口偏上的节点，浮窗不被顶部工具栏遮挡
- **流程：** `make export graph` → `docs/` 本地 `http.server 8765` → 重录并写回 `media/site-demo.gif`
- **引用：** [`README.md`](README.md)「在线演示」仍指向 `media/site-demo.gif`

## [2026-11-07] ingest | OpenRoboKit 三套工具

- 意图：区分 RoboKit 并行可微优化、RetargetingKit 人体动作映射与 GraspKit 抓取学习，并建立交叉引用、原文归档、源码运行时序。
- 代码开放：三个官方 GitHub 仓库均已公开，具有实际可运行入口；不等同于本地成功复现或第三方整机适配认证。
- 关键页：wiki/overview/openrobokit-toolkit.md；wiki/entities/robokit-openrobokit.md；wiki/entities/retargetingkit.md；wiki/entities/graspkit.md。
- 阻塞：OpenReview 论文全文受验证阻碍，官网正文不可直接读取；相关页面保持 draft。

## [2026-11-07] ingest | Meta Astryx — 开源 React 设计系统与 Agent CLI

- 意图：将 Meta Astryx 收录为独立工程工具节点，区别于机器人运动控制/强化学习框架。
- 原文：阅读官方根 README、Core/CLI 说明及 Theme、Button、CLI init 核心源码，并以提交 SHA 固定技术依据。
- 开源状态：组件、主题与 CLI 公开；MIT；官网与 README 组件数量口径不同，按来源分述。
- 写回：wiki/entities/astryx.md，关联 react-bits.md；归档 sources/repos/astryx.md 与 sources/sites/astryx.md。
- 图表：补真实架构数据流和 CLI init 源码时序图；无运行时测试断言。

## [2026-11-07] ingest | sources/papers/jitrl_arxiv_2601_18510.md — JitRL（arXiv:2601.18510）免梯度测试时经验检索 + logit 修正；新增 wiki/entities/paper-jitrl-just-in-time-rl.md，官方代码归档 sources/repos/jitrl.md；附 Table 2/13 骨架差异说明；Bilibili 解读未能获取

## [2026-11-07] ingest | Mechtool 与关联页健康度补读：螺纹文章全文含图、公差仿真 / 轴承 / 膝谐波 / 模态四篇 Zane Hub 文章重读，Mechtool 站点部分读完

## [2026-11-07] ingest | Mechtool 在线机械设计工具站 + 紧固件全接触《常用螺纹基础知识》：新增 mechtool 实体与 screw-thread-basics 概念页，复核公式与表值

## [2026-11-07] ingest | 健康度补强：physical-rsi 两页重读原文并更新（项目页 2026-10-08 口径、代码已开源）、研究地图补阅读骨架；pylabrobot 文档站补读 machine-agnostic 部分，仍未读完

## [2026-11-07] ingest | PyLabRobot（文档/仓库/Device 论文/机器清单/可视化器）与 Google 自主湿实验三篇论文及博客：各建独立节点并互链；PyLabRobot 通读未完成（Device 正式版 403、文档站大部分页面未读），保持 draft

## [2026-11-07] ingest | Scenario skills 与 Scenario for Blender：按实读内容校正结构数字、验证声明与机器人关联，通读范围如实标注（均未完成）

## [2026-11-07] ingest | scenario-labs/skills（Scenario Agent Skills）：新增 scenario-labs-skills 实体与来源归档

## [2026-11-07] ingest | Anthropic《The components of agent memory》与 Claude Cookbook《Dreaming with a dynamic workflow》：新建 agent 记忆与 dreaming 概念页和 cookbook 节点，核对 sleep-time compute 引用与仓库（MIT），读完页面与 notebook 全部说明/关键代码，未运行代码

## [2026-11-07] ingest | 通读 Hi Robot 原文与附录，复用既有节点，纠正高层回应流向和指标口径，补原文溯源与开放状态核查。

## [2026-11-07] ingest | 通读 SmellDiffusion v2，升级既有节点，核对源修正、规划与语言评测及代码访问边界

## [2026-11-07] ingest | 通读触觉多模态融合综述 2605.17336v1，整理数据、融合、评测边界；核实官方仓库为文献索引，无自有运行实现。

## [2026-11-07] ingest | sources/papers/tampagent_arxiv_2611_03584.md — 通读正文与附录 A–H，升级既有 TAMPAgent 节点，补混合信念流程、评测边界与 Code soon 状态

## [2026-11-07] ingest | 完成多模态信息论综述 v1 正文核对，补全原始归档与 GMIL 实体、流程图及适用边界；未发现本文官方可运行实现。

## [2026-11-07] ingest | 核对 XMimic 论文全文与 JAX 多卡训练源码，区分同名项目并澄清 54 秒、3.5 倍和未证实的 85 倍口径

## [2026-11-07] ingest | HachimoDock — 核对软件源码、硬件物料与150元参考成本，补充流程和固件时序，记录许可冲突与验证边界

## [2026-11-07] ingest | 通读 VLA-Tug 与 Latent RL Priors，补充故障预警和冻结控制先验节点、论文流程及 VLA-Tug 源码时序；核实仿真数据已开放与真机/待发布边界。

## [2026-11-07] ingest | Zane Hub 第385篇《Isaac Sim 和 MuJoCo 能取代 Adams 吗？》：新建三类物理问题选型对比页，一手核对 MuJoCo 3.17.0、Newton 1.0、Isaac Sim 许可/MCP、Cadence 收购（Isaac Sim 6.0 日期来源不一致且最新已 6.2）

## [2026-11-07] ingest | Ctrl-F：通读论文与项目页，归档三维力接口、反向力训练及G1增载实验；官方代码待发布

## [2026-11-06] lint | 全量 lint：清零 10 条信息型预警（execute/TensorRT 停用词、SCvx 选型落地、两篇论文补对比段、ITW-Ego Plus 数据速查、4 处陈旧措辞加日期限定）

## [2026-11-07] ingest | RoboScope Kit：升级既有实体，通读论文与附录、核对官方源码，补双图及25.04倍与同硬件6.88倍口径

## [2026-11-07] ingest | sources/videos/stanford-hai-discovery-loop-2026.md — 通读官方完整字幕，归档闭幕对谈与议程，区分科研闭环目标与已验证结果；未列运行代码，补流程图并关联科研自动化。

## [2026-11-06] ingest | wiki/entities/gamma-robotics.md — 公司路线新增伽马智能（Gamma Robotics）：TeleAI 孵化，6 个节点（NEXUS / TTCO / GR-BC / KODA / 成立与 GBot-W·GData / LocoManiLab）；同步首页、对照矩阵、开放程度与来源索引

## [2026-11-06] ingest | 复核 KODA 论文全文与官方源码，更新既有节点；厘清 Robot-OOD、人类示范覆盖和部分开源边界

## [2026-11-06] ingest | 通读 π₀ 官方全文，补官方 OpenPI 服务时序，纠正流匹配目标符号与评测口径

## [2026-11-06] ingest | sources/sites/itw-ego-plus.md — 收录真实环境第一视角与全身运动数据，许可和数据细节待核验

## [2026-11-06] ingest | roadmap/depth-rocket-landing.md — 新增火箭回收控制纵深路线（Stage 0–6）；补入 Blackmore 2010、Szmuk 2018×2、Malyuta 教程、Gaudet RL、Klumpp Apollo、NASA CR-820 等一手资料与 3 个 wiki 页（动力下降制导 / 无损凸化 / 序列凸化）；Klumpp 正文与 Açıkmeşe&Ploen 2007 原文未取得

## [2026-11-06] lint | 清零 16 条信息型预警：7 个评测页补评测基准枢纽回链；MAPO / MiTA / VFlash 补对比区块、VFlash 补 venue、对比学习人形页拆出实验与评测；新建 concepts/supervised-fine-tuning.md；lint 豁免 env（Gymnasium 变量）/ v1（arXiv 版本号）与运行时「最新画面」并补测试

## [2026-11-06] lint | 修复 6 个陈旧页面：physical-rsi-research-map 补 Compounding Physical RSI 与配套论文 arXiv:2610.18842；variational-automation / paper-gap-graph-as-policy 补 VABAR 基准；paper-agentic-real2sim 补 HomeBody 交叉链接；paper-gpt-policy / AMO 核对 HomeBody 开源状态更新后无需改事实

## [2026-11-06] structural | roadmap/depth-pcb-design.md — 新增 PCB 设计纵深路线（Stage 0 电路零基础 → Stage 7 EMC/可靠性/量产），注册到路线总览、README、index、首页按钮、depth-filters 与 hardware 主题

## [2026-11-06] ingest | 深蓝学院《在宇树 G1 上实战 BeyondMimic，31 个动作一个策略》：新建视频到 G1 舞蹈的 GVHMR→GMR→BeyondMimic 实操 Query，四个官方仓库静态核对（-s 是 static_cam、GMR 写 pickle、BeyondMimic 官方库已退役、教程混用两套 csv_to_npz）

## [2026-11-06] structural | scripts/original_reading.py — 原文已读完改为结构化判据：wiki frontmatter original + sources 归档「通读：已完成|未完成」，lint 阻塞校验；回填 32 份归档与 25 个 wiki 页

## [2026-11-06] structural | wiki/ — 统一 frontmatter status 为 stub / draft / complete（200 页：planned→stub，drafting/in-progress/preview→draft，stable/active→complete），lint 新增取值门禁

## [2026-11-06] structural | scripts/generate_link_graph.py — 图谱健康度改用可自动检查的 ingest 信号：元数据 + 阅读骨架 + sources 核对范围记录（原文已读完）

## [2026-11-06] structural | scripts/generate_link_graph.py — 图谱健康度按 ingest 完成度封顶（stub/planned ≤1，draft 类 ≤2），图例同步

## [2026-11-06] ingest | 新机器视觉《张正友标定、九点标定、手眼标定到底怎么选？》：识别为大凡光学原文的浓缩转载（经3D视觉工坊），新建转载归档并把精度链/布点验证/复校要点补进相机标定三层概念页，不新建重复节点

## [2026-11-06] ingest | 具身智能小站 2026-11-06《只在虚拟房间练习，人形机器人走进陌生房间能做什么？》20 篇逐篇建独立节点（新建 16 / 复用 4），Cosmos 交互世界仿真技术报告补 arXiv 2611.06627，公众号称 Open-HumanScale 的论文 arXiv 题名为 Open-HumanWAM

## [2026-11-06] ingest | sources/sites/liber-0.md — 核对 Liber-0 官方博客与实验口径

## [2026-11-06] ingest | sources/repos/stanford-tml-homebody.md — 更新 HomeBody 代码开放状态与仿真运行入口

## [2026-11-06] ingest | paper-dynasolver — 补充论文实验、局限与项目流程图

## [2026-11-05] ingest | sources/papers/vflash.md — 收录 VFlash 视觉图元解码与实验，核查代码链接

## [2026-11-06] ingest | 七项机器人资料 ingest：HALO、VABAR、Compounding Physical RSI、A³-VLA、Soccerbot 五篇论文全文通读并建独立节点；Cosmos 3 博文补全既有节点（更正三类/仅AgiBot说法）；智元 GO-3 新闻稿三榜核对（仅 RoboTwin 完整成立）

## [2026-11-05] ingest | 补全 Proactive AI for Superintelligent Agents（203 页综述）：读摘要/概念框架/六项能力/具身章节/跨行业趋势/治理，核对官方仓库（综述配套资源，无智能体实现，未运行其脚本），建唯一 wiki 实体、项目页与仓库归档，含流程图与 papers.py 静态时序图

## [2026-11-02] ingest | sources/papers/scaffoldforge.md — 记录论文线索与全文核验阻塞

- 全文当前无法获取，论文实体和流程图待通读后补充；官方代码状态尚未核实。

## [2026-11-05] ingest | sources/papers/raven_active_slam_arxiv_2610_37111.md — 收录无人机主动 SLAM，源码待发布，原文复核继续

## [2026-11-05] ingest | sources/papers/global_style_transfer_arxiv_2608_11546.md — 收录 GST 风格迁移与评测，作者页面未公开实现

## [2026-11-05] ingest | sources/papers/yolo27-alphaxiv-overview.md — 收录 YOLO27 预发布综述，记录官方预览差异与待核查项

## [2026-11-05] ingest | sources/papers/gradsplat_2610_38951.md — 收录梯度细化三维高斯重建，公开推理与权重

## [2026-11-05] ingest | sources/repos/pixelfx.md — 收录像素特效库、技能及渲染流程

## [2026-11-05] ingest | 具身智能小站 2026-11-05《人能稳稳提起的罐子，换一双机器手怎么提？》43 篇逐篇建独立详情节点（新建 28 / 复用 15），arXiv 标题日期逐篇核对，开源状态逐项核查

## [2026-11-05] structural | docs/main.js, docs/style.css — 路线页（主路线 + 纵深路线）知识地图改为学习路线图样式（左侧阶段栏 + 单阶段小树 + 前置 / 通往标签，全局总览改为各阶段小树自上而下排开），参照 Robot_Learning_Paper_Notebooks 首页推荐学习路线图

## [2026-11-05] ingest | sources/blogs/alex_where_physical_ai_is_heading_2026-11-05.md — 整理 Physical AI 仿真、形态与部署观点

## [2026-11-05] ingest | Zane Hub 第383篇《现在人形关节模组电机的驱控算法框架是什么？》：新建人形关节驱控三层框架概念页，核对 MIT Cheetah / Berkeley Humanoid 数字（电流环 400 Hz 无依据），回链阻抗控制页

## [2026-11-05] ingest | sources/repos/lerobot.md — 固定版本核对 LeRobot README 模型清单，模型独立详情与原文源码核查待完成

## [2026-11-05] ingest | LeRobot README 32 个模型逐个核对一手论文与固定提交源码：新建 8 个节点（VQ-BeT/Multitask DiT/TDMPC/GR00T N1.7/XVLA/EO-1/SARM/Robometer 等），其余补流程图与源码运行时序图并纠正旧页差异

## [2026-11-05] ingest | sources/repos/tencent-hy4-preview.md — 整理 Hy4 预览版架构、微调源码与正式版待核验边界

## [2026-11-05] ingest | 具身智能研究室《这三年，机器人训练越来越快，数据重定向能不能借通用运控再往前走一步？》：16 个方法逐个核对均已有独立节点，补全 7 个并新建概览页，更正 GMR 论文归属与 FRDV 代码状态

## [2026-11-05] ingest | sources/repos/mika.md — 收录 Android 手机智能体及其源码调用路径

## [2026-11-05] ingest | sources/repos/frax.md — 收录开源 JAX 运动学动力学库及人形重定向实例

## [2026-11-05] ingest | sources/repos/tencent-octop.md — 收录自托管多用户多智能体平台及源码启动流程

## [2026-11-04] ingest | 新机器视觉《2026年了，主流具身智能算法有哪些？》：20 个算法逐个回一手资料核对（新建 3 / 复用 15 / 无法定位 2），指出 Octo、OpenVLA 数字与 README 不一致

## [2026-11-04] ingest | 大凡光学《一文看懂三种标定原理：张正友、九点、手眼标定全解析》：新建相机标定三层概念页，核对 Zhang 2000 一手摘要，回链手眼标定页（无论文/项目清单）

## [2026-11-05] ingest | sources/papers/tacclaw_infinity_arxiv_2610_20320.md — 补充论文全文、实验结果与源码调用，确认部分开放

## [2026-11-04] ingest | 深蓝AI《彻底搞懂RRT与RRT*》：新建 RRT / RRT* / RRT-Connect / Informed RRT* 四个方法节点（算法教程，无项目清单）

## [2026-11-04] ingest | 具身智能小站 2026-11-04《机器人做错了，奖励模型为什么还在加分？》57 篇逐篇建独立详情节点（新建 54 / 复用 3，保留库内已有 6 篇来源笔记）

## [2026-11-04] ingest | sources/papers/hug_arxiv_2611_01659.md — 收录 HuG 人形全身抓取；项目页未能访问，开放状态待核查

## [2026-11-04] ingest | sources/repos/agent-autoloop.md — 收录命令行智能体事件调度与源码运行路径，协议复核待完成

## [2026-11-04] ingest | 收录 MAPO 和 VLALagBench 论文草稿，全文及开源状态待核查

## [2026-11-04] ingest | sources/papers/visualforce_arxiv_2610_04741.md — 收录视觉预测力与开源实现，区分论文和代码反馈路径

## [2026-11-04] ingest | sources/repos/vlx-flow.md — 收录流式视频理解与双层记忆，模型实现和权重待发布

## [2026-11-04] ingest | sources/papers/nablamimic-2610-41654.md — 补充论文机制、实验边界与训练源码调用链

## [2026-11-03] ingest | Qt历险记《C/C++ 运动控制开源库分类整理》：14 个项目逐个回一手资料核对（新建 7 / 复用 7），标出许可与实时性等与文章不一致处

## [2026-11-03] ingest | FAITH 可行性感知安全过滤 RL

- 通读 arXiv v1，归档论文与方法，区分仿真量化结果和真机定性演示。
- 未发现可运行官方代码；记录理论假设及安全保证边界。

## [2026-11-03] ingest | sources/papers/diffdreamer_arxiv_2610_31617.md — 收录世界模型与可微仿真耦合四旋翼控制，官方实现待核实

## [2026-11-03] ingest | sources/papers/policy_coverage_reliability_arxiv_2611_01565.md — 收录策略覆盖度与可靠性；代码链接未开放，附录待通读

## [2026-11-03] ingest | sources/papers/vista_arxiv_2610_35051.md — 收录主动探索与 harness 进化，记录项目页访问限制

## [2026-11-03] ingest | sources/papers/uni_robot_agent_arxiv_2611_01560.md — 收录开放模型机器人智能体，原文及项目页面待完成核查

## [2026-11-03] ingest | 具身智能小站 2026-11-03《机器人不仅要完成任务，还要知道哪里出了错》49 篇逐篇建独立详情节点（新建 44 / 复用 5）

## [2026-11-03] ingest | sources/repos/sts32xx-firmware.md — 收录伺服固件控制流程、源码路径与演示边界

## [2026-11-03] ingest | Zane Hub《目前灵巧手行业的核心企业，它们的优势体现在哪些方面？》：新建综述概念页 + 8 个企业/产品独立节点，复用 4 个（企业数据为作者表述未核实）

## [2026-11-03] ingest | NablaEngine 与 Movens-Zero 一手资料归档；全文及源码分析待完成

## [2026-11-03] ingest | sources/papers/s4vy_arxiv_2609_36875.md — 收录 S4VY 4D 实例分割及官方实现

## [2026-11-03] ingest | sources/papers/gpt_6_astra_embodied_intelligence_arxiv_2610_22471.md — 更新报告与源码开放状态，补充评测协议及真机结果草稿

## [2026-11-02] ingest | sources/papers/care_arxiv_2602_03473.md — 收录 CaRE 双层路由持续学习及官方代码数据

## [2026-11-02] ingest | sources/sites/moss-api-platform.md — 收录模思开放平台接口与开源边界

## [2026-11-02] ingest | Khushi X Article《From RT1 to MolMoAct2: A brief history of VLAs》：10 个模型各一个唯一节点（新建 SmolVLA、MolmoAct，复用 8），含时间线地图

## [2026-11-02] ingest | FACT EgoForce：核对论文全文与项目页，补充流程图及人类力标签边界；代码和数据待发布

## [2026-11-02] ingest | sources/sites/roboatom.md — 补充 RoboAtom 官方发布入口和已有论文节点，待完成全文核对

## [2026-11-01] structural | roadmap 巡检：硬件设计/力矩电机/RL 运动控制三条纵深补入 11-01 新增知识节点链接

## [2026-11-02] ingest | sources/repos/unclebob-ub-mode.md — 收录编码代理技能及质量工具边界

- 已核对 README 与 SKILL.md；配套工具分属独立仓库。

## [2026-11-02] ingest | 智生于形《从开源仓库到真机差几步：人形机器人复现门槛与 Sim2Real 缺口》：新建复现门禁概念页（工程经验综述，无项目清单）

## [2026-11-02] ingest | 算法工程师的学习日志《从 1/4 车模型到最优控制器：MWORKS 主动悬架建模与参数优化》：新建主动悬架状态反馈与约束优化概念页（单作者实操，无项目清单）

## [2026-11-02] ingest | senlanke 具身运控lab《10.26-10.30前沿论文动态》Manipulation 57 篇逐篇建独立详情节点（新建 31 / 复用 26）

## [2026-11-02] ingest | sources/papers/human_superior_colliculus_predictive_7t_2026.md — 归档上丘感觉预测论文与公开分析代码，全文待核读

## [2026-11-02] ingest | sources/papers/any3r_arxiv_2610_36150.md — 收录 Any3R 论文、项目与推理源码，全文与附录待复核

## [2026-11-02] ingest | senlanke 具身运控lab《10.26-10.30前沿论文动态》人形/四足 33 篇逐篇建独立详情节点（新建 23 / 复用 10）

## [2026-11-02] ingest | GitHubStore《19款能自己造的开源人形机器人》：19 个项目逐项目建独立节点（新建 13 / 复用 6），含 README 核对与归类存疑标注

## [2026-11-02] ingest | GRAFT 地形残差适配

- 通读论文 v1，记录冻结跟踪器、下肢残差、蒸馏、评测与局限。
- 项目页未发现算法代码或权重入口；待后续核实开放状态。

## [2026-11-01] lint | health-check — 全量 lint 清零：DONIC/Spotter 陈旧页同步源核查进度、SeeR-VLA 补「与其他工作对比」、7 评测基准页 + 3 感知页 + 1 FOC 页补枢纽回链、6 篇非感知论文修正误标 perception tag、none/VGGT 加入缺页候选豁免

## [2026-11-01] ingest | 古月居转载《四足机器人开源项目深度解析》：36 个 GitHub 项目逐项目建独立节点（新建 24 / 复用 12 个节点），含 README 核对

## [2026-11-01] ingest | Zane Hub 公众号《灵巧手负载与核心指标》：新建 dexterous-hand-payload-core-metrics 概念页（工程经验综述，无论文/项目清单）

## [2026-11-01] ingest | Zane Hub 公众号《机器人关节方案选型》：新建 robot-joint-scheme-selection 概念页（工程经验综述，无论文/项目清单）

## [2026-11-01] ingest | Zane Hub 公众号《关节模组减速器的难度拆解》：新建 joint-module-reducer-difficulty 概念页（工程经验综述，无论文/项目清单）

## [2026-11-01] ingest | Zane Hub 公众号《关节模组电磁隔离》：新建 joint-module-electromagnetic-isolation 概念页（工程经验综述，无论文/项目清单）

## [2026-11-01] ingest | Zane Hub 公众号《关节模组：额定电流小于负载电流的后果》：新建 joint-motor-rated-current-overload 概念页（工程经验综述，无论文/项目清单）

## [2026-11-01] ingest | Zane Hub 公众号《关节模组空载：速度环与力矩环的输出扭矩区别》：新建 joint-module-no-load-velocity-vs-torque-loop 概念页（工程经验综述，无论文/项目清单）

## [2026-11-01] ingest | Zane Hub 公众号《关节模组轴的径向热膨胀校核》：新建 joint-module-shaft-thermal-expansion 概念页（工程经验综述，无论文/项目清单）

## [2026-11-01] ingest | Zane Hub 公众号《机器人结构公差仿真工具链》：新建 structure-tolerance-simulation-toolchain 概念页（工程经验综述，无论文/项目清单）

## [2026-11-01] ingest | Zane Hub《人形机器人结构设计工程师需要懂哪些跨专业知识？》：新建跨专业知识地图（无论文/项目清单，不做逐项目节点）

## [2026-11-01] ingest | sources/sites/qiayuan-liao-homepage.md — 收录个人主页与相关人形控制研究和开源入口

## [2026-11-01] ingest | Zane Hub 公众号《人形机器人线束类型与防护》：新建 humanoid-wire-harness-types-protection 概念页（工程经验综述，无论文/项目清单）

## [2026-11-01] ingest | 深蓝具身智能《四年十篇顶会Best Paper，谁在持续霸榜？》：正文 9 篇 CV 三大会 Best Paper 逐篇建独立详情节点（新建 7 / 复用 2）

## [2026-11-01] ingest | 具身智能小站 2026-11-01《模型还在思考，目标已经移动》56 篇逐篇建独立详情节点（新建 43 / 复用 13）

## [2026-11-01] ingest | sources/blogs/videodb_how_a_robot_remembers_2026-10-27.md — 收录外部视频记忆架构并区分 SDK 与服务端开放范围

## [2026-11-01] ingest | sources/repos/semi-design.md — 收录 React 组件库及设计系统，待补完整技术文档核对

## [2026-10-31] lint | health-check — 全量 lint 清零：TopoRetarget 4 个陈旧页同步官方代码、9 个评测基准页回链枢纽、GenSfM 补「与其他工作对比」

## [2026-10-31] ingest | sources/repos/deepseek_dsh_superagent.md — 记录 DeepSeek 编码代理原型及待核实的文章原文

## [2026-10-31] ingest | 具身智能小站 2026-10-31《一张照片变成仿真场景》50 篇逐篇建独立详情节点（新建 34 / 复用 16）

## [2026-10-30] lint | 全量 lint 修复：6 陈旧页 + 1 过期页复核，7 篇 paper-* 补「与其他工作对比」，codec4AI 补训练输入适配，12 页补评测基准 / 感知栈枢纽回链

## [2026-10-30] ingest | Reflex 仿真源码更新

- 更新已有 Reflex 节点：补充 arXiv 编号、仿真源码运行时序及复现边界。
- 修正人类接球成功率和 7 m 真机结果；训练与评测代码已公开，真机部署栈未公开。

## [2026-10-30] ingest | sources/courses/cs146s-modern-software-developer.md — 收录 2026 秋季课程日历与学习路径；讲义和代码入口待核实

## [2026-10-30] ingest | sources/repos/time-series-library.md — 收录时序分析实验库，核对源码入口与基准局限

## [2026-10-30] ingest | 具身智能小站 10-30 期 28 篇（Event2Reward 深读）逐篇建独立详情节点，含步骤 2.5 开源核查；节点为索引级摘要，未通读各论文全文

## [2026-10-30] ingest | sources/papers/codec4ai_arxiv_2610_33079.md — 收录稀疏解码与 DDRA，核查代码开放及实验边界

## [2026-10-30] ingest | sources/repos/crowdbricks.md — 收录 SOMA 批量运动生成运行时与性能、许可边界

## [2026-10-30] ingest | 具身智能研究室 SuperMimic 42 个社区复现/扩展项目盘点归档（raw + blog 清单），论文页补生态概览；未逐仓核查、未新建项目页

## [2026-10-30] ingest | 深蓝《路径规划5大主流算法详解》10-30 重发版（具身智能基础第11篇）归档 raw，并入既有 source 与五范式对比页，未新建 wiki

## [2026-10-29] ingest | shadcn-admin

- 收录管理后台 UI 集合，源码公开且使用 MIT 许可；记录技术栈、使用边界与组件更新注意事项。
- 主节点：[shadcn-admin](../wiki/entities/shadcn-admin.md)。

## [2026-10-29] ingest | sources/papers/ventral_groove_fibers_aeh5116.md — 收录仿生腹沟纤维，基于摘要概述并标注全文与源码待核实

## [2026-10-29] ingest | InstantTrack

- 收录 G1 仿真动作跟踪项目，记录训练流程、实验口径、接触优化及开源边界。
- 项目页未能直接访问；依据源码仓库 README 核查。

## [2026-10-29] ingest | sources/blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md — 具身智能小站 10-29 期 58 篇（新建 35 节点 / 复用 23）+ 阅读坐标页

## [2026-10-29] ingest | senlanke《借着supermimic复习：Model base，可微世界模型，BPTT》公众号：点名项目逐一映射唯一节点，新建 TD3 / MBPO / BPTT，其余复用并补反链

## [2026-10-29] ingest | sources/papers/phase_conditioned_loco_manipulation_arxiv_2610_36270.md — 收录多阶段 PPO、真机部署与代码状态待核实

## [2026-10-29] ingest | PhysWord 项目页

- 核对项目主页，补充机构、方法与评测信息，复用已有论文节点。
- 项目页标注代码即将发布，未列数据或权重下载链接。

## [2026-10-29] ingest | Pico.Lab | 嵌入式调试工具

- 收录官方仓库及 Hackaday 项目页，归纳波形、总线和电源功能。
- 固件、PCB 与 CAD 已公开；明确普通 Pico 与专用板的功能边界。

## [2026-10-29] ingest | sources/papers/ssio_arxiv_2610_25988.md — 收录 SSIO 自监督惯性里程计；代码、权重与 G1 数据公开，复用已有 cuArms 与 DeFM 节点

## [2026-10-28] lint | 全量 wiki lint 修复：7+3 陈旧页复核并回链新来源，11 篇 paper 补「与其他工作对比」，11 篇评测基准页回链评测基准选型闭环，PartLLM 回链感知栈闭环；问题与信息型预警清零

## [2026-10-28] ingest | 未知载荷下的自适应全身 MPC

- 收录 arXiv:2610.12972，关联质心动力学与 MPC/WBC 集成。
- 依据摘要整理；全文与官方代码开放状态待核实。

## [2026-10-28] ingest | sources/papers/omni_woodpecker_arxiv_2610_27103.md — 收录音视频幻觉纠正方法及官方开源推理入口

## [2026-10-28] ingest | 具身智能小站 2026-10-28 盘点 59 篇（深读免渲染事件仿真；59/59 独立节点：新建 55 / 复用 4）+ 技术地图

## [2026-10-28] ingest | ThinkWorld 世界想象与动作推理

- 收录论文与技能组合实验，关联 VLA 和世界模型知识。
- 项目页访问失败，代码、权重与数据开放状态待核实。

## [2026-10-28] ingest | 复核 FluxVLA 论文 v2、公开源码及 Hugging Face 模型卡，记录基准口径差异并复用既有实体

## [2026-10-28] ingest | sources/sites/worldarena2-challenge-iros2026.md — 补充赛事三赛道与指标修正；PACE 和 ForeTac 已有实体页

## [2026-10-28] ingest | 收录 Humanize 工作流及四项算子生成 benchmark，补充评测基线与硬件界限的区别

## [2026-10-28] ingest | sources/papers/robosimstudio_arxiv_2610_18835.md — 深读论文与源码，补充多物理、反馈闭环和擦拭示例实现；源码已公开

## [2026-10-28] ingest | sources/papers/defm_arxiv_2601_18923.md — 收录深度表征与机器人实验；公开推理代码和权重，vision-demos 复用已有节点

## [2026-10-28] ingest | sources/repos/genie_sim.md — 核查官方开源边界并补充双运行栈与 ROS 2 实践

## [2026-10-28] ingest | sources/papers/learning_dexterous_manipulation_survey_2610_33486.md — 整理灵巧操作训练与评测综述，逐篇数据可向作者索取

## [2026-10-28] ingest | Proteus 跨人形动作跟踪项目页

- 归档项目页并整理共享关节布局、本体描述、跨仿真器评测及未见本体泛化边界。
- 论文和代码标注即将发布；网页提供 ONNX 仿真演示，真机表现待验证。

## [2026-10-28] ingest | sources/repos/jeremyipark-vision-demos.md — 收录运动视频分析示例，注明开源代码与外部模型调用边界

## [2026-10-28] ingest | sources/papers/self_improving_agents_embodied_lifelong_learning.md — 对比自我改进机制与具身终身学习，复用 RISE、SERL、Arcadia 节点并记录开放边界

## [2026-10-28] structural | roadmap/depth-cross-embodiment.md — 联网核对一手论文，新增「谱系速览」表（Devin 2017 → NerveNet → RoboNet → SMP → MetaMorph → OXE → Polybot → Mirage → Octo → CrossFormer → HPT → URMA → RoVi-Aug → LAPA → Embodiment Scaling Laws）并在各 Stage 补一手出处

## [2026-10-28] lint | 修复 2 个陈旧页面：contact-rich-manipulation.md 补入 GENESIS-Handover（arXiv:2610.08003）；paper-harness-vla.md 同步 RoboFoundry 2026-10-28 开源状态复核

## [2026-10-28] structural | roadmap/depth-cross-embodiment.md — 新增第 28 条纵深路线「跨本体迁移（Cross-Embodiment）」：本体差异画像 → 统一接口 → 形态感知架构 → 多本体混训 → 人→机 → 新本体适配；同步首页按钮、路线总览、README/index、depth-filters 与 topics 种子

## [2026-10-28] ingest | sources/papers/grifips_science_advances_2026.md — 收录梯度夹层触觉传感器并补充触觉感知；代码与数据开放状态待核实

## [2026-10-28] ingest | 收录三篇 Claude 使用与上下文工程文章，补充 agent 工作中的未知、目标与验证边界

## [2026-10-28] ingest | sources/papers/world_t3_arxiv_2610_17434.md — 收录位姿索引的测试时记忆世界模型；项目页暂未列代码或数据链接

## [2026-10-28] ingest | 补充人形 VLA、潜在动作与技能工程四项研究；复用既有 Machenta 和 MimicAll 页面，记录官方仓库及项目页核查状态

## [2026-10-28] ingest | sources/papers/partllm.md — 收录三维部件分割与官方推理、训练入口；项目页源码列有代码和权重链接

## [2026-10-28] ingest | 收录 FlowHMR、RLMuscle、AutoPhyX 并复核 RoboFoundry 官方代码状态

## [2026-10-28] ingest | Modly 本地三维网格生成

- 应用代码以 MIT 许可开放；模型扩展与权重分别核查。记录机器人仿真资产的验证边界。
- VoiceStudio 已有实体和来源归档，本次复用已有节点。

## [2026-10-28] ingest | 收录布料显式摩擦、MimicAll 与 CADonatello，注明开放状态核查边界

## [2026-10-27] lint | 全量 wiki lint 清零：修复 20 条信息型预警（12 篇 paper 补「与其他工作对比」、Kine2Go/StreamEval 补评测章节、4 篇感知页回链感知栈枢纽、Gen-HumanWholeBody 按 HF 数据卡补许可证与模态、sim2sim 并入 Sim2Real 缩写速查、Default/inference 停用词）

## [2026-10-27] ingest | sources/papers/reduced_3dgs.md — 补充 3DGS 内存压缩与复现边界

- 在空间表征页面补充 Reduced 3DGS 的实验结果与部署边界，复用主分支已有论文、项目页和代码归档。
- 3DGS 与 NeRF 原论文已收录，沿用现有节点。

## [2026-10-27] ingest | sources/sites/machenta-instant-robot-learning.md — 收录 Machenta 单次示范即时学习

- 汇总仿真训练、真实示范提示及公司自报的跨机器人和一小时测试；官方文章未列公开代码、权重或数据。

## [2026-10-27] ingest | sources/repos/wise-motion.md — 收录开源动效参考技能及源码导出流程

## [2026-10-27] ingest | sources/repos/deepfounder-desigma.md — 收录开源界面设计工具及 MCP 接口

## [2026-10-27] ingest | sources/repos/muscle-supermimic.md — 收录肌骨世界模型跟踪项目、实验与开放边界

## [2026-10-27] ingest | sources/repos/somark-oxr.md — 收录 OXR 文档解析模型，代码及权重公开，记录部署与许可边界

## [2026-10-27] ingest | 复用 PLaW-VLA、V-JEPA 2、π0.5、Motus 与 LIBERO-Plus 现有节点，补充方法和评测关系

## [2026-10-27] ingest | FastHarness 官方资料

补充官方代码归档和项目页入口，核对开放状态并完善已有实体的机制、实验与运行时序；项目页本次不可访问，数据状态待核实。

## [2026-10-27] ingest | sources/papers/wevisone_vl_arxiv_2610_31329.md — 收录统一视觉感知、缺失目标拒答及官方开放状态

## [2026-10-27] ingest | sources/papers/holi-world.md 等 — 收录 Ropedia 第一视角重建与空间推理，复用 W0 和 DONIC，记录开源状态

## [2026-10-27] ingest | sources/repos/unlimited-ocr-2.md — 收录百度长文档解析模型及推理入口

- 官方仓库提供推理代码并链接权重；训练资料开放范围与权重许可待核实。

## [2026-10-27] ingest | SimpleTouch

- 收录论文、项目页与公开代码，归纳触觉专家与评测；项目页和权重页面访问失败，开放状态待核实。

## [2026-10-27] ingest | GENESIS-Handover | 生成式任务导向人机递物

- 意图：归档论文并提炼任务导向递物的候选生成、手型选择与真机评测。
- 开源结论：arXiv 页面未列项目页或官方代码仓；论文附录所含提示词代码不构成完整可运行实现。
- 关键页：[GENESIS-Handover 实体](../../wiki/entities/paper-genesis-handover.md)、[操作任务](../../wiki/tasks/manipulation.md)。
- 证据：arXiv HTML 全文（v1），尤其方法、评测与失败分析。

## [2026-10-27] ingest | sources/blogs/wechat_githubstore_18_open_source_robot_projects.md — GitHubStore 18 个开源机器人项目：12 个新独立实体节点 + 6 个既有节点复用，索引 overview；fetch_wechat_article.py 抓取成功，未核对官方仓库

## [2026-10-27] ingest | Roboparty/RP1 总览仓库全部子项目补齐独立详情节点：新增 rp1-hardware、romomo、rp-hand、rp1-test、roboparty-joystick、rp1-description、rp1-firmware、robolab 的 AMP/BeyondMimic/Parkour 三模块与 VIStream；roboparty_train/deploy 与旧 atom01 页为同一仓库改名（HEAD 相同），合并为 roboparty-train/roboparty-deploy 并登记别名；核验并更新 P-BFM（已发布训练/推理代码）、I-BFM（已发布 ibfm 代码）、UFO、MimicLite、Party OS 开源状态。总览仓库仅 README，无工程代码；rp1 总览页『RP1 部署 Coming soon』『训练仅 RPO』已按仓库现状修正。

## [2026-10-27] ingest | awesome-supermimic 社区索引

- 收录 DiaoOrg/awesome-supermimic，记录其作为社区导航清单的定位及 CC0 策展许可。
- 更新 SuperMimic 实体页，明确资源索引不代表官方代码，也不独立验证各复现结果。

## [2026-10-27] ingest | 收录 Replica Skill 并整理应用重建 Agent Skills 的流程与使用边界

- 核验官方 GitHub README 与 MIT 许可，记录 11 阶段技能工作流及 Python 工具。
- Wiki 说明其可用于机器人周边软件原型的工作流参考，并明确不提供机器人控制实现。

## [2026-10-27] ingest | RoboParty RP1 — 官方开源范围与子仓库核查

新增 RP1 官方产品/文档和仓库归档及实体页，记录硬件、模型、固件、ROS 2 部署与训练入口；对照 RPO 既有条目，注明当前训练/部署文档的范围差异及分仓许可，避免把厂商规格写成独立实测。

## [2026-10-26] lint | 全量 lint 清零：8 个陈旧页补回链、lcp.md 修正公式与论文出处、7 条信息型预警（Pace 停用词 / 3 篇 paper 补对比 / 2 页回链知识链枢纽）

## [2026-10-27] ingest | Every Joule — 收录 Claude + Three.js 十课互动物理教学项目，并注明未发现官方源码仓库

## [2026-10-26] ingest | AgenticRoboRL — 归档论文、项目页与已开源仓库，并建立统一 planner 后训练实体页

- 项目页与仓库核实：代码 Apache-2.0，Qwen3.5-9B checkpoint 已发布；teacher SFT 轨迹未发布。
- 重点记录 RPent / CaP-X 两类 harness、SFT + 多轮 RL 流程、LIBERO 结果与复现边界。

## [2026-10-26] ingest | RoboCompiler-Humanoids（arXiv:2610.14237）

- 复核已存在的收录，依据 arXiv 摘要与作者 README 补齐方法、数值验证和复现入口。
- 开源状态：代码、报告结果、论文手稿和视频公开，Apache-2.0；实验为 MuJoCo / Pinocchio 仿真，未据此外推真机性能。
- 更新论文、仓库来源归档与 wiki 实体页。

## [2026-10-26] ingest | Anthropic 以 Claude 扩展形式化软件验证规模

- 收录 Anthropic 与 Theorem 关于 OSS-Fuzz 二进制级 Rocq 证明的研究文章及公开证明/语义仓库。
- 区分文章实验总体（123 个库、239 个入口）与公开仓库当前展示的子集（25 个库、60 个入口），记录 RISC-V/Linux 等证明假设。

## [2026-10-26] ingest | ARROW — 收录顺序无关的 4D 重建与跟踪实现

- 收录 ARROW 论文、项目页和官方仓库；核实推理代码、权重与交互式查看器均已发布。
- 记录代码 MIT / 部分源码 Apache-2.0 与权重 CC BY-NC 4.0 + Waymo 非商业条款的许可边界。
- 在 D4RT 页面加入对照，明确 ARROW 借鉴其查询接口，但属于独立项目。

## [2026-10-26] ingest | sources/repos/blender-skills.md — 收录 94 个 Blender 代理技能及 MCP 资产工作流

## [2026-10-26] ingest | RealSimLoop：收录在线可微 Real-to-Sim 论文

- 归档 arXiv:2609.09828，新增论文 source 与论文实体页，并与 One-Shot Real-to-Sim 建立互链。
- 论文报告滑窗视觉反馈、神经降阶仿真和伴随梯度可在线估计时变材料参数；双臂软杆场景约 6.23× 加速。
- 开源状态：论文称录用后发布代码；截至入库日未提供公开仓库链接。

## [2026-10-26] ingest | senlanke 具身运控lab 10.19-10.23 两篇周报（人形/四足 39 + Manipulation 37）：72 个唯一 arXiv，新建 48 篇独立 paper-* 节点，复用 24 篇，新增 2 篇 blog 源与 2 张技术地图

## [2026-10-26] ingest | RoboMIND v1.0 / v2.0 论文、项目页与数据入口

## [2026-10-25] structural | roadmap/depth-{bfm,rl-locomotion,contact-manipulation,motion-retargeting,navigation,loco-manipulation,embodied-data}.md — 巡检最新知识节点，补入 12 个未被引用的新页面（SuperMimic/MorphMimic/ScaleBFM/Push the Limit/IL 接触综述/HandsOn/STR/SuperNav/NavMCP/EvolvingNav/WholeBodyMem/SoftMimicGen）

## [2026-10-25] lint | 全量 wiki lint 清零：修复 12 条警告与 9 条信息型预警（atom01-train 过期复核、11 页陈旧页面按新 sources 复核回链、modeling-and-solving 并入 Blackmore 来源、paper 三段式/缩写速查/评测枢纽回链补齐、Generate 停用词）

## [2026-10-26] ingest | Real-Time EXPO-FT 官方代码状态复核

- 官方项目页现链接至 `pd-perry/expo-ft`；更新论文与项目页归档的开放状态。
- wiki 补充 README 中 server/client、异步训练入口及 OpenPI/DROID 分支复现路径。

## [2026-10-26] ingest | Robotics Toolbox for Python — 归档官方仓库、在线文档与 ICRA 2021 论文入口；整理机械臂及移动机器人算法能力和实机部署边界。

## [2026-10-26] ingest | NVIDIA Isaac Sim Agent Skills 官方文档与技能索引

归档官方文档及源码索引，补充 Isaac Sim 实体页的 Agent 工作流和安全边界说明。

## [2026-10-26] ingest | NVIDIA Isaac Lab Agent Skills 官方资料

归档官方技能设计、3.0 EA 发布记录及 RL 调试和 PhysX/Newton 策略迁移示例，补充现有 Isaac Lab 实体页。

## [2026-10-26] ingest | blurt 工程反馈整理工具

- 来源：AGI Hunt 官方仓库 README；项目页列于 README。
- 开源结论：仓库公开，MIT 许可证；README 描述本地优先语音转写并提供多平台工作流。
- 判断：作为通用 AI 辅助工程工具归档至 sources；暂不升格为机器人知识页。

## [2026-10-25] ingest | Artisan

- 收录 Artisan 论文、项目主页、官方代码和 benchmark 数据入口；建立统一项目实体页，归纳其多智能体资产生成与物理仿真验证流程。

## [2026-10-25] ingest | NVIDIA SimReady Assets — 归档 Hugging Face 数据集规模与开放边界

- 核对 NVIDIA 的 Hugging Face 数据集页、SimReady Central 公告及官方文档。
- 记录页面所示 2,575 个 3D Assets、222 行 Viewer 数据与 472 GB，并注明页面尚无 dataset card。
- 明确 SimReady 属性随资产/profile 而异，许可需按资产包核查；补充现有 SimReady Central 实体页与来源映射。

## [2026-10-25] ingest | 归档 SpaceX 火箭回收动力下降的一手资料

- 收录 SpaceX 着陆工程师 Lars Blackmore 关于 F9R 精确着陆的技术文章，以及 SpaceX 官方 Grasshopper 垂直起降测试视频。
- 区分 Grasshopper 悬停测试与 Falcon 9 轨道返回时的末段着陆燃烧；未将 G-FOLD 推断为 Falcon 9 实际使用算法。

## [2026-10-25] ingest | sources/papers/push-the-limit-humanoid-sprinting-arxiv-2610-14840.md — 收录 CoRL 2026 人形高速冲刺与转弯论文及实体页

## [2026-10-25] ingest | HandsOn：归档跨具身灵巧操作基准、PhysX 示范数据与开源边界

- 论文、项目页、官方仓库和 Hugging Face 数据卡已互相核对；源码与 PhysX 数据可公开获取，Newton 数据待发布，模型 checkpoint 未随仓库分发。
- 建立统一项目实体，并链接至操作任务总览；不把闭环任务成功率与动作预测误差混为一谈。

## [2026-10-25] ingest | 接触丰富任务中的模仿学习综述

- 收录 SAGE / IJRR 2026 综述，覆盖接触操作示教、传感模态、IL 方法与应用。
- 该文为综述；文章页未列独立项目、代码或数据入口。
- 将论文实体关联到现有 IL 与接触丰富操作知识页。

## [2026-10-25] ingest | PageIndex | 收录无向量、基于树索引与推理的长文档 RAG 引擎

- 开源结论：官方 GitHub 仓库公开 Python 实现并采用 MIT 许可证；官网与仓库及文档互相衔接。
- 关键页：归档 repo 与官网资料，并建立 PageIndex 实体页；补充 DeepTutor 对应的 RAG 引擎交叉引用。

## [2026-10-25] ingest | MorphMimic — 收录报告、项目页与官方实现，整理跨本体共享方法和仿真/真机评测边界

- 官方代码已开源；核对其与 SuperMimic 的关系，注明基础算法为作者独立复现。
- 归档 128,786 段公共动作评测、统一策略仿真生存率和 G1 真机演示；区分仿真结果与硬件结论。

## [2026-10-25] ingest | sources/papers/openbot_arxiv_2008_10631.md — 归档 OpenBot 手机驱动低成本轮式机器人的论文与源码，关联既有 PhoneBot 人形平台

## [2026-10-25] ingest | sources/papers/wholebodymem_arxiv_2610_31166.md — 补录 arXiv HTML v1 明示的 MBZUAI 作者单位，并同步项目页归档与 WholeBodyMem 实体标签

## [2026-10-24] lint | 全量 lint 清零 19 条信息型预警：3 篇 paper 补英文缩写速查与结论，10 篇 paper 补方法/评测/对比三段式，LynxScenes 补重定向就绪度，4 页补选型闭环枢纽回链

## [2026-10-24] ingest | Learn ESP32 — 收录交互课程与 MIT 源码，标明浏览器教学模型和真机验证边界

- 归档课程站、公开源码与 16 课结构，补充 ESP32 入门路线及其与 Wokwi 的用途区别。

## [2026-10-24] ingest | sources/papers/from_video_to_vision_arxiv_2610_14561.md — 收录 MOVER 视频驱动人形搬运论文并链接 Loco-Manipulation

## [2026-10-24] ingest | sources/repos/horizon_robotics_holomotion.md — 更新 HoloMotion v2 预览、SuperMimic 与 H-bot 官方资料

## [2026-10-24] ingest | sources/sites/nvidia-simready-central.md 与 sources/repos/nvidia-simready-central.md — 归档 NVIDIA SimReady Central 及其 Agent 仿真资产工作流

## [2026-10-24] ingest | Konnect — 归档 KiCad 10 的 AI/MCP PCB 设计插件

- 核查官方仓库 README 与许可证：源码已公开，标注 AGPL-3.0，项目处于 Beta。
- 收录 Rust/MCP 与 KiCad IPC 工作流，补充机器人板级硬件设计中的工程复核边界。

## [2026-10-24] ingest | 收录 HumanoidPose 视频人形姿态估计论文、ETH 项目页与待发布代码状态

## [2026-10-24] ingest | Qucs-S 电路仿真工具

- 收录官方项目主页与源码仓库；确认 GUI 已开源（GPL-2.0），仿真依赖外部后端。
- 新增 Qucs-S wiki 实体，说明它在机器人板级电子分析中的用途与适用边界，并关联 KiCad。

## [2026-10-24] ingest | OmniInteract：归档论文与项目页，并将双人形接触交互接入多机协调知识图

- arXiv:2610.26043 提出 SCAR 表面接触感知重定向与基于 SONIC 冻结先验的流匹配残差策略。
- 官方项目页提供演示入口；截至入库日未见可运行代码或数据下载链接。
- 建立论文实体并从人形多机协调概念页添加关联入口。

## [2026-10-24] ingest | DistScene 单图组合式 3D 场景生成

- 归档论文与官方项目页；官方代码、checkpoint、数据集仍标为待发布。
- 编译 DistScene 实体页并关联 Real2Sim 纵深路线，区分几何指标与机器人仿真可用性。

## [2026-10-24] ingest | RTNav — 实时零样本物体导航与异步部署

- 归档论文、项目主页、仿真评测代码和 Stretch 3 / Jetson AGX Thor 部署仓库。
- 在 ObjectNav 任务页补充实时墙钟评测与异步系统案例。

## [2026-10-24] ingest | Jev 仿真机器人与自动驾驶应用

- 意图：收录三个 Jev 应用演示，补全 TypeSafe Jev 在无人机、驾驶模拟和机械臂操作中的工程案例。
- 开源结论：jev-drone 与机械臂演示为 MIT 开源；JevPilot 仓库公开但未声明许可证。三项均是 MuJoCo 或 Three.js 仿真，未将演示性能外推为实机结论。
- 写回：更新 Jev 总览及对应仓库 / 项目页来源归档。

## [2026-10-24] ingest | StreamSplat（ICLR 2026）— 在线动态 3DGS 重建

归档论文、作者项目页和官方代码；记录其开源代码与 checkpoint 入口，并建立动态场景重建 Wiki 实体。该方法是通用视频重建研究，机器人导航/控制闭环仍需另行验证。

## [2026-10-24] ingest | touchanything-egotouch | 收录视触觉估计论文、数据与实现

- 意图：将 TouchAnything 及 EgoTouch 的论文、项目页、代码和数据集入口纳入机器人研究知识库。
- 开源核查：代码公开，MIT 许可；EgoTouch 公开托管于 Hugging Face。采集说明与转换后压力网格尺寸不同，页面明确保留该核查点。
- 写回：新增一篇论文来源、一篇项目页来源、一篇代码仓来源及一个统一 wiki 项目节点，并链接至视触觉融合概念页。

## [2026-10-23] lint | 全量 lint 清零 3 条信息型预警：SuperDyno 补 venue、From Hitter to Player 对比章节标题、GameHorizon 回链评测基准选型闭环

## [2026-10-23] ingest | CTP | 复核机构与代码开放状态

- 复核官方仓库 README：确认其为论文官方实现，列出数据转换、两阶段训练与开环评测入口，并确认 README 中的五个机构。
- 更新已有论文来源、代码来源和 CTP 实体页；保留项目页在 2026-10-17 核查时未列代码链接的记录。
- 开源结论：训练及开环评测代码已公开；许可证、权重与真机数据状态未核实。

## [2026-10-23] ingest | gamescaling | 归档论文、项目页与推理实现

- 意图：将 GameScaling 的游戏具身智能体持续预训练研究纳入知识库，并与游戏基准及强化学习路线互链。
- 开源结论：官方已公开推理客户端与模型权重（Apache-2.0）；训练代码和游戏训练语料未公开。
- 关键页：论文、项目页、代码仓库来源归档及 GameScaling wiki 实体页。

## [2026-10-23] ingest | RoboAug 官方代码与数据发布后更新既有实体

- 更新论文、项目页与 wiki 中的机构及开源状态：官方来源列东南大学；代码仓库开放 Stage 1–2，数据集已发布，Stage 3 区域对比策略学习仍待开源。

## [2026-10-23] ingest | OpenCAL 计算轴向光刻打印机与配套资料

- 收录 OpenCAL 文档、控制软件、CAL-hardware、Tomo、VAMToolbox 与两篇 CAL 论文的一手来源。
- 建立 OpenCAL wiki 实体并关联 Blender 上游网格工作流；记录 VAMToolbox 许可边界。

## [2026-10-23] ingest | ACP：以接触规划连接动作先验与人形移动

- 归档 arXiv 论文、项目页入口和代码/数据开放状态核查；项目页当前无法读取，开放状态标为待核实。
- 新增 ACP 论文实体，并挂接到人形移动索引。

## [2026-10-23] ingest | Awesome-Affordance-Learning 清单逐篇入库：206 篇去重论文，新建 198 个索引级节点，8 篇复用站内既有页；新增技术地图并链入 affordance 综述页

## [2026-10-23] ingest | sources/models/zero-to-cad-qwen3-vl-2b.md 与 sources/datasets/zero-to-cad-100k.md — 收录 Zero-to-CAD 模型、精选集与论文，核实 Apache-2.0 发布范围

## [2026-10-23] ingest | 补充 HiRe-MoMa 底座与机械臂分层修正的因果说明

- 已有论文实体页与原始来源归档；本次在实体页中解释底座位置误差如何传递到末端，以及手臂残差为何条件化于底座修正。
- 按论文消融区分动作头拆分与单向条件的贡献；官方项目页未提供代码仓库链接。

## [2026-10-23] ingest | 收录 NIDAR 论文、项目页、代码与模型工件

- 交叉核对 arXiv、项目页、GitHub 和 Hugging Face；记录项目页未同步及仿真验证、数据和许可边界。
- 新增来源归档与实体页，并从 LiDAR 传感概念页建立站内入口。

## [2026-10-23] ingest | BeJEPA | 世界锚定的人形行为世界模型

- 意图：整理 BeJEPA 论文的双预测器方法、长程跟踪结果与 G1 真机验证。
- 开源状态：arXiv 未提供代码仓库链接；项目页本次访问超时，代码开放状态待核实。
- 关键页：[BeJEPA](../wiki/entities/paper-bejepa.md)、[论文来源](../sources/papers/bejepa_arxiv_2610_23127.md)、[项目页核查](../sources/sites/bejepa-project-page.md)。

## [2026-10-23] ingest | 收录 Robotics Bench 交互式硬件教学演示

- 归档 Robotics Bench 与 Colorfigures 官方说明，并新增知识实体页。
- 将演示定位为浏览器交互式教学图形；源码开放状态和电路级仿真能力尚未核实。

## [2026-10-23] ingest | HeyPCB — 归档 AI PCB 设计服务及其开源状态

- 官方首页介绍自然语言生成 PCB 原理图、布局与制造文件，并提供 Circuit World 开放硬件目录。
- 截至入库日，首页未列出产品代码仓库；记录为未提供代码链接，未将产品能力陈述当作独立验证。
- 建立项目实体并连接机器人硬件与 EDA 页面。

## [2026-10-23] ingest | OpenTac 官方触觉皮肤生态归档：新增项目实体与 CAD/读取/仿真/Xampling 说明

## [2026-10-23] ingest | SOMA：核实 SKIM V2、更新论文与项目资料，并补齐评测及数据访问边界

- 更新现有论文、项目页、源码归档和唯一 Wiki 实体，不另建重复节点。
- 根据官方仓库 README 与数据门户，记录 2026-10-20 SKIM V2 / RELEASE_V2 发布及登录要求。
- 补入 arXiv v2 未见动作评测数值，并说明全局 MPME 与 LBS 的差异。

## [2026-10-23] ingest | COWBOY CoRL 2026 — 记录项目主页开源状态与全身操作方法

- 官方项目页报告：24 个真实场景、100 多种物体、91% 抓取成功率；整机策略协调 32 个自由度。
- 项目页未列出代码仓库，因此归档中标记源码入口未确认，不作开源承诺推断。
- 论文、项目页进入 sources，核心方法与评测边界整理至 wiki 实体页，并链接到移动操作任务页。

## [2026-10-23] ingest | CompactSplat（NeurIPS 2026）— 收录前馈 3DGS 纹理自适应分配方法、公开实现与复现差异

## [2026-10-23] lint | 全库过期巡检：outdated 与陈旧声明均为 0；补齐唯一来源更新未同步的零样本目标导航节点，纳入 MoCaP 的操作站位约束、分项评测及实验边界。

## [2026-10-23] ingest | MoCaP — 归档论文、项目页与公开代码，补充移动操作中的抓取站位知识

## [2026-10-23] ingest | 归档 spectral swarm robotics 期刊论文

- 记录正式发表信息、预印本摘要、作者与作者单位，并注明预印本可核对范围及期刊落地页访问限制。
- arXiv 未列项目页或代码仓库；本次只新增来源归档，暂不升格为 wiki 实体页。

## [2026-10-22] ingest | sources/papers/duplexagent_rsi_arxiv_2610_11299.md — 收录全双工语音协作的 harness RSI 论文、项目页开放状态与核心实体页

## [2026-10-22] ingest | sources/blogs/wechat_tiannan_sharpa_academic_route_2026-10-22.md — 天南 Sharpa 学术路线：15 个项目逐个对应独立节点，新建 POISE/SharpaDex/CraftNet/D01/W02/AE01 六个节点，其余复用；POISE 代码未发布，SharpaDex 无官方页面可核

## [2026-10-22] ingest | sources/papers/nerf_arxiv_2003_08934.md、sources/papers/3d_gaussian_splatting_arxiv_2308_04079.md — 归档 NeRF 与 3DGS 奠基来源并补充具身空间表征选型

## [2026-10-22] ingest | 收录无配对跨模态几何对齐研究

归档论文、作者项目页和官方代码，并整理统一论文实体。结论聚焦共享表示几何可支持粗粒度无配对对齐；CKA 质量分析仍依赖少量配对样本，且论文没有机器人实证。

## [2026-10-22] ingest | 收录 PlatformIO Core 官方仓库

- 归档 PlatformIO Core 的官方源码、Apache-2.0 许可与 CLI/构建/包管理能力；区分固件工程工具和机器人实时控制。
- 新增工具实体页，并在 Wokwi 相关知识页补充工具链交叉引用。

## [2026-10-22] ingest | 更新 img2threejs v2 官方归档与项目页

- 将主仓身份更新为 `img2threejs/img2threejs`，记录 Skill v2.0.0、Apache-2.0 与 GitHub API 的 18,663 stars 快照。
- 增补官方在线画廊归档，并修订 wiki 对基础代码重建、独立插件及机器人仿真资产边界的说明。

## [2026-10-22] ingest | MOSS-Image 与 MOSI 生态官方资料归档

- 新增 MOSS-Image 仓库、项目页和 arXiv 技术报告来源归档及统一 wiki 实体页。
- 记录 Base/SFT 权重、API 状态与许可证；将 MosMos 作为独立的关联桌面产品归档。
- 论文指标以作者自报结果标注，明确模型本身不是机器人控制模型。

## [2026-10-22] ingest | 收录 RoboDoctor RSI 项目，记录官网 54 项任务结果并标注代码待开放

- 归档项目页与官方 Preview 仓库；核实其 GitHub 仓库当前为空，官网尚未发布论文/arXiv。
- 官网数据（2026-10-21）：Warmup 49.74，RSI 57.10；RSI 301/324 个有效 episode，46/54 项任务完整。
- 将项目作为具身 RSI 案例关联至递归自我改进概念页。

## [2026-10-22] ingest | Maxinsights Physical AI 第一人称数据服务

- 归档 Maxinsights 官网与官方数据规模/密度文章，并同步 Dyna Robotics 的 Dyna-2 官方技术报告作为对照一手资料。
- 官网自报累计经验 200 万小时、月产能上限 45 万小时；将产能、累计量与客户交付区分记录。
- 公开一手资料未核实 Maxego、5000 名采集员、2024 年成立或具体客户关系；Dyna-2 报告称数据来自未具名合作方及内部采集，未点名 Maxinsights。
- 新增公司实体并接入 Ego 数据采集分类页。

## [2026-10-22] ingest | Scalabot（松延动力技术子品牌）归档并新增 Noetix 公司路线

## [2026-10-22] ingest | wiki/entities/zhaima-common-know-how.md — 宅码 Common Know-How 飞书文档升格为独立信息节点，补全文主题索引

## [2026-10-22] ingest | sources/papers/yoco_arxiv_2610_11657.md — 收录 YOCO 论文并升格实体页，遥操作页回链

## [2026-10-22] ingest | sources/papers/cosmos_interactive_world_simulation_tech_report_2026.md — 收录 Cosmos Interactive World Simulation 技术报告与四个开源模型，升格项目实体页，Cosmos 3 与 NVIDIA Cosmos 页回链

## [2026-10-22] ingest | sources/papers/spotter_arxiv_2609_36808.md — 收录 Spotter 论文与 MIT 代码仓核查，升格实体页，具身基础模型枢纽回链

## [2026-10-22] ingest | sources/papers/vla_rolling_arxiv_2610_25043.md — 收录 VLA-Rolling（论文、项目页列出的代码与模型）并升格实体页，VLA / action-chunking 页回链

## [2026-10-22] ingest | sources/papers/adadex_arxiv_2610_28628.md — 收录 AdaDex 并升格实体页，操作任务页回链

## [2026-10-22] ingest | sources/papers/uniwalm_arxiv_2610_28626.md — 收录 UniWALM 并升格实体页，世界动作模型页回链

## [2026-10-22] ingest | sources/blogs/schaeffler_newton_digital_twin_2026-10-21.md — 收录 Schaeffler × NVIDIA Newton 人形执行器数字孪生新闻稿，Newton 页回链

## [2026-10-22] structural | 实验室路线新增 RoboParty Lab（lab.roboparty.com）

## [2026-10-22] ingest | sources/blogs/nvidia_isaac_platform_gtc_berlin_2026-10-21.md — 收录 NVIDIA Isaac 平台新闻稿（Isaac OS、Isaac Lab 3.0、Human 1M、Agile ONE SN），更新 GTC 表格

## [2026-10-22] ingest | sources/sites/supermimic-project.md — 补项目页 wall-clock >50× 与脚注换算核查

## [2026-10-22] ingest | sources/papers/towards_agentic_robots_arxiv_2610_20442.md — 记录 arXiv 版本核查（仅 v1）

## [2026-10-22] ingest | sources/papers/harnesspai_arxiv_2609_29166.md — 记录 arXiv v2（2026-10-21）版本信息

## [2026-10-22] ingest | sources/sites/mobiagent-dual-loop-self-improvement.md — 补 HKU/Astribot/SUSTech 项目页机构与 Hugging Face 权重链接

## [2026-10-22] ingest | sources/papers/legged_robots_advances_challenges_scirobotics_2026.md — 补牛津 ORA（CC BY 4.0 accepted manuscript）、KAIST Pure 记录与 ETH RSL 未收录核查

## [2026-10-22] ingest | EmbodyHOI / AIM / ERR / Agensh / StairMaster / EVO-WAM 论文入库（sources/papers + wiki/entities/paper-*），飞书 Common Know-How 归档 sources/blogs；补 hub 回链

## [2026-10-21] ingest | 具身运控lab 10-12 至 10-16 Manipulation 33 篇（公众号）— 19 篇新建独立详情节点，14 篇复用已有节点（arXiv 去重）；新增 wiki/overview/senlanke-manipulation-10-12-16-33-papers-technology-map.md

## [2026-10-22] ingest | PhoneBot 项目页开放状态复核

- 更新 PhoneBot 项目页与论文来源归档：当前官网已列 CAD、BOM、STEP、打印件和装配教程，硬件资料采用 CC BY-NC 4.0。
- 同步修订 PhoneBot wiki：区分可下载的非商业硬件资料与仍未找到公开入口的软件源码；未创建虚构的代码仓条目。

## [2026-10-21] lint | 全量 wiki lint 清零：修复 3 条信息型预警（AA-Robotics 回链评测基准选型闭环枢纽；FIBER / 微型机器人导航 paper 页补「与其他工作对比」章节）

## [2026-10-21] ingest | sources/papers/visionhope_arxiv_2609_33325.md — 补充机构归属、主表数值与消融，修正评测表述为可核对数字

## [2026-10-16] ingest | sources/repos/anthropics-claude-for-legal.md — 归档官方法律工作流插件仓库并整理 Claude for Legal 的模块化 agent 配置、开源状态与适用边界

## [2026-10-16] ingest | sources/papers/trackeverything_arxiv_2609_30222.md — 归档论文、项目页与占位仓，新增实体页并关联状态估计；官方代码待发布

## [2026-10-16] ingest | sources/papers/gator_arxiv_2610_11215.md — 收录 GATOR 论文与官方项目页，整理图像重建、agent 精修及代码待发布状态

## [2026-10-16] ingest | WorldOlympiad、WorldArena 1.0 Track-1 与 EWMBench 三项具身世界模型基准归档和选型页更新

## [2026-10-16] ingest | 归档 Contact-UAN 执行器残差 Sim2Real 与 λ₀ 全身人体数据 VLA；核查官方代码、模型与数据开放边界

## [2026-10-21] ingest | sources/papers/fiber_arxiv_2610_24675.md — 补充 FIBER 真机分任务成功率（6/10、4/10、2/10）与仿真分任务口径；核查「可控导航」：论文与项目页仅有 waypoint steering 定性演示，无导航基准，已在实体页标注边界

## [2026-10-21] ingest | sources/repos/jj-vcs-jujutsu.md — 收录 Jujutsu 版本控制工具及其 Git 互通、change 工作流与适用边界

## [2026-10-21] structural | 增加香港大学 OpenDriveLab 与 MMLab 实验室技术路线，复用已有节点并注明官方归属、排序锚点与开源边界；首页拆分独立实验室卡片并支持展开。

## [2026-10-21] ingest | sources/papers/minute_scale_microrobot_navigation_arxiv_2608_00854.md — 收录 Nature Machine Intelligence 微型机器人导航分钟级训练论文、官方代码与 wiki 提炼

## [2026-10-19] structural | docs/checklists/graph-perf-audit-2026-10-19.md — 图谱 2D/3D 渲染性能与 Bug 复审：实测基线（6,216 节点 / 53,740 边）、14 条已确认 Bug、P0/P1/P2 分批方案

## [2026-10-21] ingest | sources/repos/rpl-amazon-far.md — 补入 RPL 官方训练、蒸馏与 sim2sim 代码并更新既有论文实体

## [2026-10-21] ingest | sources/papers/fiber_arxiv_2610_24675.md — 收录 FIBER 论文、项目页与官方仓库，并补充移动操作潜空间规划路线

## [2026-10-21] ingest | PrismWAM

- 归档论文、项目页与官方代码入口，并核对代码、训练配方及基准权重已公开。
- 新增 PrismWAM 实体页，补充 WAM 概念页：记录语义目标与深度物理 rollout 解耦的设计、报告指标及复现边界。

## [2026-10-20] lint | 全量 lint 修复：18 条预警清零（1 陈旧页 + 17 信息型）—— visual-servoing 回链 OptiSight；9 页补评测/感知栈枢纽回链；5 个 dataset 补模态维度；JOGA 补 venue；MemT 补评测/对比区块；Evo-DexBench 改写绝对化措辞

## [2026-10-20] ingest | SKY-001 工程世界模型（官方博客与产品页）

- 归档 Skyfall AI 官方博客与产品页，并新增 SKY-001 实体页。
- 记录发布方披露的 CAD/BIM 流程和基准结果；标明缺少公开复现代码及官方页面未列源码/权重入口。

## [2026-10-20] ingest | 收录 Reason Network 编码代理协作通信引擎

- 新增 `sources/repos/reason-network.md`，记录官方仓库与产品页、点对点通信和 MCP 协作能力。
- 核实开源边界：通信引擎采用 MIT OR Apache-2.0；Reason CLI 免费但不开源。
- 作为开发代理协作工具归档，暂不新建机器人知识实体页。

## [2026-10-20] ingest | 收录 AutonomousVehicleControlBeginnersGuide 自动驾驶控制学习仓库

- 新增 `sources/repos/autonomous-vehicle-control-beginners-guide.md`，记录官方仓库、MIT 许可、Python 环境和算法示例范围。
- 作为自动驾驶算法学习资源归档，暂不新建 wiki 实体页。

## [2026-10-20] ingest | KODA 官方代码与资产分阶段发布更新

- 复核官方仓库并更新论文、项目页、代码仓库来源与 KODA 实体页：M1/M2 运动学和机器人机载视角管线已有安装与 CLI；M4 部分开放示例 MCAP 和 tracking checkpoint。
- 记录 M3 控制器动作标签/对齐数据导出与 M5 策略训练仍待发布，说明当前代码可运行范围和第三方资产许可边界。

## [2026-10-20] ingest | 收录 CrossFly 空中移动交互基准（arXiv:2610.18381）

- 归档论文与项目页，覆盖异构环境中的长程飞行、IRR 交互就绪度指标及 CrossFly-Agent。
- 项目页提供 ModelScope 数据集；代码标注 “Code soon”，当前未列公开仓库。

## [2026-10-20] ingest | HFHPE — 归档 MIT 头部姿态估计代码与权重，记录姿态网络与检测器的延迟边界。

## [2026-10-20] ingest | 深蓝具身智能「LeCun 2026 JEPA 11 项研究」：新建 6 个项目详情节点（Causal-JEPA、Value-Guided JEPA 规划、Music-JEPA、Rectified LpJEPA、SkyJEPA、TDV）；LeWM / LeVJEPA / When Does LeJEPA / LpWM / Patch Policy 复用既有节点。

## [2026-10-20] ingest | 收录 DeepSeek 推理与协议适配仓库

- 归档 DeepInfer 与 deepseek-recipe 官方仓库及 README。
- 两者源码均公开；许可证分别为 Apache-2.0 与 MIT。
- 记录机器人落地边界：DeepInfer 支持的硬件偏向数据中心加速器；deepseek-recipe 负责协议与提示词适配，不含模型推理及机器人控制。

## [2026-10-20] ingest | R²-WAM 视觉表征实证

- 收录 arXiv:2610.13024 论文摘录与项目页核查，并新增单一 WAM 论文实体。
- 项目页仓库是静态网站源码；截至核查日未找到官方研究代码、权重或数据集链接。
- 关键结论：优先强通用视觉表征与通道维联合预测，多源特征按任务相关增量选择。

## [2026-10-20] ingest | MemT-1.0 — 归档原生记忆模型论文、官方项目与实现，并整理参数化记忆评测边界及软件 / 权重许可范围。

## [2026-10-20] ingest | OptiSight：语义推理与几何控制的具身导航

- 收录 arXiv:2608.23354 与 CSIC-UPC 官方技术报告记录 IRI-TR-26-02。
- 作者代码仓库公开；地图和模型权重需另行准备，汇总成功率由人工统计且暂无自动评测脚本。
- 关联视觉–语言导航与视觉伺服，归档论文、官方记录和仓库来源。

## [2026-10-20] ingest | 三项论文/项目收录：Rubik Skills、Blind Precise Assembly/mjfit、ActiveAlign；补充三篇实体页及互链，记录公开实现边界

## [2026-10-20] ingest | sources/sites/unitree-uniuni.md — 归档宇树 UniUni 桌面集群编舞系统，新增实体页并关联人形多机协调；闭源软件

## [2026-10-20] ingest | 公众号综述「全球85个具身智能数据集全景」：新增44个数据集节点（dataset-*），复用41个已有节点，总览页 embodied-dataset-panorama-2026-10 给出逐项去重映射

## [2026-10-20] ingest | 归档 Haoyang Weng 的可 steer 物理交互札记并补充 VLA/S0 接口分析

## [2026-10-20] ingest | ACE-Brain-0.6 媒体披露归档，并更新 ACE-Brain 统一实体页的版本与开源状态

## [2026-10-20] ingest | sources/papers/fcts_failure_guided_curriculum_tree_search_arxiv_2610_22619.md + sources/papers/replanmem_vla_arxiv_2610_22556.md + RAX skills/data — FCTS、RAX 在线探索与 RePlanMem 视觉记忆资料归档

## [2026-10-19] lint | 全量 wiki lint：补齐 7 个 paper 实体「与其他工作对比」区块（LIBERO-PeRM 另将「基准设计」改为「方法与基准设计」），3 个 benchmark/evaluation 页回链评测基准选型闭环枢纽；信息型预警 10 → 0

## [2026-10-20] ingest | 补充 MAMMA 官方新闻稿：奖项候选、iPhone 多视角、精度与工作流耗时口径

## [2026-10-19] ingest | Commander–Executor 协作操作 — 归档 arXiv:2610.14911；仿真人形角色方法，代码待发布。

## [2026-10-19] ingest | 收录 Tidy Design 开源产品画布及 MCP 设计协作工作流

- 官方 GitHub 仓库标注 Apache-2.0，README 确认 agent 可经 MCP 读取、创建和编辑可编辑设计，并支持自托管。
- 官网入口由官方 README 确认；官网页面本次未能直接读取，因此功能总结以仓库 README 为准。
- 新增项目实体并关联 MCP 与 coding agent 知识页；机器人面板用途注明为工程推论，未表述成已验证集成。

## [2026-10-19] ingest | sources/papers/onewm_arxiv_2610_21095.md — 归档 OneWM 论文、项目页和代码，并提炼因果联合训练与互惠评测结果

## [2026-10-19] ingest | sources/papers/*_2610_22*.md — 收录六篇具身机器人论文，并核查项目页、代码开放状态和评测边界；T2-z 已在 main 完整归档

## [2026-10-19] ingest | π0.7 训练图解补充来源

- 收录 Kuan Yi Wang 的第三方 π0.7 训练前向流程图解，并关联既有论文摘录与方法页。
- 教程 URL 在截图中被截断，已标注待核实；截图未提供代码或权重入口。

## [2026-10-19] ingest | 收录 book-to-skill 资料编译工具

- 归档公开仓库，并新增实体页说明其将技术书籍与文档集整理为按需加载的 Agent Skill。
- 补充 LLM Wiki 关联，区分资料级 skill 编译与跨来源知识库维护。

## [2026-10-19] lint | 修复 Wiki Lint「可能过期」12 页（updated 2026-04-21 满 180 天）：仅刷新 updated 日期，未做内容复核

## [2026-10-18] structural | roadmap 巡检补入 28 个新知识节点中的 23 个（VLA/WAM/遥操作/重定向/接触操作/评测/动作生成/架构/硬件/电机/数据 11 条纵深路线）；HiRE、Visuomotor Pruning、DA-AO、26 篇技术地图暂无合适阶段未收

## [2026-10-18] lint | 全量 lint 修复：4 个陈旧页 review、paper-* 三段式/结论补齐、评测基准枢纽回链、RAX 陈旧声明改写

## [2026-10-18] ingest | OpenAI GPT-6 软件维护实践入库

- 归档 GPT-6 Astra / GPT-6.1 Sol 在代码清理、过期实验审查与 CI 优化中的证据驱动工作流。
- 将成本/发现数与 CI 时延数据作为 OpenAI 内部案例补充到 agentic coding 概念页，并明确机器人场景属于方法映射。

## [2026-10-18] ingest | sources/papers/theophys_eval_arxiv_2610_16733.md — 归档自主理论物理 agent 基准与评测边界

## [2026-10-18] ingest | 收录 arXiv:2610.15073 刚-软混合链路人形机器人论文
- 新增论文来源归档与单一论文实体页，保留仿真与真机实验的证据边界。
- 更新 Whole-Body Control 页的反向引用。

## [2026-10-18] ingest | EdenLive 官方资料、代码、权重均可用

- 归档项目页、技术报告入口、官方仓库与公开权重；代码和模型权重已开放。
- 归纳实时生成渲染的系统分工，记录 RTX 5090 约 24.7 FPS 的测量边界及 FLUX 工作台许可范围。
- 新增项目实体，并从生成式世界模型方法页建立交叉引用。

## [2026-10-18] ingest | sources/sites/opencfs.md — 收录 openCFS 官方站点、源码与配套论文；核实 MIT 开源状态，建立统一多物理场仿真实体并关联 FEMM。

## [2026-10-18] ingest | sources/blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md — 具身智能小站 10-18 期 26 篇：新建 19 + 复用 7 个独立论文节点，阅读坐标 embodied-station-2026-10-18-26-papers-technology-map；开源结论 16 已开源 / 3 待发布 / 7 项目页未列 GitHub

## [2026-10-18] ingest | 归档 Alex Wang 联想记忆教程与公开实验代码，补充 Transformer 的参数/上下文记忆区别及容量假设边界

## [2026-10-18] ingest | REK Simulator 与 iOS TestFlight

- 归档 REK 官方模拟器页与 Steam 产品页，更新既有 REK 实体节点。
- 官方称模拟器使用与真实机器人相同 AI，并将模拟赛与线下 pilot 资格连接；更新记录镜像提到 iPhone/iPad TestFlight，标注日期为 10 月 9 日。
- 尚无一手依据确认 10 月 17 日首发或完整 AI 在手机本地运行。

## [2026-10-17] lint | 全量 wiki lint（full git history）：修复 20 个陈旧页（locomotion→DA-AO、NavDP→DevGRU、EmbodiedEvo→RAX、IL/RL/online-vs-offline-rl→robomimic，另 14 页补 CoSMO/CTP/ForceWAM/PACER/R²-WAM/RoboFQE/SANDO/SEAR/TRACE 回链）与 45 条信息型预警（29 篇 paper 补「与其他工作对比」、评测/感知栈枢纽回链 10 页、DA-AO venue、NavAble 重定向就绪度、VT-Bridge 措辞、缺页概念停用词 beta/choice/pip）

## [2026-10-17] ingest | 收录 DA-AO 外骨骼步态相位估计论文

- 归档 T-RO 论文 DOI 与摘要级方法、跨地形实验指标。
- 建立 DA-AO 实体页并关联步态生成与运动控制；全文细节和代码状态待核验。

## [2026-10-17] ingest | WorldScape R²SI — 记录技能修订与 WSP 更新的报道型系统节点

- 新增 R²SI wiki 实体页与媒体报道、Manifold AI 官网、RoboDojo 官方榜单核查归档。
- 将案例互链到 WorldScape Policy 2.0、RoboDojo 与 RSI 纵深路线。
- 开放状态：未找到 R²SI 独立论文、代码或权重；RoboDojo 第一名及步数结果按报道口径记录，未独立核实。

## [2026-10-17] ingest | sources/repos/awesome-embodied-self-improvement.md — 归档机器人 / Physical AI 部署后自我改进清单，并将 L1–L4 可执行能力判据接入 RSI 概念页

- **范围：** 45 项具身自改进工作、3 个 benchmark、4 篇综述；区分 episode 内反馈、持久能力优化、技能扩张和改进机制演化。
- **开源判断：** 清单仓库 MIT、内容开放；它是文献导航，不是机器人自改进运行时；逐篇代码状态需另行核验。
- **关键页：** 更新递归自改进概念与通用 Awesome RSI 的机器人读者入口。

## [2026-10-17] ingest | RoboJuDo / FRoM-W1

- 归档 RoboJuDo 官方部署仓库与 FRoM-W1 官方项目页，核对代码、论文、数据开放状态及策略发布边界。
- 更新既有 FRoM-W1 实体，按单一项目节点收录 RoboJuDo 部署流程；记录 RoboJuDo LICENSE 与 README 许可链接标签不一致，供复用前核查。

## [2026-10-17] ingest | 归档 Alex Wa 的前沿模型训练方法综述，保留其训练流水线与后训练参考；暂不单独升格为 wiki

- 来源：`sources/blogs/alex_wa_frontier_model_training_methodologies_2026-01-31.md`
- 处理：归纳架构、稳定性、数据配比、预训练与后训练要点；说明向机器人基础模型迁移时仍需验证具身数据和闭环部署约束。

## [2026-10-17] ingest | Claude Code 团队代码评审：补充 Agentic Coding 软件工程页

- 归档 Anthropic 工程博客，提炼 PR 前验证、AI 与安全评审、风险分级人工审批、渐进发布和评审收敛机制。
- 更新既有 Agentic Coding 知识页；文章报告的回滚率变化属于 Anthropic 内部结果，已标明外推边界。

## [2026-10-17] ingest | RAX 在线探索机器人 agent

- 核对官方项目页、论文 PDF 与代码仓；代码已公开，仓库以分支提供仿真、Franka 和 Rainbow RB-Y1 实现，根目录许可证待核实。
- 建立统一 RAX wiki 实体，整理冻结 coding agent、在线探索、程序编译、技能复用和评测口径，并链接 ASENA / EmbodiedEvo 相关路线。

## [2026-10-17] ingest | sources/blogs/wechat_embodied_station_robot_36_papers_2026-10-17.md — 具身智能小站 36 篇论文逐篇建独立详情节点（30 新建、6 沿用），含开源核查与技术地图索引

## [2026-10-17] ingest | SANDO 动态未知环境安全规划

- 归档 arXiv v3、作者项目页与 MIT ACL 官方代码；代码已开源，README 提供 ROS 2 Humble 仿真 / 硬件入口，并依赖 Gurobi WLS Academic license。
- 新增 SANDO 统一实体页，记录 STSFC、热图 A*、硬约束 MIQP、变量消元与真机评测；明确形式安全的假设及真机估计 / 跟踪误差边界。
- 作者主页称 T-RO 有条件接收，当前仓库 README 写作已接收；保留来源表述差异。

## [2026-10-17] ingest | Infinigen V2 程序化 3D 场景生成

- 官方代码与 V2 文档已核对并归档；仓库公开，BSD-3-Clause。
- 新增项目来源页与实体页，关联场景生成、仿真接入和机器人数据用途；截图中的「一分钟」未找到官方基准，标记为未核实。

## [2026-10-17] ingest | sources/sites/motphys.md — 归档 Motphys 官网首页/关于/研究/版本日志；新增公司入口并加入公司路线（复用 Motrix / GS-Playground / UniLab 三节点）

## [2026-10-16] lint | 全量 wiki lint（full git history）：修复 25 个陈旧页（ppo 补 RoboFFT、π0-policy 复核 kai0，另 23 页补新 paper 实体回链）与 10 条信息型预警（paper 三段式/结论/缩写速查、评测与执行器枢纽回链）

## [2026-10-17] ingest | 收录 KODA、T2-z 与 VLA 因果分析论文和官方实现

## [2026-10-17] ingest | Agentic Design Patterns — 归档出版社书目与 GitHub 配套材料，标注书籍版权及仓库许可核查边界，并关联 agent 评测与 coding-agent 工程知识页

## [2026-10-17] ingest | Whole-Body Dexterity — 收录论文与项目页，补充全身接触重定位知识节点

- 项目页未列代码或数据下载入口；新增论文、项目页来源归档及 WBD 实体页，并接入 Loco-Manipulation 任务页。

## [2026-10-17] ingest | Intern-Decision（arXiv:2610.22131）— 归档论文、项目页与代码；记录权重与训练数据的开放边界

## [2026-10-16] ingest | vt-bridge | 预训练 VLA 的触觉残差适配

- 意图：整理 VT-Bridge 的方法结构、真机结果与复现入口，补充视触觉残差适应路线。
- 开源结论：项目页 GitHub 仅为静态网站源码；研究训练/推理代码与权重无公开链接，Code 按钮标为 “Coming soon”。
- 关键页：新增论文来源、项目页/源码归档与 VT-Bridge wiki 实体；更新视触觉融合概念页建立入口。

## [2026-10-16] ingest | Open-BENI — 归档 BENI 训练、恢复与浏览器仿真项目

- 官方仓库包含 BENI 机器人模型、训练源码、策略文件及 MuJoCo + ONNX 浏览器游戏；MIT 许可。Hugging Face Space 为在线部署。
- 归档项目页与代码仓库，并整理 wiki 实体，关联 mjlab、PPO 与 locomotion 学习路径。

## [2026-10-16] ingest | RoboHarn-Evo | 记录可演化物理知识操作框架

- **意图：** 收录 arXiv:2609.37583 及其项目页和官方实现，作为长时程机器人操作中“从交互经验更新外置知识”的案例。
- **开源结论：** 项目页链接的官方仓库含 Python agent、RMBench / LIBERO-PRO 应用与运行脚本，Apache-2.0；模拟器、模型与许可资产另行配置。
- **关键页面：** 统一实体页讲解 HPK 双层知识、内外循环、不同评测协议与代码复现入口；并链接 VLA 和 manipulation 页面。

## [2026-10-16] ingest | Humanoid Rickshaw Pulling 来源档案与实体页更新至 arXiv v2

- 核对 arXiv v2 的机构、版本与当前公开入口；论文仍未列独立代码仓库。
- 将来源档案及 wiki 页面 HTML 链接升级至 v2，保留 115 kg loaded rickshaw mass 的口径说明。

## [2026-10-16] structural | docs/lab-roadmaps.json 新增 CMU LeCAR Lab 实验室路线（10 个节点，成立年份按 Guanya Shi 2023 秋加入 CMU RI 计），首页实验室路线入口与计数同步

## [2026-10-16] ingest | MoDE-VLA 与 IMCopilot 论文及项目页核查

- 来源：arXiv:2603.08122 与官方项目页。
- 开源结论：项目页未列代码、权重或数据下载链接，截至核查日未确认开源。
- 更新：补全已有 MoDE-VLA 策展实体页，并增加论文摘录与项目页归档。

## [2026-10-16] ingest | WorldOracle — 归档论文、项目页和评测代码，补充动作条件世界模型的配对真机评测入口

- **开源结论：** 项目页链接官方评测代码（Apache-2.0）与 Hugging Face 数据入口；README 提供评测脚本和运行说明。
- **关键判断：** WorldOracle 对照同一起点、同一低层动作条件下的预测与真机执行；其 SV/MV 指标与 EWMBench / WorldScore 互补，汇总分不可直接换算。

## [2026-10-16] ingest | 收录 VLA 神经表征几何论文（arXiv:2610.20840）

- 新增 `sources/papers/vla_neural_geometry_arxiv_2610_20840.md`，记录论文问题、方法、摘要中的几何发现与 steering 结论。
- 截至入库日，arXiv 条目和作者主页未列项目页或代码仓库；七类概念清单及详细实验设置待精读正文补齐。

## [2026-10-16] ingest | HumanoidTTT：入口证书驱动的人形完整动作复用

- 收录北京大学 HumanoidTTT 论文、项目页与官方代码；代码和初始策略权重公开，应用运行时仍需接入 OMG、HoloMotion、动作资产及执行验证循环，仓库根目录未见许可文件。
- Wiki 记录 200 个测试入口 0 次不安全接受、固定高复用请求轨迹端到端 16.4 倍加速，并注明证书冻结与测试范围等边界。

## [2026-10-16] ingest | minisql-moonbit-vs-rust | 比较 AI agent 使用 MoonBit 与 Rust 的实际成本

- 意图：记录 Agent-Friendly by Design 预印本、实验设计与可复核的逐试验数据。
- 开源结论：官方仓库公开源码、harness、测试、运行资料与结果；复跑需要 Claude Code / 模型访问，根目录未见 LICENSE。
- 关键页：[MoonBit/Rust 实验实体](../wiki/entities/paper-agent-friendly-by-design.md)、[Agentic Coding 软件工程基础](../wiki/concepts/agentic-coding-software-fundamentals.md)。

## [2026-10-16] ingest | GroundAnything — 并行扩散视觉 grounding

- 归档论文、官方项目页与 GitHub 仓库，并核实 DLM、自回归 VLM 权重及代码已公开。
- 新建 GroundAnything 实体页，记录解码策略、benchmark 结果、推理入口和部署边界。
- 与 GroundingPI（arXiv:2609.39601）明确区分并互链；二者共享作者团队，但项目入口与研究重点不同。

## [2026-10-16] ingest | sources/sites/rhoda-ai.md — 归档 Rhoda AI 官网研究页/新闻稿/团队页（3 篇研究博文：DVA、网络视频预训练 scaling、触觉+DVA）；确认未开源；新增公司入口与 3 个研究节点并加入公司路线

## [2026-10-16] ingest | SolidCraft — 收录浏览器端参数化 CAD 工具

项目页与公开源码已核实并互链。新增 SolidCraft 来源归档与工具实体页，并从 FreeCAD、OpenSCAD 页增加 CAD 工具对照入口；官方项目当前标注为 early，曲面、工程图、CAM 与仿真仍在规划。

## [2026-10-16] ingest | Kinematic Vision AI 高尔夫挥杆分析工具

- 收录公开仓库 `anhtranp/golf`：浏览器内 MediaPipe 姿态跟踪与 Gemini 运动学反馈演示。
- README 声称 MIT，但仓库缺 LICENSE 文件；PGA 基准与杆速指标未见公开验证来源，来源页已标出边界。

## [2026-10-16] ingest | 深辰凯动 AI 羽毛球机器人官方资料

- 归档官方产品页、公司纪录介绍与 Guinness World Records 条目，确认 1,452 次纪录及公开技术资料边界。
- 新建项目实体，并与人形羽毛球研究节点建立对照链接；官方页面未列代码、数据或模型权重。

## [2026-10-16] ingest | RoboSimStudio 多物理与真机案例

- 更新既有项目、仓库与论文来源，并补充 wiki 实体页；没有创建重复节点。
- 核实项目开源状态、Newton / MuJoCo Warp 与 SuperDex FEM 的分工、18 项任务仿真，以及约 56 分钟的双臂擦拭和单项真机闭环案例。
- 将演示结果限定在具体案例，说明不构成普遍仿真精度或 Sim2Real 成功率保证。

## [2026-10-16] ingest | ASENA arXiv:2609.39207 v2 论文核验与详情页深化

- 复用已有 ASENA 实体，不重复建节点；归档 v2 论文与 NVIDIA GEAR 官方项目页。
- 补充 coding-agent robot harness、ASENA-VLN、工作区演进、仿真 / 真机评测和安全边界。
- 官方项目页未列训练 / 推理代码、权重或数据下载入口；没有确认到可运行官方实现。

## [2026-10-15] lint | 全量 wiki lint：修复 10 条信息型预警（paper 三段式×5、dataset 重定向就绪度×1、评测基准枢纽回链×3、DMP 缺 queries 落地×1），复跑 0 问题 0 预警

## [2026-10-16] ingest | OpenCut：归档官网与主仓库，区分可用 classic 版和重写中的 Rust 多端版

- 官方 README 确认 opencut.app 当前运行 classic 版本，主仓库的 Rust 核心与插件、MCP 等能力仍处于重写路线图。
- 仅收录 sources 资料；该通用视频编辑工具不升格为机器人知识 wiki 节点。

## [2026-10-15] ingest | OpenViking — 收录火山引擎开源上下文数据库

- 官方仓库和项目页均已核查，代码公开，许可证为 AGPL-3.0。
- 建立 repo/site 来源归档与统一实体节点；将其定位为 Agent 上下文基础设施，不误列为机器人策略或数据集。

## [2026-10-15] structural | roadmap/motion-control.md — 主路线讲解视频重渲：讲稿勘误、读音与字幕修正、BT.709 编码，时长 1:39:40，更新章节时间点

## [2026-10-15] structural | 将 Amazon FAR 作为 Amazon 内部研究团队加入公司路线，复用七个已有项目节点并说明排序年份依据

## [2026-10-15] ingest | Virtual Humans：归档图宾根大学数字人体课程与公开讲义，注明课程专属开源工具箱尚未核实

## [2026-10-15] ingest | Paper2Agent：归档 Nature 论文、官网与 MIT 仓库，编译论文到 MCP 工具的验证流程和局限

## [2026-10-15] ingest | sources/sites/project_instinct.md — 复核 Project Instinct 官网页面状态与训练/部署代码入口并更新统一实体

## [2026-10-15] ingest | sources/papers/umv_arxiv_2602_22118.md — 归档 UMV 系统设计、状态估计与约束强化学习控制

## [2026-10-15] ingest | LCMS2Yield — 归档论文、官方代码与数据，记录开源边界和研究方法

- 论文与源码统一汇入一个 wiki 实体页，区分论文报告的定量框架与仓库公开的 SMILES→CF 预测入口。
- 官方仓库公开 Apache-2.0 代码及多组数据；此工作面向分析化学与高通量实验，可作为自动化实验系统的分析组件线索。

## [2026-10-15] ingest | Fast Feature Field (F³) 事件预测表示

- 收录 Science Robotics 期刊版论文及 arXiv 预印本，论文、项目页和官方实现合并为单一实体节点。
- 项目页核查确认代码已开放，含预训练推理、训练入口、下游任务与 AOTI 导出；数据集需单独配置。
- 整理事件预测表示、多任务评测及表示计算频率与下游任务频率的区别。

## [2026-10-15] ingest | sources/papers/worldline_arxiv_2609_38059.md — 归档动作驱动视觉仿真论文、项目页和无可运行代码仓；补充生成式世界模型与功能分类交叉链接

## [2026-10-15] ingest | GreenWAM 与 BEHAVIOR Robot Suite：核查官方页面、代码与数据

将 GreenWAM 和 BRS 各自整理为单一项目实体，并记录官方开放资源与复现依赖。GreenWAM 提供四个 benchmark checkpoint 与评测代码；BRS 公开算法、硬件控制和数据入口。Pi API beta 已有来源页和 wiki 实体，核对后复用原记录，未建重复节点。

## [2026-10-15] ingest | EmbodiedEvo：核实项目页、论文与公开代码

将论文、项目页和官方代码归档为同一实体节点。项目页与 arXiv HTML v2 已核对；代码公开，README 说明复现依赖单独配置 RoboDojo / Isaac Sim 资产、GPU 和模型凭据，且项目级许可证尚未选择。评测来自 RoboDojo-Sim，精细接触操作仍是主要短板。

## [2026-10-15] ingest | CodeActionBench — 归档论文、项目页与官方代码，并建立统一基准实体节点

- 开源结论：MIT 代码已公开；完整轨迹、观察图像与视频数据集仍待发布。
- 关键页：单一论文实体整理任务边界、agent 工具闭环、隐藏 verifier、评测读法与复现限制。

## [2026-10-15] ingest | sources/repos/ultimaker-cura.md — 归档 Cura 切片工作流并连接 CAD 与增材制造知识页

- 官方 README 确认其为模型切片应用，列出数百项设置与社区打印配置；主程序许可为 LGPL-3.0。
- 记录打印机、材料与参数的适用边界，并区分独立 CuraEngine 仓库的 AGPL-3.0 许可。

## [2026-10-15] ingest | PAMI：部位锚定的人-物交互动作生成

- 收录 arXiv:2609.38466、项目页和官方仓库，整理部位锚点、逐帧轨迹融合与接触精修方法。
- InterAct 接触召回据论文报告较此前最佳方法高 14.5%；官方实现与权重截至入库日仍待发布。

## [2026-10-15] ingest | 31 篇论文逐篇建立独立 paper-* 详情节点与 sources/papers 归档（2026-10-15），并新增索引页 embodied-papers-2026-10-15-31-digest-map

## [2026-10-15] ingest | 归档微信公众号 31 篇具身论文速览（2026-10-15）：sources/blogs 摘要 + sources/raw 原文；未升格 wiki 实体

## [2026-10-15] ingest | beltromatti/get-it — 归档本地优先的文档学习应用，补记模型数据流、分数边界与官网版本提示差异

## [2026-10-14] lint | 全量 lint 清零：6 条陈旧页面 + 17 条信息型预警全部修复（补回链/缩写速查/对比与结论段/venue 键），并为 floorplan-3d 补 sources 归档使覆盖率达 6009/6009

## [2026-10-15] ingest | sources/repos/motiontemplates-385712.md — 归档 .motion 开源视频模板与项目页；项目页当前 404，暂不升格为机器人 wiki 节点

## [2026-10-15] ingest | sources/repos/kaankiziltug-logo-design-skill.md — 收录跨代理 Logo 设计 Agent Skill，并记录第三方 SVG 商标许可边界

## [2026-10-14] ingest | sources/blogs/wechat_ethercat_opensource_libs_qt_adventure.md — EtherCAT 开源库清单，14 个独立节点（13 新 + SOEM）；无法访问文中仓库，开源状态均标未核查

## [2026-10-15] ingest | LiteReality-Agent arXiv:2610.01863 — 更新既有项目实体并补全论文、交互查看器、仿真导出和开源状态

- 复用现有 LiteReality-Agent 实体，未创建重复节点。
- 论文报告 10 个室内场景、620 个采集视角，并明确几何/感知指标不等于真实物理属性验证。
- 官方仓库为 Apache-2.0；LiteReality Scanner App Store 免费提供。

## [2026-10-14] ingest | sources/papers/qwen_drive_1_0_arxiv_2609_00111.md — 收录 Qwen-Drive-1.0 论文、官方文章和代码，补充端到端自动驾驶地图

- 核查结论：权重及推理 / 评测源码公开；训练代码未随当前官方仓库发布。
- 主要观察：共享 VLM、BEV 感知与 flow-matching 规划统一在一个驾驶模型中；评测按各自协议解读。

## [2026-10-14] structural | 首页各社区互链 Top-10 节点读者化：清理 ingest/独立节点/升格/图谱 hub 等维护者口径（wiki 约 170 页 + 4 个生成器同步），摘要与首段改为读者视角

## [2026-10-14] ingest | sources/papers/ha_liu_2015_multiple_contact_humanoid_fall_planning.md — 归档 IROS 2015 多接触跌倒规划；官方页未列可运行源码，补充 Balance Recovery 关联

## [2026-10-14] ingest | sources/papers/loopwam_arxiv_2610_18873.md、ila_arxiv_2610_18855.md、axis_bench_arxiv_2610_18869.md、accrue_arxiv_2610_18596.md — 新增 4 篇论文知识节点与项目页/代码开放核查；Splat2Act 已在 main 收录

## [2026-10-14] ingest | sources/papers/partigen_arxiv_2610_13373.md — 收录部位感知的语言条件人形动作生成论文，补充扩散运动生成方法页；未发现官方项目页或代码

## [2026-10-14] ingest | 具身智能小站 2026-10-13（34 篇）与 2026-10-14（41 篇）两期公众号盘点：用 fetch_wechat_article.py 抓正文，75 篇论文全部建立独立详情节点（新建 66、复用 9 个已有 arXiv 节点），每篇经 git ls-remote / 项目页核查开源结论；新增两张阅读坐标地图（embodied-station-2026-10-13/14），注册 University of Almería 机构。关键页：paper-streamingwam、paper-mulligan-performance-guided-data-collection、paper-freebar-barrier-guidance-async-vla。

## [2026-10-14] ingest | Ha & Liu 2015 多接触跌倒规划全文归档

- 阅读 Internet Archive 保存的 7 页作者版 PDF，记录方法、仿真与实机指标及限制。
- 归档论文、项目页和官方演示视频；核实作者页 PDF 直链 404、Wayback PDF 可读，项目页未提供代码仓库。
- 新增跨主题 wiki 详情页，并从 SafeFall 节点建立联系，区分经典解析式接触规划与后来的学习式保护控制。

## [2026-10-14] ingest | sources/papers/ha_liu_2015_multiple_contact_humanoid_fall_planning.md — 归档 IROS 2015 多接触跌倒规划；官方页未列可运行源码，补充 Balance Recovery 关联

## [2026-10-14] ingest | sources/papers/loopwam_arxiv_2610_18873.md、ila_arxiv_2610_18855.md、axis_bench_arxiv_2610_18869.md、accrue_arxiv_2610_18596.md — 新增 4 篇论文知识节点与项目页/代码开放核查；Splat2Act 已在 main 收录

## [2026-10-14] ingest | GroundingPI — 从论文、项目页和官方仓库整理视觉 grounding 能力与复现边界

- 核验官方项目公开了 grounding 模型权重、推理/训练/评测代码；权重是视觉 grounding checkpoint，不是直接执行机器人动作的通用策略。
- 详情页采用官方 README 中的 Hugging Face 模型路径，并补入 arXiv 与 Hugging Face 论文页面。

## [2026-10-14] ingest | OpenMuse — 归档 MIT 公开源码，并整理其任务代理架构与机器人上层应用边界

- 开源结论：GitHub 源码已公开，MIT；项目 README 标注 Alpha。
- 关键页：sources/repos/copilotkit-openmuse.md、wiki/entities/copilotkit-openmuse.md；在 Hermes Agent 节点增加关联入口。
- 机器人关联：可参考高层任务代理、浏览器/文件工具与行动审阅；项目不提供 ROS、感知或实时控制。

## [2026-10-14] ingest | Natura-1 与舞肌 Wuji Hand 2 硬件协作补充

- 更新既有 Natura-1 来源归档与实体页，不新建重复项目节点。
- 记录舞肌科技提供灵巧手硬件支持训练与部署，并链接官方产品手册；明确该硬件合作不代表 Natura-1 模型、权重、训练数据或部署代码开放。
- Wuji Description、遥操作和 SDK 页面仅作为舞肌公开硬件生态参考，未将其误认作 Natura-1 已确认使用的软件栈。

## [2026-10-14] ingest | DirtyMoCap — 无序光学 marker 到 SMPL-H 的鲁棒重建

- 收录 arXiv:2609.19927 及 SIGGRAPH Asia 2026 官方项目页信息，梳理代理锚点、滑窗跟踪、可微 Gauss–Newton 和评测边界。
- 官方仓库目前仍为待发布占位、HKMALA-Motion 数据 Coming soon；已在归档和实体页注明不可复现/下载状态，并接入动作重定向知识链。

## [2026-10-14] ingest | CodeGraph — 归档官方仓库与项目页，整理本地静态代码图、MCP 查询、增量同步及对机器人软件仓库的适用边界

## [2026-10-14] ingest | DSLC 多机器人隐蔽执行器攻击下的分布式安全学习控制

- 收录 arXiv:2609.06896 与 IEEE Transactions on Robotics Early Access 论文；归纳 attacker–actor–critic + DMPC 方法、1000 机器人仿真、轮式机器人实测及适用边界。
- 截至入库日未找到作者项目页或公开代码仓库，标注为待复核，而非假定已可复现。

## [2026-10-14] ingest | arXiv:2608.17038 — 收录 DEM 全局 A* 与 LiDAR 局部修正的 UGV 导航论文，注明仅 Gazebo 仿真与未发现公开代码

## [2026-10-14] ingest | Chronos Policy — 刷新项目页论文入口与当前代码开放状态

## [2026-10-14] ingest | sources/blogs/anthropic_redesigning_source_control_2026-10-13.md — 归档 Anthropic Scion 内部版本控制案例，并补充 agentic coding 软件工程基础设施瓶颈

## [2026-10-13] lint | 全量 lint 清零：10 页补「与其他工作对比」/「结论」与知识链枢纽回链（13 条信息型预警）；6 个 stale 页复核（sim-vs-real-eval-gap 补 Fidelity Gap 证据，FluxVLA 引用页仅作者单位变更，更新 updated）

## [2026-10-14] ingest | Atomic Machines / Matter Compiler — 归档官方制造系统资料、伯克利研讨会摘要与 PrimeSwitch 证据边界

## [2026-10-14] ingest | LiteNWM arXiv:2610.12368 — latent multi-horizon visual navigation evaluator; no official project/code link found; added paper source and entity plus generative-world-models cross-reference.

## [2026-10-13] ingest | Physical Intelligence Pi API beta — 归档官方服务公告与申请页，整理托管微调/推理工作流和公开状态；独立实体链接到 π₀ / π₀.₅

## [2026-10-14] ingest | Dynamic Creatures Hugo — 收录 Hugo 原型与 SnowJay 官方资料

- 归档官方 LinkedIn 公告、SnowJay 平台页与公司新闻页；官方披露 43 个执行器及目标访客场景。
- 官方入口未列公开代码或数据集；页面注明 Hugo 实时情境理解与响应仍在开发，避免把平台级能力误写成已验证的本体能力。
- 新增统一项目实体并关联 Boston Dynamics 页面。

## [2026-10-14] ingest | Gemini 4 Argon 机器人官方博客

- 归档 Google Developers 官方博客，并建立独立实体页，避免与 Gemini Robotics 2 混为一个模型版本。
- 博客展示涵盖机器人控制辅助、Real2Sim、hindsight 标注与仿真数据改进；代码、权重和数据集未随文提供，实验数值按官方自报记录。

## [2026-10-13] ingest | Parkour MJLab：PHP 社区实现与 G1 / Go2 跑酷训练

核对 yhx1203/parkour_mjlab 的训练、回放与 MuJoCo Sim2Sim 入口，并与 PHP 官方仓库区分；归档其未声明许可证及截图提及权重但下载位置未核实的边界。

## [2026-10-13] structural | media/site-demo.gif — 按最新图谱重录 README 演示 GIF（5947 节点）

- **触发：** 用户要求更新首页 GIF
- **脚本：** [`scripts/record_readme_demo.cjs`](scripts/record_readme_demo.cjs)（82 frames / 1.99 MB；图谱 **5947** 节点 / **51954** 边）
- **流程：** `make export graph` → `docs/` 本地 `http.server 8765` → 重录并写回 `media/site-demo.gif`
- **引用：** [`README.md`](README.md)「在线演示」仍指向 `media/site-demo.gif`

## [2026-10-13] ingest | sources/papers/geomvla_arxiv_2609_13812.md — GeomVLA论文/项目页、3D几何一致VLA方法、基准与开源链接核查；建立实体页并关联VLA总览

## [2026-10-13] ingest | EGalahad sim2real 文档、排行榜与 GRIT 策略节点

核实文档站、评测协议与官方代码开放边界；将 15 个榜单策略条目映射到 10 个独立项目详情，补齐 GRIT 策略节点并与另一 GRIT 机载集成仓库区分。

## [2026-10-13] ingest | viola | VioLA：人类数据驱动通用人形控制

- 论文：arXiv:2610.12435；官方项目页核查代码仍为 coming soon。
- 写入 / 更新：更新既有 VioLA 实体 paper-viola-human-data-control.md，补充论文和项目页归档。
- 关联：SONIC motion tracking 页面加入 VioLA 交叉引用；保留原有前沿论文动态来源。
- 要点：以共享 body/hand latent 同时接收人类和机器人演示监督；记录主要结果、zero-shot 口径、人类数据配比及接触反馈 / 单机评测限制。

## [2026-10-13] ingest | sources/sites/naturewill-natura-1.md — 归档 Nature Will 灵巧手基础模型研究预览，区分闭源数据规模主张与可复现证据

## [2026-10-13] ingest | grit-deploy-onboard-g1 | G1 机载部署项目

- 来源：截图中项目链接 https://github.com/KurokoWeidong/GRIT_Deploy_Onboard_G1
- 归档：sources/repos/grit-deploy-onboard-g1.md
- 写入：wiki/entities/grit-deploy-onboard-g1.md；补充 axellwppr-motion-tracking 交叉引用
- 要点：将 Loco、VR Tracking 与 NPZ 轨迹播放作为多模式机载集成样例记录，区分部署代码与上游策略 / 训练实现，并补充 README 中的部署安全提示。

## [2026-10-13] ingest | sources/repos/pentagi.md — 收录多智能体安全测试平台并记录授权范围、容器隔离与外部服务边界

## [2026-10-13] ingest | SDD / TDD / BDD / ATDD — 补齐方法定义并说明分层组合关系

- 归档 GitHub Spec Kit、SDD 预印本、Kent Beck 对 canonical TDD 的说明、Dan North 的 BDD 原文，以及 Cucumber / Agile Alliance 实践文档。
- 区分外层规格追溯、跨角色行为/验收共识、开发者组件级测试反馈环；附机器人仿真/HIL/真机验证读法。
- 已存在的 SDD 课程与 TDD skill 保留为工具/实践节点，不重复建项目页。

## [2026-10-13] ingest | GenIA — 归档论文、项目页与官方代码，并整理测试时输入对齐和开源复现边界

- 官方项目页确认 GenIA 在冻结 SAM 3D 先验上做推理时对齐；动态版本依赖 ActionMesh。
- 官方仓库已开放，安装要求 CUDA 12.8 / NVIDIA GPU，SAM 3D Objects 权重需先接受许可。
- 单独对比生成式补全、输入可见区域指标和每场景推理成本，关联 SAM 3D 策展入口。

## [2026-10-12] lint | 全量 wiki lint：清零 39 条信息型预警（25 页回链评测基准/感知栈枢纽、8 页补论文三段式、3 页补数据集维度、VaDar 补 venue、Actor/export 归入缺页概念停用表）

## [2026-10-12] ingest | CADCraft：原生 Rust CAD 与 MCP 自动化入口

- 收录 CADCraft 官方代码与产品页，沉淀可运行 CAD 应用实体，并记录 DXF / DWG 读写边界与版本显示差异。
- 与 CAD Skills 建立入链，区分 CAD 应用本体与 Agent 技能工具链。
- 核查项目为 MIT OR Apache-2.0 开源，GUI / CLI / MCP / WASM 共用 Rust 绘图引擎。

## [2026-10-12] ingest | RSI-Anything 与 JEPA-Anything 论文、代码和 PhAI Labs 官网

- 分别建立 RSI-Anything 与 JEPA-Anything 论文实体，归档论文、官方代码仓及项目页；新增 PhAI Labs 机构节点和机构标签。
- 关联递归自我改进与生成式世界模型方法页，区分任务级可验证闭环、潜在预测模型与完全自主后继模型的证据边界。
- RSI-Anything 与 JEPA-Anything 均为独立项目；前者以 JEPA-Anything 为预测模型基础。代码开放范围和项目方实验限制已分别记录。

## [2026-10-12] ingest | sources/papers/agentgarten_arxiv_2610_12374.md — 收录显式 code worlds、神经渲染器、playbook 学习轮次及部分开源边界，并连接生成式世界模型与 Code-as-World

## [2026-10-12] ingest | sources/papers/csf_contextual_safety_filtering_arxiv_2610_12467.md — CSF 论文、项目页与 Apache-2.0 仓库归档；新增 G1 场景语义安全过滤实体并关联 CBF / Safety Filter。

- 核心结论：冻结文本动作生成器，以场景门控、生成器自身安全/危险参考和采样内 CBF-QP 调整动作；执行中场景变化时改写未执行轨迹。
- 核对范围：arXiv 全文、官方项目页、官方 GitHub README；明确区分论文真机链路与仓库浏览器 demo 的复现条件。

## [2026-10-12] ingest | EgoHumanoid-V2：核对论文全文、项目页与部分开放源码，补全现有实体及移动操作路线交叉链接

## [2026-10-12] ingest | sources/blogs/waytoagi_feishu_top_ai_papers_2026-09-07-09-13.md — WaytoAGI 飞书周报（转载 DAIR.AI）10 篇论文：新建 9 个独立详情节点、复用 1 个已有节点（递归自我改进综述）；偏 LLM 智能体而非机器人，各节点写明与机器人学习的关系；飞书正文用 scripts/fetch_feishu_doc.py 获取。

## [2026-10-12] ingest | CLOTHO（TOG/SIGGRAPH Asia 2026）：归档 DOI 与项目页，整理松散惯性服装 IMU canonicalization 方法、跨服装评测与待核实开源状态。

## [2026-10-12] ingest | sources/blogs/wechat_embodied_station_34_papers_spatial_grounding_2026-10-12.md — 具身智能小站 34 篇盘点：34/34 新建独立论文详情节点（逐篇读 arXiv 摘要/HTML 并核查项目页/仓库），汇总页 embodied-station-34-papers-spatial-grounding-technology-map；公众号抓取用 scripts/fetch_wechat_article.py。

## [2026-10-12] ingest | sources/blogs/frontier_humanoid_quadruped_2026_10_05_10_09.md 与 frontier_manipulation_2026_10_05_10_09.md — 10.5-10.9 两篇周报共 73 篇不同论文：新建 60 个独立论文详情节点（逐篇读 arXiv 摘要/HTML 并核查项目页开源状态），13 个已有节点补链；5 篇在两期重复；另新增 scripts/fetch_wechat_article.py 作为公众号抓取兜底。

## [2026-10-12] ingest | DiffWAM 导航预测表示接地

收录 DiffWAM 统一论文实体和官方资源归档，整理早期预测表示、Grid-Motion、Latent2Pose、同 rollout 几何蒸馏与 FastDreamer 异步交接。区分标准/Flash 计算预算、候选轨迹与安全控制、端点 SR 与真机完成、archived Thor 延迟与当前解码器；记录项目页差异及未发现公开实现的边界。补流程与系统时序 Mermaid，从 Fast-WAM 建立入链。

## [2026-10-12] ingest | DiscoRL：核查 Nature 论文、官方项目页及 JAX 源码，补充双层优化、运行时序与机器人迁移边界

## [2026-10-12] ingest | sources/repos/xbotics-embodied-ai-handbook.md — Xbotics 21 讲实践教程：新建课程实体页并接入 LeRobot；项目页无，代码同仓 MIT，L1-16 较完整、L17/18/21 占位

## [2026-10-12] ingest | sources/sites/research-beingbeyond-com.md — 公司路线新增智在无界 BeingBeyond（26 节点，新建公司总览）

## [2026-10-12] ingest | MEgoVista 公制手头重建

收录 MEgoVista 统一论文节点，整理三鱼眼基线尺度、佩戴者手归属、离线整段优化与独立动捕审核。强调对齐后 4.29 mm 指标、野外定性演示和在线吞吐边界；硬件官网未发现专属代码/权重/评测数据入口，不虚构运行脚本。补充 Mermaid 并连接重定向知识链。

## [2026-10-12] ingest | Being-M0.7 新版复核

更新已有 Being-M0.7 节点，补入 arXiv:2610.11283 三阶段训练、门控视觉上下文、22D/63D/66D 接口与部署时序。区分 7 月项目介绍与 10 月论文，修正旧真机领先结论及 50 Hz 专家推理误读。论文和演示公开，未发现项目专属代码/权重/处理后语料入口；同步 WAM 路线与交叉页，未新增重复实体。

## [2026-10-12] ingest | Reflex：视觉驱动人形接球与代码待发布状态

- 收录官方论文、项目页和代码仓来源；统一编译到一个项目实体，保留仿真与真机指标边界。
- 三阶段方法以特权状态控制、延迟状态推断和 RGB-D 时序 grounding 逐层接入；官方仓当前未发布可运行训练 / 评测 / 真机代码。
- 从人形 RL 身体系统栈的视觉闭环层加入入链。

## [2026-10-12] ingest | World Model Consultation Study | 世界模型与直接评分对照

- Q: 世界模型在什么任务条件下值得参与控制决策？
- A: 预测式控制在部分导航条件中有优势，但 Push-T 中固定容量/视野的预测器未稳定胜过直接评分器；证据限于模拟。
- 写回：新增论文与代码归档、实体节点，并链接至 Model-Based RL 方法页。

## [2026-10-12] ingest | Human-Like Redundancy Resolution | 双臂遥操作冗余解析

- Q: 如何在双臂遥操作中保留腕部任务优先级并获得更类人的肘姿态？
- A: 自适应肘偏好通过层级速度 IK 的腕任务零空间起作用；公开演示仓面向 JAKA K1 + Viser，无仿真后端或硬件输出链路。
- 写回：新增论文、项目页、代码归档和实体节点，并链接至 Teleoperation 任务页。

## [2026-10-12] ingest | VaDar | 长时程 RL 价值分解

- Q: 如何保留子目标监督，同时避免策略只追逐当前阶段？
- A: VaDar 用共享 critic 分别建模各子目标价值，并以未来阶段价值训练部署时不显式接收子目标的策略；官方页面未提供算法实现仓。
- 写回：新增论文与项目页来源归档、实体节点及 HRL 入口链接。

## [2026-10-11] structural | roadmap 巡检补入 RoboDuet / VioLA / CoDance / DreamTrue / AD-E2E-JEPA / VTInstructor / OpenLET-Roban2

## [2026-10-11] ingest | Math To Manim — 归档开源生成管线并关联 Manim

- 上游公开源码与 MIT 许可证；当前主线使用 Astra 分阶段生成、Manim 渲染和证据审阅。
- 将其作为 Manim 之上的独立教学内容管线记录；注明模型调用、渲染依赖及非仿真/非形式证明边界。

## [2026-10-11] lint | 全量 wiki lint：修复 4 条信息型预警（TEMPO 对比区块、BenchHub 时效措辞、RealCompanion/UltraText-Bench 回链评测基准枢纽）

## [2026-10-11] ingest | wheresryan22/anatomy — 收录把技术原理转成交互式等轴测机械插图的 Claude Skill

- 归档官方项目页与 MIT 仓库，补充生成流程和几何审计方式。
- 明确 SVG/HTML 解释图、WebGL 与 beta 3D 动作的能力边界；它不替代制造级 CAD 或机器人仿真模型。
- 在文字生成 CAD 与 3D 网格工作台知识页增加交叉引用。

## [2026-10-11] ingest | LT-Mem 官方奖项与量化结果更新

- 核对 LT-Mem 项目页、论文与 APRL 官方新闻/论文列表；项目页仍显示 finalist，实验室公告已更新为 IROS 2026 Best Paper Award Winner。
- 更新既有 LT-Mem 独立节点及来源档案，补充 LT-VQA 量化表和代码仍为 TBD 的核查状态。

## [2026-10-11] ingest | wiki/entities/voicestudio.md — 补充 VoiceStudio 短音频克隆与视频配音流程

- 复核官方 README：补充 3 秒参考音频零样本克隆与视频转写、翻译、合成、时间对齐后导出 MP4 的流程。
- 明确 646 种语言是 OmniVoice 引擎口径，实际覆盖与效果因引擎和语言而异；沿用现有实体节点，避免重复收录。

## [2026-10-11] ingest | CoDance | 单视频到人形双手顺应舞伴交互

- 意图：收录 CoDance 论文及其官方项目页、代码实现和数据下载入口，集中到一个项目实体节点。
- 开源结论：训练、评测、顺应增强与 G1 部署代码已公开，数据集和检查点由官方 README 指向 Hugging Face；根项目 Apache-2.0，但第三方组件另有许可。
- 关键页：[CoDance 实体](../wiki/entities/paper-codance.md)；论文、项目页和仓库来源分别归档，关联人-人示范的人形交互论文页。

## [2026-10-11] ingest | AD-E2E-JEPA | JEPA 自动驾驶世界模型论文与官方仓库

- 意图：将 arXiv:2609.34085 与作者指定 GitHub 归并为一个项目实体，并连接 JEPA 世界模型与端到端驾驶主题。
- 开源结论：论文称 source code available；但官方仓库截至 2026-10-11 仅有 README 与示意图，README 预计 2026-10-19 前发布源码，当前不可复现。
- 方法与评测：归档 DINOv3、SIGReg 压缩投影、NAVSIMv2 目标条件零样本规划，并区分 256 与 8192 候选的性能/时延工作点。
- 关联：建立论文、仓库来源归档与唯一实体页，并从 LeWM 页面添加反向链接。

## [2026-10-11] ingest | Tao 的 Math 2.0 讲稿 — 梳理 AI 时代的数学研究目标、共同体与解释价值

- 将讲稿与作者配套说明作为一手来源，区分原文主张与机器人领域映射。
- 补充 Math 2.0 知识页，并连接 The Bitter Lesson 方法论页。

## [2026-10-11] ingest | OpenLET LET-HumanMotion-Roban2 — 收录申请制动作数据、LejuBFM 运控部署与 Roban 全身 VLA 教程，注明数据规模与模态待核验

## [2026-10-11] ingest | Hugging Face 本周论文 5 篇（独立实体）

- 来源：图片列出的 2026-10-05 至 10-11 榜单；逐篇核对 arXiv。
- 入库：RealCompanion、Cross-Tokenizer OPD、Kandinsky 6.0 Video、UltraText Bench、Self-Retrospection Distillation，各建独立来源与 wiki 实体。
- 开源：Kandinsky 6.0 Video 和 UltraText Bench 有官方代码/项目归档；另三篇在所查 arXiv 页面未找到可核实运行仓库。
- 注：Self-Retrospection Distillation 与已收录 State–Readout Decoupling 虽同缩写，按不同 arXiv 身份分开记录。

## [2026-10-11] ingest | RFPO 一手资料复核 — 补齐官方项目页归档、源码运行时序与身份字段

- **去重：** 复用已有 arXiv:2610.10453 实体；未新增重复项目节点。
- **复核：** 对齐论文、项目页与官方仓库的作者、机构、机器人评测及 Isaac Lab 训练 / 采样步数评测入口。
- **边界：** 仓库为实验复现代码，不包含部署包；论文平台覆盖范围不等于仓库均提供可部署配置。

## [2026-10-10] ingest | arXiv:2609.38170 像素扩散对抗后训练 — 新增论文实体与来源档案，更新 Diffusion Model 交叉链接

## [2026-10-11] ingest | Agnes Harness — 收录开源 FDE agent runtime，区分其同名 MHS 设备协议与 Anthropic/HHMI 标准，并核查 pre-alpha 安全边界

## [2026-10-11] structural | roadmap 巡检：将 10-09 以来新增的 18 个知识节点补入 9 条纵深路线（loco-manipulation / contact / RSI / WAM / eval / VLA / RL-loco / sim2real / real2sim）

## [2026-10-10] lint | 全量 wiki lint：修复 12 个陈旧页面、方法落地/论文元数据/三段式/结论/陈旧声明/评测枢纽回链 11 条信息型预警，lint 归零

## [2026-10-10] ingest | sources/papers/activescale_arxiv_2609_18514.md — 复用ActiveScale节点，核查官方代码、六组权重及任务数据目录；补齐机制、消融和两张Mermaid图，纠正吞吐与遥操作边界

## [2026-10-10] ingest | 遗传算法与遗传编程一手资料及机器人应用

- 归档 Holland、Goldberg、Koza、Hansen/Ostermeier 与 Storn/Price 的原著和论文。
- 新增 GA、GP 方法页，补充 CMA-ES / DE 对照，并链接现有双足 GA 步态案例。

## [2026-10-10] ingest | OpenSCAD 参数化脚本 CAD 工具

- 意图：归档 OpenSCAD 官方网站、文档与源码，建立独立软件实体页。
- 工程定位：保留脚本参数化、CSG、CLI 网格输出等能力，同时明确其与 B-rep/STEP CAD 及 URDF/MJCF 的边界。
- 互链：连接 Text-to-CAD、FreeCAD 与 URDF 页面。

## [2026-10-10] ingest | The Coding Train、p5.js 与 ml5.js

- 意图：归档三项相互关联但职责不同的创意编程与浏览器机器学习资源，供机器人可视化、交互原型和教学演示参考。
- 开源结论：p5.js 官方代码仓采用 LGPL-2.1；The Coding Train 网站代码仓采用 MIT；ml5.js 官网宣称开源并链接实现仓，代码仓 SPDX 许可状态待以仓库许可文件复核。
- 关键页：建立三个独立项目实体，分别对应教学社区、创意编程库与浏览器机器学习库；互相链接并注明非实时机器人控制栈。

## [2026-10-10] ingest | sources/papers/pso_foundations_1995_2012.md — 收录 PSO 奠基论文、惯性权重与收缩理论，核查作者来源 SPSO-2011 源码，新增方法节点及算法/源码 Mermaid 并连接 AutoPSO、CMA-ES 与优化选型

## [2026-10-10] ingest | AlphaStar（Nature 2019 + AlphaStar Unplugged 研究工具）

- 目的：将 Grandmaster 论文、DeepMind 官方资料、PySC2 与后续 AlphaStar 代码发布放在同一项目脉络中。
- 开源核查：部分开源——公开通用架构、离线 BC 工具和环境；README 明确没有发布 online RL training code。
- 关键入口：AlphaStar 独立实体页，连接 self-play 机器人案例与 RL 方法页。

## [2026-10-10] ingest | AlphaZero（Science 2018）

- 目的：整理多棋类自我对弈强化学习的一手论文与官方棋谱入口。
- 开源核查：论文与部分棋谱可公开；未找到官方训练/推理源码或权重。
- 关键入口：独立实体页连接强化学习历史、Runner、自我对弈与既有 MuZero 节点。

## [2026-10-10] ingest | AlphaGo Zero（Nature 2017）

- 目的：收录从规则和自我对弈开始的围棋强化学习工作。
- 开源核查：论文与棋谱公开；官方训练/推理代码和权重未找到。
- 关键入口：独立实体页，关联 self-play Runner、AlphaGo 历史和 MuZero。

## [2026-10-10] ingest | AlphaGo（Nature 2016）

- 目的：将 AlphaGo 的一手论文、DeepMind 官方回顾与复现边界整理为独立详情节点。
- 开源核查：论文和比赛棋谱公开；未找到 DeepMind 官方训练/推理代码或权重，第三方实现单独标识。
- 关键入口：AlphaGo 实体页与深度强化学习游戏里程碑概念页。

## [2026-10-10] ingest | OCLO 无人体动作数据移动操作

- 意图：收录 arXiv:2610.05678，记录在线姿态补全与全身顺应控制的任务证据。
- 开源结论：论文列出官方项目页；当前环境无法读取页面，代码/数据状态待核实，未创建未经证实的仓库入口。
- 关键页：建立 OCLO 独立论文实体，并从 Loco-Manipulation 任务页建立入链。

## [2026-10-10] ingest | ResGAC：GAC + 残差 RL 的人形末端跟踪

- 归档 arXiv:2610.09479 与项目页；截至核查日，项目代码尚未公开。
- 新建论文实体，并接入 loco-manipulation 与 residual policy learning 知识图。

## [2026-10-10] ingest | OccPlanner（arXiv:2608.14160 v2）

- 新增独立论文实体页与 arXiv 来源归档；L3ROcc 作为同一论文内的数据监督流程，不拆分重复节点。
- 更新导航纵深路线 Stage 3，建立详情页入链。
- 仅按 v2（2026-09-17）记录仿真与 Go2 闭环结果；官方代码、权重、数据链接未能从一手资料核实，明确保留该边界。
- 使用标准三反引号 Mermaid 方法流程图；源码运行时序图注明不适用原因。

## [2026-10-10] ingest | MotionSpaceFlow（MSFlow，arXiv:2609.34190）

- 归档论文、官方项目页、LY Corporation 临时开源代码仓与 Hugging Face 权重，并新建单一项目详情节点。
- 补入 diffusion-motion-generation 方法页和动作生成纵深路线；论文比较区分 263D / XYZ 结果，注明 SnapMoGen FID 并非最优及人体动作到机器人部署的边界。

## [2026-10-10] ingest | sources/repos/php_parkour.md — 核实 PHP 官方开源，更新既有实体与项目页归档，补齐训练/ONNX Sim2Sim 时序与五组数据复现边界

## [2026-10-10] ingest | H-JEPA 分层视觉规划世界模型

- 根据截图确认项目为 **H-JEPA: End-to-End Learning of Hierarchical World Models for Visual Planning**，论文 arXiv:2610.06805，官方项目页与代码仓库已核实。
- 新增独立详情节点，覆盖分层 action-conditioned JEPA 训练、SIGReg/IDM、top-down 子目标规划、模拟环境与 DROID 离线结果，并加入 Mermaid 流程图。
- 新增论文、官方 GitHub 仓库、项目主页三份来源档案。
- 与 HWM（arXiv:2604.03208）建立方法对照；明确 DROID 的 Fréchet fidelity 是离线路径指标，不是真机闭环成功率。
- 明确区分另一篇使用相同缩写 H-JEPA 的 Hamiltonian JEPA（arXiv:2609.33497），避免重复或混淆。
- 关联 Model-Based RL、LeWM、HWM 与 WAM 概念页。
- 未修改 Actions、workflow、或派生文件。

## [2026-10-10] ingest | sources/sites/archon-tech.md — 公司路线新增源策未来（Archon Robotics）：新建公司总览与 Whole-Body Intelligence 博文详情，关联 RoboNaldo、EgoHumanoid-V2

## [2026-10-10] ingest | sources/sites/wlrobo-com.md — 公司路线新增西湖机器人：新建公司总览、TITAN O1、傲天 U1 详情，GAE 补官方下载版与开放状态

## [2026-10-10] ingest | sources/sites/lightwheel-blog-index.md — 公司路线新增光轮智能：官网 18 篇博客与新闻稿全部对应详情，新建 4 页、补全 RoboFinals / LW-BenchHub / EgoSuite DevKit / SimReady 资产页

## [2026-10-10] ingest | sources/sites/aetherlabs-blog-index.md — 公司路线新增 Aether AI：CRIS-0 等官网 11 篇博客与 3 条新闻全部对应详情，新建 8 个页面

## [2026-10-10] ingest | Physical AI 云工具链、世界模型与视觉触觉数据

- 复核了 5 个入口：VPP2 已有独立详情页、arXiv/项目页/代码归档与 Mermaid 图，沿用现有条目，不重复创建。
- 新增 4 个独立详情页：AWS Physical AI Toolchain、Odyssey-3、RoboRender、TouchScale；各自维护唯一 `project_id` 与独立来源链接。
- 新增来源归档：AWS 官方仓库、Odyssey 官方发布页、RoboRender 论文与项目页、TouchScale 论文与 Hugging Face 数据卡。
- 每个新详情页包含一句话定义、缩写表、方法/架构说明、Mermaid 流程图、适用边界和来源。
- TouchScale 单独区分论文所述 500 小时全量与当前已公开的 100 小时，并记录数据许可及访问条件。
- 未修改 Actions、workflow、目录索引或派生文件。

## [2026-10-10] ingest | Swinging Door Trending 原始资料与机器人遥测应用

- 新增概念页：`wiki/concepts/swinging-door-trending-compression.md`，介绍 Swinging Door Trending（SDT）的误差走廊、在线折线压缩过程、参数影响、重建误差与机器人遥测应用边界。
- 新增资料归档：Bristol 的 US4669097A 专利、1990 年会议论文书目记录、AVEVA PI 官方压缩文档与官方演讲材料，以及一个 MIT 许可的社区 Python 参考实现。
- 资料边界：1990 年 Bristol 论文目前归档的是可核验的书目记录，未找到可公开访问的原文全文；社区实现仅作可读代码示例，不代表 Bristol 或 AVEVA 官方实现。
- 更新可观测性和具身数据清洗页面，建立相关概念链接，并说明有损遥测压缩应用于非实时归档副本，不能替代数据清洗或原始安全/诊断记录。
- 未修改 Actions、workflow 或派生文件。

## [2026-10-10] ingest | COAP 论文与 RoboDojo 评测关联

- 归档 arXiv:2610.12369，新增 COAP 独立论文实体；区分离线 coding-agent 开发与运行时纯代码控制。
- 记录作者报告的 42 项 RoboDojo 仿真结果（70.24%），并明确截至入库日无官方项目页/公开 COAP 代码入口；补充基准和评测枢纽正文反向链接。

## [2026-10-10] ingest | SAM-V — 收录多视角几何感知分割项目

- 归档 arXiv 论文、代码仓库、checkpoint 与许可边界；区分 Apache-2.0 源码、CC BY-NC 权重和 VGGT 研究许可。
- 新增 SAM-V 实体页与训练/推理流程图，并从 SAM 与 VGGT 相关页面补入反向链接。

## [2026-10-09] lint | 全量 lint 清零：补缩写速查/对比/结论/评测枢纽回链，复核 π0 陈旧页，meta 加入缺页概念停用词

## [2026-10-09] ingest | sources/blogs/simate_beta_robodojo_2026-09.md — 公司路线新增 Simate（硅基伙伴）：新建 Simate-beta 详情，补公司背景与 RoboDojo 成绩

## [2026-10-09] ingest | The Last AI Built by Humans — 收录 RSI 五级自治与 HCI 框架

- 归档 arXiv 论文、项目页及 CC0 文献清单；核查未见论文专属可运行实现。
- 在 RSI 概念页补充论文互链，区分证据框架、汇总指标与真正闭合的递归自改进。

## [2026-10-09] ingest | Artificial Analysis 模型评测平台

- 归档官网、方法说明和数据 API；官网未链接可自托管平台代码仓库。
- 新增独立实体页，界定其通用模型/服务商选型用途与机器人专项评测边界；连接 Agent 评测和具身评测知识链。

## [2026-10-09] ingest | REA 软件逆向调查工具 — 归档官方源码、项目页并提炼 MCP/CLI 工程边界

- 意图：将 morluto/rea 收录为单一工具实体，解释它如何把本机软件分析工具接入编码代理。
- 开源结论：GitHub 主仓库为 MIT 开源；深度原生分析依赖 Hopper、Ghidra 或 IDA 等外部 provider，REA 不等于反编译引擎。
- 关键说明：区分静态分析和运行时采集，并记录本机分析、Agent/model provider 数据边界及目标授权责任。

## [2026-10-09] ingest | sources/sites/generalistai-blog-index.md — 公司路线新增 Generalist AI：核对官网 10 篇博文，新建 GEN-0、GEN-1 详情并补齐公司总览

## [2026-10-09] lint | 修复近期合并的 7 个页面共 10 张 Mermaid 图（9 个非标准围栏与 1 个保留字节点 ID），增加与本站同版本的全库语法 CI 门禁和回归测试。

## [2026-10-09] ingest | Agent Lightning v1.0 发布说明与异步训练机制

- 复用已有 Agent Lightning 主实体与 v1.0 技术报告，不创建重复项目节点。
- 新增 Microsoft Research 发布说明来源归档，补充共置异步 RL、rollout 样本聚合与 Coding Agent 运行文档。
- 官方代码为 MIT 开源；训练仍依赖 verl / vLLM GPU 栈。

## [2026-10-09] ingest | StreamRig 论文与项目分设详情并接入状态估计知识链

- 新增 StreamRig 论文页与独立项目页，分别整理方法 / 评测结论和代码 / 训练 / 权重 / 许可 / 复现成本。
- 归档 arXiv、官方项目页和代码仓库；区分论文项目页列出的四组评测与 README 当前提供的 NCLT、KITTI-360 训练评测入口。
- 将 StreamRig 接入状态估计 hub，并记录其使用 G2G relocalization checkpoint warm-start 的方法链关系。

## [2026-10-09] ingest | Graph-MambaNav（arXiv:2608.13723v1）——归档唯一论文节点并接入 ObjectNav

- 核验 arXiv v1 HTML 与作者主页；论文已接收 IEEE RA-L，计划在 ICRA 2027 展示。
- 归纳目标条件图扫描、逐物体时间 Mamba、AI2-THOR/RoboTHOR 评测与单场景轮式机器人演示；官方入口未提供代码仓库链接。
- 更新 ObjectNav 任务页交叉引用，并登记无锡学院、南京邮电大学机构标签。

## [2026-10-09] ingest | sources/sites/x2robot-blog-research-index.md — 公司路线新增自变量机器人：核对官网 8 篇技术博文，补齐 WALL-WM / WALL-OSS / X-Tokenizer 详情，新建 WALL-OSS-0.5、XRZero-G0

## [2026-10-09] ingest | Cartwheel Comic 单目视频动捕模型

- 收录 Cartwheel Comic 官方模型页、Comic 4.2 发布说明、API/插件状态及 Cartwhl/cartwheel-mcp 开源接入仓库。
- 建立单一实体节点，说明脚部物理发布声明与历史 EMDB 指标口径不同，并区分云端模型和 MIT MCP 客户端。
- 记录 Unreal Engine 5.6 插件可用；Maya、Unity、Blender 官方插件仍显示 Notify me；添加到动捕与人类视频分类页。

## [2026-10-09] ingest | X2Real 论文与项目拆分归档并校正评测指标

- 复核已有 paper-x2real 节点，保留原独立论文页并修正误写的 0.74：arXiv 摘要报告的是仿真与真机评测结果 0.84 线性相关。
- 新增独立 X2Real 项目详情，区分论文贡献、项目工程定位与公开静态资产集；原 Day 6 索引现分别链接论文和项目。
- 归档 arXiv 题录、项目页及 Hugging Face 资产来源；注明静态资产不等于论文提及的近 300 小时轨迹，代码入口与许可证状态需继续核验。

## [2026-10-09] ingest | NVIDIA 前沿代理辅助仿真工作流：归档官方案例并补充 Omniverse 节点

- 文章是平台生态案例汇总，不对应单一代码项目；保持现有 NVIDIA Omniverse 实体节点，不重复建项目节点。
- 归纳代理串联场景/物理/渲染/传感器与人工审阅的迭代流程，并注明官方演示的证据边界。

## [2026-10-09] ingest | sources/sites/xiaomi-robotics-website.md — 核对小米机器人官网 4 篇 Research 博文，补 XR-0 后训练管线开源（2026-04-27）节点与详情

## [2026-10-09] structural | 公司路线新增小米机器人 — 6 个时间轴节点复用既有 wiki，同步首页入口、公司对照矩阵、开放程度与官方来源索引

## [2026-10-09] ingest | AutodidactWAM（arXiv:2610.08119）— 自生成视频到 G1 动作的跨模态自蒸馏

- 归档 Skoltech / MWS R&D Center 的 arXiv 预印本；论文注明投稿 ICRA 2027，未表述为已接收。
- 记录 Cosmos 3 Nano → G1 + BrainCo 手部动作恢复与 DPO+SFT+DTW 后训练，区分抽取器重放与学习策略的真机结果。
- 强调初始 embodiment adaptation 需要混合遥操作数据；试验限于单臂单任务、每对象 / 条件 N=20，并记下伪标签域偏差风险。

## [2026-10-09] ingest | Awesome Open LLMs — 开源模型月度追踪目录

- 收录 liucongg/awesome-open-llms：以 Markdown 月档整理开源模型信息，提供搜索、月份导航、月度重点与截图浏览。
- 记录解析器的条目格式约定，并明确该项目是发现入口而非官方模型卡或标准化评测；README 当前覆盖 2025-06 至 2026-09。

## [2026-10-09] structural | 公司路线新增小鹏机器人与理想汽车具身团队 — 13 个时间轴节点复用既有 wiki，同步首页入口、公司对照矩阵、开放程度与官方来源索引

## [2026-10-09] ingest | ME-Brain-1.0 arXiv v2 与仓库状态复核

- **处理：** 更新现有独立节点，不另建重复论文或项目页；补充 v2 作者、日期、ME-VLM 与 Focus-VLWA 基准结果。
- **来源：** arXiv v2（2026-09-29）及 MachEmbodied 官方 GitHub README。
- **开源结论：** Focus-VLWA 训练/推理代码已发布；预训练权重、ME-VLM、完整框架和仿真/真机集成仍列于 Todo。
- **文件：** `wiki/entities/paper-me-brain-1-0.md`、`sources/papers/me_brain_1_0_arxiv_2609_24271.md`、`sources/repos/me_brain_1_0.md`

## [2026-10-09] ingest | OTRetarget 动态动作证据与控制边界

- 补充 `sources/papers/otretarget_arxiv_2609_36602.md`、`sources/sites/otretarget-project.md` 和现有实体页 `wiki/entities/paper-otretarget.md`。
- 论文图 1 可核实的动态例子是四足爬行中的手部触地、带全身旋转的移动；将具体侧手翻接后空翻支撑切换标为未由论文确认。
- 明确逐帧 IK 生成重定向参考，真机跟踪由下游强化学习策略承担；代码链接复查仍未发现可运行仓库。

## [2026-10-09] lint | 全量 lint：补 5 篇 paper-* 「与其他工作对比」、legoflow/gem-x-cpp 枢纽回链、复核 3 个陈旧页（sonic/wt-umi/unitree-g1）；lint 0 问题 0 信息型预警

## [2026-10-09] ingest | Drawnix

- **意图：** 归档用户给出的 Drawnix 开源白板仓库，并说明其在机器人研究流程图 / 思维导图中的适用边界。
- **开源结论：** 仓库根目录声明 MIT；Web 应用源码开放。自动保存实现为浏览器 IndexedDB / LocalStorage，不能据此假定存在云同步或实时协作。
- **关键内容：** 补充 Mermaid 转可编辑画布的源码路径、Markdown 转思维导图、图像 / JSON 导出及本地运行入口；新增独立项目知识页，并链接 Mermaid.js 节点。

## [2026-10-09] ingest | AI Engineering from Scratch — 接入开源 AI 工程课程并梳理机器人相关入口

- 核验 GitHub MIT 许可与官方课程网站，归纳强化学习、Sim-to-Real、3D 视觉和 Agent 工程相关 lesson。
- 明确 Sim-to-Real 示例是 GridWorld 侧滑参数教学代码，不是真机机器人实验；链接至 RL 与 Sim2Real 知识页。

## [2026-10-08] ingest | FineEnvs — 收录面向 LLM agent 的可复现强化学习环境与训练配方

- 核查上游公开仓库与许可：代码根目录声明 Apache-2.0；RL 环境指南内容目录单独声明 CC-BY-4.0。
- 汇总项目矩阵、Wordle 可运行 rollout、LaTeX OCR 训练闭环与 PortSimEnv 的边界，并链接 RL / Gymnasium 知识节点。

## [2026-10-08] ingest | RSS 2026 HOP：时域最优轨迹规划

- 来源：RSS XXII 在线论文录、RAP Lab 官方项目页与 Python tutorial 仓库。
- 开源结论：项目代码公开，含 HOP-LQR/HOP-DDP 的 toy-system 运行示例；仓库未见 LICENSE 文件，也没有真机接口。
- 关键内容：整理 LFT/Riccati 的 horizon 复用、HOP-DDP 扩展、brute-force/shift-horizon 对照；区分作者测试结果与一般全局最优保证。
- 写入：新增 HOP 论文实体与 paper/site/repo 来源档案；更新轨迹优化、LQR/iLQR 方法页建立互链。

## [2026-10-08] ingest | Bode 图概念与原始资料

- 新增 Bode 图概念页，解释幅频/相频、增益与相位裕度、机器人控制应用及边界，含 Mermaid 图。
- 归档 Bode 1940 年期刊论文、1945 年专著书目和 MathWorks 官方文档；复用库中已存在的 MIT 2.161 Bode 讲义。
- 将新页连入阻尼系统与 PID 控制页面；公众号只作为用户提供的主题线索，正文未读取，不据此写入技术结论。

## [2026-10-08] ingest | Raylib — C99 跨平台图形库

- **意图：** 将截图中的 Raylib 作为独立工具实体收录，整理模块边界、跨平台绘制与机器人可视化用途。
- **开放结论：** 官方仓库公开，zlib/libpng 许可；6.0 增加 CPU 软件渲染与内存帧缓冲平台。
- **关键关联：** 与已有 GenoView-InverseKinematics 节点互链；明确 Raylib 不负责机器人动力学或物理仿真。

## [2026-10-08] ingest | NVIDIA 公司路线补充 Isaac Gym Preview 的 2020-10 时间点；核查官方论坛开放公告、后续博客与论文日期，复用既有详情并同步日期依据。

## [2026-10-08] ingest | FreeSpeed — arXiv:2610.05734

- **意图：** 归档 FreeSpeed 论文与官方项目页，整理其动作块测试时速度控制方法。
- **开放结论：** 论文、项目页已公开；项目页标注 “Code soon”，未提供可运行官方仓库，暂按代码待发布记录。
- **关键页：** 单一论文/项目实体页 wiki/entities/paper-freespeed.md；对照已收录的 SpeedTuning。

## [2026-10-08] lint | 核查公司路线 22 个未注明日期节点：18 个补齐官方发布或版本事件月份，4 个保留空日期并说明边界；同步排序、详情来源与日期证据链接。

## [2026-10-08] ingest | Long-WAM 长上下文世界–动作模型

- 新增一个独立论文/项目节点；论文、Long-WAM 代码子目录与模型资源合并归档。
- 标注其位于 NVlabs/LongLive monorepo 的 Long-WAM/ 子树，并链接至 WAM 概念页。
- 归档 arXiv、官方代码和项目页，补充方法/运行 Mermaid 图及性能和复现边界。

## [2026-10-08] ingest | PredActor 评测代码与 checkpoint release 更新

- 更新已有 PredActor 实体，不新建重复节点；核对官方 GitHub、Hugging Face checkpoint 仓与项目页团队入口。
- 将开源状态更正为部分开放：MuJoCo evaluator 与 PDP051/MotionCLIP 工件已发布；训练、数据采集、DAgger 和真机部署仍未公开。
- 补充可运行评测时序图，并同步修正路线、locomotion、diffusion policy 和 whole-body control 中的过期状态描述。

## [2026-10-08] ingest | sources/repos/localai-gem-x-cpp.md — 收录 GEM-X 的 C++/GGML 本地实现

- 新增独立实体，归纳仓库实现、模型资产、实时/离线流程、许可与验证范围。
- 与上游 GEM-X、GENMO/GEM、SOMA-X 和 SONIC 建立双向关联；未将社区移植混记为新论文或机器人控制器。

## [2026-10-08] ingest | 星动纪元公司路线：核对成立、论文与产品时间，复用 VPP/Humanoid-Gym 与三项工程实体，区分研究开放和 ERA-42 产品边界

## [2026-10-08] ingest | sources/sites/rai-institute.md — 新增 RAI Institute 公司路线，按 2022 年成立排序，复用六个项目详情，区分并行研究、公开事件和开放资产。

## [2026-10-08] ingest | GitDiagram 仓库与在线应用

- 归档 GitDiagram 源码、站点及独立实体页；记录生成、校验、Mermaid 渲染和 MCP 路径。
- 区分模型生成的架构解释与完整静态/运行时分析，并标注私有代码信任边界。

## [2026-10-08] ingest | Gitingest 仓库与在线入口

- 收录 Gitingest 的仓库、托管站点与独立实体页；区分整库文本打包与 RAG 检索。
- 关联 RAG 上下文构建，并记录本地/托管使用的代码隐私边界。

## [2026-10-08] structural | 修复 CAE/CFD 总览及 53 个项目组成的孤岛：从系统工程、仿真保真度、CAD 综述和电机设计工作流建立双向入口及代表项目链接；连通分量从 54 节点孤岛接入全图。

## [2026-10-08] ingest | BRACE | 收录力与地形感知的人形全身跟踪

- 资料：arXiv:2610.07052、Multiply Labs 项目页与公开网站仓 README。
- 开源结论：发现的 GitHub 仓为静态项目页源码；算法训练/部署代码尚待官方链接。
- 提炼：平地参考的地形/wrench 适配、双教师 DAgger、proprioceptive flow-matching student。

## [2026-10-08] ingest | QF3 | 收录过滤 Q 梯度的离策略 flow policy RL

- 资料：arXiv:2610.08789 与项目页。
- 提炼：记录按动作维度过滤 critic 梯度、humanoid locomotion / tracking 与 manipulation fine-tuning；官方代码入口暂未核实。

## [2026-10-08] ingest | ACG-WAM | 收录动作条件几何监督 WAM

- 资料：arXiv:2610.06965、项目页、官方训练仓与 HF checkpoint。
- 开源结论：Apache-2.0 代码与 RoboTwin 40k 权重已公开；演示数据、teacher cache 与真机栈需分别核查。
- 提炼：ACG-JEPA 多时域几何目标、共享视觉表征训练、部署时剥离辅助分支。

## [2026-10-08] ingest | Exploy — 归档 RAI 官方文章、文档与源码，整理 RL 策略计算管线一体化导出及硬件接口边界

- 开源核查：Exporter、C++/ROS Controller、Isaac Lab/MjLab adapter 与 MIT 许可证均可在官方仓库查看。
- 关键边界：ONNX 输出对齐需用 evaluator 实测；传感器、驱动、总线和安全逻辑仍由机器人集成方负责；官方博客未提供统一定量 benchmark。
- 归纳到独立实体页，并从 ONNX 格式页建立回链。

## [2026-10-08] ingest | uni-vlat | 更新已有 canonical 节点

- 复核：Uni-VLaT 已在 `paper-uni-vlat.md` 有独立详情节点；不新建重复页。
- 更新：根据 arXiv v2 与项目页补齐 8 位作者、四校机构、2026-09-30 版本信息及当前项目页入口。
- 开源状态：项目页仍写 “Code coming soon”，本次未发现官方代码仓库链接；未新增虚构的代码归档。
- 来源：更新既有 arXiv 与项目页 source archive；修正官网 “coming soon” 与已发布 v2 的版本不一致说明。

## [2026-10-07] lint | 全量 lint 清零：补 4 篇 paper 对比章节、复核 3 处陈旧声明、FlashDexRetarget 回链 HOI-Retarget/SPIDER；陈旧声明巡检豁免「滞后于论文最新版本 / 核对最新 LICENSE」核对提示

## [2026-10-07] ingest | ibelick-ui-skills | 收录 UI Skills 设计工程技能目录及 CLI / MCP 入口

- 资料：公开仓库 https://github.com/ibelick/ui-skills 与官网 https://www.ui-skills.com/
- 判断：新增一个独立工具项目实体；站点和仓库作为同一项目的两份来源归档。
- 变更：新增 wiki 项目详情、仓库 / 官网 source 归档，并从 find-skills 建立 wiki 入链。
- 开源：仓库公开且 package metadata 标注 MIT；第三方技能的授权仍按各自上游核对。
- 验证：后续检查 Wiki Lint、CI smoke、Search & Export Quality、Tests。

## [2026-10-07] ingest | InterMimicGen 自演进人形移动操作模仿

- 收录 arXiv:2610.06850 与官方项目页，建立唯一 InterMimicGen 论文/项目实体；未拆分重复 project/repo 节点。
- 归纳异质 HOI 数据融合、接触保持重定向、PPO 通用跟踪器和迭代仿真筛选飞轮；记录跨配置重定向、5 轮约 142–151 倍验证动作增长及真机演示。
- 项目页未链接该方法的代码、数据或模型；没有把相关 MimicGen / HumanoidMimicGen 仓库归为本项目实现。
- 从 WBT 知识链与 HumanoidMimicGen 详情补充关联入口。

## [2026-10-07] ingest | PixelUMM 统一像素空间图像视频模型

- 归档 arXiv 论文、官方项目页、GitHub 实现与 Hugging Face checkpoint，并集中编译为一个独立项目实体。
- 记录推理代码与 toy training 已公开、源码与权重许可不同，以及未验证机器人闭环/实机控制的边界。
- 从视觉感知骨干知识链建立 wiki 内链；未修改工作流、派生目录或统计文件。

## [2026-10-07] ingest | 具身智能从入门到精通 Day 5–6

新增世界模型与决策、工程与实机部署两篇独立导读。复用已有论文和项目详情，为缺少节点的工作补建独立实体；项目源码状态未核实处标记为待核。

## [2026-10-07] ingest | FlashDexRetarget 灵巧手多参考动作重定向

- 归档 arXiv v2、官方项目页与 GitHub 仓库状态；项目页源码已公开，但训练/推理算法代码待发布。
- 记录 XHand 50 动作基准 90% SR_SPIDER / 86% 严格 SR_MT、29 GPU-hours；CHORD 对照 46% / 0%、2,847 GPU-hours。
- 新建项目实体并保留方法机制、基准口径、真实机器人回放与复现限制。

## [2026-10-07] ingest | 微信公众号 CAE/CFD Agent Skills 清单 — 核对 53 个公开仓库并建立独立详情节点

- 将文章摘要归档到 `sources/blogs/`，每个可唯一映射的公开 GitHub 项目分别建立 `sources/repos/` 资料和 `wiki/entities/` 项目节点。
- 同名仓库按 owner/repo 身份拆分；HPC-Skills 中的子技能保持在父仓库节点内。已核对仓库公开状态与可见许可证元数据。
- 文章提及 `ANSYS-automatic-wwj` 但未给出唯一仓库链接，本次不猜测身份，标记待核实。

## [2026-10-07] ingest | Workhorse — 人类示范驱动的人形全身移动操作

- 归档作者演示公告与截图可确认的目标、署名和 G1 演示；没有把尚未公开的论文方法、评测或开源状态写成事实。
- 新建独立项目详情节点，并从 loco-manipulation 数据入口页建立链接。

## [2026-10-06] lint | 全量 wiki lint：修复 9 条信息型预警（paper 三段式/结论/缩写速查、评测基准枢纽回链），0 问题 0 预警

## [2026-10-06] structural | 补充并发 ingest 去重 — 同步 main 后将新增 CARI4D 论文与官方实现合并；FORM-HOI 保留独立数据节点并修正图表围栏，累计合并 63 组、迁移 65 个旧详情 ID

## [2026-10-06] structural | 项目节点去重 — 合并 62 组论文、项目与官方源码节点，迁移 64 个历史详情入口；复核保留不同研究的 NEXUS；新增 Wiki Lint 项目身份与新增实体 CI 门禁

## [2026-10-06] ingest | 核对宇树八节点公司路线时间：区分 G1 产品发布、工程仓代码历史起点及 UnifoLM 分阶段开源，更新 WLA 9 月后续开放状态。

## [2026-10-06] ingest | sources/repos/party_os.md — 增补 RoboParty 公司路线并复用八个既有详情；区分整机、并行运控与上层规划，补核 INTACT 上游已发布而组织镜像仍为预览，飞书目录及部分正文已核查，新增章节未确认

## [2026-10-06] ingest | dynibo — 更新已有实体与一手资料归档

- 复核 GitHub 主仓、v0.5.1 Release、中文用户指南、Cargo 元数据与 PyPI；沿用已有 Dynibo 节点，修正从 v0.1.0 后演进的固定/浮动基座、质量矩阵、ABA 正动力学和多语言接口。
- 同步 Pinocchio、ABA/RNEA、ssik 与快速上手页面中关于 Dynibo 能力范围的交叉说明；记录上游性能数字的任务对象与版本边界。
- 源码和 bindings 已公开（MIT）；仓库示例 URDF 保留第三方许可。

## [2026-10-06] ingest | JEPA, from language models to world models — VideoDB Labs

- 归档 VideoDB Labs 长文并新增独立实体页，按 latent representation、action-conditioned dynamics、hierarchical planning 分层整理其论点。
- 为文章引用但库中尚无独立详情页的论文建立 5 个非重复实体及一手 arXiv 来源归档：Semantic Tube Prediction、When Does LeJEPA Learn a World Model?、HWM、From Tokens to Thoughts、VL-JEPA。
- 复用已有 LeJEPA、LeWorldModel、V-JEPA 2.1 节点并建立交叉引用。
- 保留原文观点与论文结果的边界；标注特定数据集/机器人/规划设置和不可外推限制。
- 所有 Mermaid 使用标准三反引号围栏与基础 flowchart 语法。

## [2026-10-06] ingest | sources/repos/redot-engine.md — Redot Engine 开源游戏引擎

- 收录 Redot 官方项目入口、文档与源码仓库，建立独立实体页。
- 归纳 Godot 分支关系、MIT 许可、平台导出范围和 26.2 LTS / 26.3 RC 版本状态。
- 关联 Unity、Unreal 引擎条目，并区分游戏运行时能力与机器人控制级仿真验证。

## [2026-10-06] ingest | UniWAM — 统一物理推理、未来视觉与动作生成

- 新增 UniWAM 独立论文/项目详情页、arXiv 来源和官方代码仓库归档，并更新 WAM 概念页。
- 归档论文提出的三专家 MoT、VQA/人类视频/机器人示教分工、history-conditioned flow matching，以及 LIBERO、LIBERO-Plus、RoboTwin 与 Piper 真机评测。
- 标注论文 Table 1 与正文 robot-data 小时数存在差异；标注官方 release 暂不覆盖真机推理。
- 文件与关联：`wiki/entities/paper-uniwam-unified-world-action-model.md`、`sources/papers/uniwam_arxiv_2610_02054.md`、`sources/repos/uniwam.md`。

## [2026-10-06] ingest | sources/repos/metrox-eth-moss.md — MOSS 实机与录制仿真边界归档

- 收录 MOSS 项目页、主仓库与 MOSS × Jev 仿真仓的一手资料，并建立实体页。
- 明确标注：实体原型已行驶但尚未自主拾取；软件/固件公开，CAD 待正式发布；网页 Jev 演示为录制回放。
- 为 DimOS 实体页补充指向 MOSS 的关联，保持知识图互链。

## [2026-10-06] ingest | PointWAM 3D 世界动作建模与灵巧操作

- 收录 3D scene/hand trajectory WAM 论文与项目页；新建独立实体并在 WAM 概念页添加回链。
- 官方项目页标注 Code soon，未见仓库或权重下载，明确源码运行时序图不适用；不修改 GitHub Actions 或派生文件。

## [2026-10-06] ingest | KungfuAthleteBot 论文更新与数据/代码状态核查

- 将既有实体从 arXiv:2602.13656 更新至 2610.03388，不新增重复项目节点。
- 核实 GitHub 有可运行处理与训练代码、checkpoint；HF 当前卡显示 992 条 Apache-2.0 元数据，官网和论文附录 E 尚留 848 口径，已在来源和 Wiki 中并列注明。
- 新论文报告 G1 统一跟踪与跌倒恢复；不修改 GitHub Actions 或派生文件。

## [2026-10-06] ingest | FastOPD 少步 VLA on-policy 蒸馏

- 新增 FastOPD 论文与项目页来源，以及独立论文实体；方法聚焦单点教师速度监督 + 区间自一致性。
- 项目页未列代码/权重/数据下载链接，明确标为待发布；不修改 GitHub Actions 或派生文件。

## [2026-10-05] ingest | Humanoid Badminton 动态球拍技能

- 更新已有 arXiv:2609.31840 独立项目节点，避免与同名 Multi-Stage RL 论文混淆。
- 新增论文与项目页来源归档；记录官方项目页未提供可运行代码、公开数据或权重链接。
- 补充三阶段潜技能训练、仿真/实机指标、动捕依赖和落点控制局限。

## [2026-10-05] ingest | ForeDrive — 收录规划相关潜世界模型论文

- 归档 arXiv v2 摘录并新增独立论文实体页，补齐生成式世界模型总览的反向入口。
- 记录 NAVSIM v1/v2 指标口径、非对称优化、TAB 与 H20 推理成本。
- 截至入库日未找到官方项目页、代码仓库、权重或独立数据包；不添加虚构复现步骤。

## [2026-10-05] ingest | Robot Native Engine — 收录 Rust 机器人仿真核心并标出验证边界

- 开源结论：官方仓库与代码公开，许可为 Apache-2.0 或 MIT；未确认独立权重或数据集页面。
- 核心归纳：固定步进、headless 与渲染分层、后端抽象、传感器与 replay 数据流；G1 / Go2 演示按仿真证据描述。
- 来源：官方仓库 README、架构、Examples、Robot Workbench、G1 locomotion、Go2 door 与 Livox Mid-360 文档。
- 互链：从 MuJoCo 实体页加入 RNE 入口；未修改 GitHub Actions 配置。

## [2026-10-05] ingest | 移动操作

- 新增 Day 4 独立导读与来源索引，涵盖 30 项工作。
- 补建 CEER2、DexWeave、CompliantWBC、VisForce 详情；Praxis 单独标注来源标题不一致。
- 标准 Mermaid flowchart；未提交派生 catalog / log 文件。

## [2026-10-05] ingest | 动作跟踪与全身控制

- 新增 Day 3 独立导读与来源索引，涵盖 29 项工作。
- 补建 H2O、StableMimic 详情；复用库内已有节点。
- 标准 Mermaid flowchart；未提交派生 catalog / log 文件。

## [2026-10-05] ingest | 公司路线资料补齐：核查官方项目与源码，区分 AstraBrain WAM/WBC、Redwood 策略/世界模型及各代 Helix；补实薄弱详情，按代码、权重、数据分别记录开放边界，并补全已归档工程入口。

## [2026-10-05] ingest | cortical-knowledge-word-concept-learning | 单独收录论文与 OSF 来源，记录神经贝叶斯模型证据、LLM 对照和代码核验边界

## [2026-10-05] ingest | Spingi

- 收录 ceccode/spingi 的官方代码与 Viewer 项目页，整理 LLM 计划校验、技能执行、安全监控和 episode 回放链路。
- 明确当前 MuJoCo G1 是运动学原型、尚无真机 adapter；关联 Unitree G1、MuJoCo、安全状态机与 LeRobotDataset 页面。

## [2026-10-05] ingest | Human Body And Accessories

- 收录 Ultralytics Platform 的 19 类人体部件与服饰分割数据集；区分可见索引信息、截图信息与未核实的访问/许可边界。
- 新增独立数据集实体页，并从图像分割分类页建立交叉链接。

## [2026-10-04] lint | 全量 wiki lint：复核 PhysMani 开源更新联动的 4 个陈旧页并 bump updated；Blind Grasp Reflex / 果蝇 RNN 补齐「对比」章节，警告与信息型预警清零

## [2026-10-04] structural | roadmap/depth-contact-manipulation.md — 接入 Blind Grasp Reflex 手部本体感觉抓取节点

## [2026-10-04] ingest | 更新 PhysMani 官方 release 与复现入口

- 复核官方 PhysMani 仓库和版本 manifest：项目已从 landing page 更新为 pre-release，开放自身训练/仿真评测代码、PhysMani-Bench 下载和 checkpoint 脚本。
- 同步更新既有论文实体与 repo/paper 来源归档，补充训练、数据、权重、Docker、锁定 submodule 版本及 Mermaid 源码运行时序图。
- 明确记录当前未打包的比较基线、默认 3 GPU 训练要求和 CC BY-NC-SA 4.0 许可。

## [2026-10-04] ingest | Mermaid.js 开源图表工具

- 新增 Mermaid 主库、官方文档站与独立 wiki 项目节点；Live Editor 和 CLI 收在同一个 Mermaid 工具链节点中。
- 从 Diagram Design 正文加入 Mermaid 节点入口，确保知识图有入链。
- 官方仓库均标注 MIT；Wiki 归纳文本语法、渲染方式、版本兼容与目标平台验证。

## [2026-10-04] ingest | Prior Evolution and Task Alignment for Aerial Grasping

- 新增论文来源归档与独立 wiki 论文节点；互链 Manipulation 与 Trajectory Optimization。
- 方法摘要：CEM 按部署优化器的求解结果进化轨迹先验；Execution-Aware Critic 以接触、抬升、完成监督，并以冻结能量形成可微抓取代价。
- 截至入库日未发现官方代码、数据或独立项目页；未拆出独立代码节点，定量结果保留为摘要可确认的定性结论。

## [2026-10-04] ingest | ChunkTrust 项目复核与单节点更新

- 对现有 ChunkTrust 详情页进行补充，论文、代码、项目页和模型卡仍收录在同一个节点。
- 依据官方 Hugging Face 模型卡补充结果记录与论文稿件存在未解决差异、未进行全套 benchmark 新重跑，以及 QHA head 需匹配冻结 base policy 等复现边界。
- 更新 GitHub 与项目页来源归档的核查日期；未创建重复论文或代码节点。

## [2026-10-04] ingest | Rerun — 补齐官方仓库、站点文档与机器人数据流

- **开源结论：** Rerun SDK 与 Viewer 已开源；Rerun Hub 为商业产品。
- **关键资料：** 核对官方 README、架构文档、ROS 2 / MCAP / URDF 文档与机器人数据预处理示例。
- **知识沉淀：** 更新既有 Rerun 实体页，明确离线导入、实时桥接、时间轴查询和训练数据准备的边界。

## [2026-10-04] ingest | OSIRIS 全球 OSINT 地图与多源地理数据图层

- 归档官方 GitHub 仓库、官网/API 文档与隐私说明，并新建 OSIRIS 实体页。
- 记录 MapLibre/WebGL + API feeds 的地图架构参考；区分全球公开情报可视化与机器人 SLAM/定位。
- 标出海事 README 与 API 文档对船舶实时性的描述差异，避免将静态参考层描述为已验证实时追踪。

## [2026-10-04] ingest | KingKongRobotics/jumper — 六足机器人训练至部署链路

- 核对公开仓库、官方产品页、硬件文档、项目指南、训练教程、仿真设计和部署说明；训练/仿真/导出代码已公开，第三方许可证由 NOTICE 和各上游许可约束。
- 新增仓库与产品页来源档案，提炼 Jumper 实体页，展开 tripod 速度跟踪、jump 参考残差、dance 轨迹评分、双 MuJoCo 后端、ONNX/layout 校验和 bundle/FSM 路径。
- 对照 Inspect Robots 详情页与 markdown 渲染器，发现页面使用 ```mermaid 围栏而渲染器只识别标准三反引号围栏；同步修复两张图的围栏，并让新页使用可渲染格式。
- 明确 Jumper 的 RKNN 板端推理和 1 kHz 电机总线闭环尚未验证，不把脚本打包能力当作真机验证。

## [2026-10-04] ingest | Inspect Robots — 补齐 G1 × VLA 评测契约与部署边界

- 更新现有 Inspect Robots wiki 节点，展开 Task/Scene、Policy、Embodiment、兼容性预检、闭环执行、评分与 EvalLog 数据流，并加入主 Mermaid 图。
- 单独归档 G1/GR00T 适配器；明确其 16-D 绝对关节位置契约、arm-sdk 接入、预检边界，以及 locomotion controller 仍负责平衡、腿和腰的范围限制。
- 核对主仓 2026-10-03 提交、G1 adapter README 与官方 Concepts/Scoring/Logging 文档；未把接口 shape 检查写成真机安全或全身控制验证。

## [2026-10-03] lint | 全量 wiki lint：修复 11 个陈旧页面 + 10 条信息型预警（LIBERO/λ₀ 补对比与结论、缩写速查、评测/感知栈枢纽回链）

## [2026-10-03] lint | recent-ingest-mermaid | 补齐近期入库节点图示

- 范围：main 在 2026-09-27 至 2026-10-03 首次新增的 238 个 wiki 页面。
- 发现：其中 101 页缺少 Mermaid 代码块；覆盖论文、工具、概念、文章索引与对比页。
- 修复：逐页按已有机制补流程、闭环、分支或阅读结构图；保留已存在的图示。
- 源码：按官方 README 为 US-VLA、Spiderbot、PPTBench 增加运行时序图和入口归档；区分概念主图、实际代码入口与未发布实现。

## [2026-10-03] ingest | ViLoMan arXiv 元数据补录与开放状态复核

- 复用已存在的 arXiv:2609.19340 唯一论文节点，补齐作者与机构信息；不创建重复节点。
- 复核项目页：补充材料仍匿名，公开了训练细节与演示，但没有代码仓库或数据下载入口。
- 同步论文来源归档、项目页归档与 wiki 结论，记录截至 2026-10-03 的开源边界。

## [2026-10-03] ingest | openai/codex — 归档本地编码代理 CLI、Rust 实现与 TypeScript SDK；注明推理服务 / 权重不随源码开源

## [2026-10-03] ingest | Discrete Forcing（arXiv:2609.39526）— 离散 token 引导连续 action refinement；LIBERO 训练代码已开放，模型权重 Coming Soon，RoboTwin 代码待发布

## [2026-10-03] structural | 滚动入场动效扩展到全站（图谱页除外）：列表页按行 / 按日、详情与路线页正文按段落入场（正文区只淡入不上移，保持 TOC 与 hash 落点）；有 prefers-reduced-motion 降级

## [2026-10-03] ingest | EmbodiedSWE：coding agent 求解与示范生成

- 来源：官方项目页、Apache-2.0 GitHub 仓库、arXiv:2609.27308 与 Hugging Face 仿真资产卡。
- 归档 benchmark、agent harness、EmbodiedSWE-Gen、Sim-to-Real 结果和第三方资产许可边界；建立独立论文实体页，并与 agentic coding 相关节点互链。

## [2026-10-03] ingest | sprite-gen 2D 精灵图生成工具

- 来源：<https://github.com/aldegad/sprite-gen>，Apache-2.0 开源 Python CLI / Codex 与 Claude skill。
- 归档：记录图集生成、透明帧处理、视频循环与文档入口；标注其与机器人仿真及控制的边界，暂不升格为 wiki 知识页。

## [2026-10-03] ingest | EgoAlign — 归档人类示范适配、因果状态重建与 G1 长程移动操作评测

- 项目页与 arXiv v2 已核查；官方仓当前只有静态站资源，方法代码与数据待发布。
- 新增论文来源、项目页/代码仓来源归档与论文实体；在 Loco-Manipulation 数据入口加入回链。

## [2026-10-03] ingest | RRSI: Regularized Recursive Self-Improvement

- 归档 arXiv v2、官方项目页与 Google Research 开源仓库；项目页已核查代码及演化浏览器。
- 新增论文详情实体并串联 RSI 概念页与 Stage 4 路线，记录 harness 搜索的迁移、成本与复现边界。
- 官方仓库 Apache-2.0，核心实现可运行；完整论文基准依赖模型服务和独立评测环境。

## [2026-10-03] ingest | Karpathy 的 ASD-STE100 与 LLM 输出可理解性

- 归档机器之心文章及 ASD-STE100 官方入口、开源 Skill 项目；项目代码与 MIT 许可证公开。
- 建立文章导读、受控语言概念和 Skill 项目独立节点，复用已有 Manim 页面并关联 LLM Wiki 方法页。
- 文章建议属于实践经验，未报告统一的量化评测；Skill 的结构检查不验证语义等价。

## [2026-10-03] ingest | λ₀ / HumanVerse-500

- 归档 arXiv:2610.00438，并新增独立论文实体页。
- 记录 HumanVerse-500、三阶段训练、仿真与真机结果；当前代码、模型和数据入口待发布。

## [2026-10-03] ingest | embodied-manipulation-benchmarks | 仿真操作评测基准

- **意图：** 整理 LIBERO-Plus / PRO、RoboTwin 2.0、RoboCasa365、BEHAVIOR Challenge 与 RoboLab。
- **开源核查：** 新增项目补入官方代码/项目页和数据入口；RoboCasa365、RoboLab 复用现有归档。
- **关键页：** 更新 LIBERO 实体页与具身评测选型总览，增加项目 sources 归档。

## [2026-10-03] ingest | π₀.₅ 官方项目页与 openpi 源码

- 为既有 π₀.₅ 论文详情补充官方项目页、PDF、代码仓来源，归档 openpi 的公开 checkpoint、推理/微调入口与开源边界。
- 保留既有 arXiv 论文节点，关联 Knowledge Insulation 技术来源；未复制生成 catalog 或 log.md。

## [2026-10-03] ingest | Muse Gadgets 设备 SDK 与边缘交互接口

- 官网与 SDK 均已公开：ESP32 固件及 Linux / Raspberry Pi 设备接入代码采用 Apache-2.0 主许可证；助手模型与云端服务实现不在本次开放范围内。
- 归档设备扩展方式，并在 LLM 机器人接口页补充权限继承和机器人安全边界说明。

## [2026-10-02] lint | 全量 wiki lint 清零：复核 6 个陈旧页（OTRetarget/LT-Mem/IronMind 交叉对照）、6 篇 paper 补「与其他工作对比」、6 个数据集补「数据模态」、2 页补感知栈枢纽回链

## [2026-10-03] ingest | KUAVO-VLA-1.0 — 归档模型发布信息、工具链、数据入口与开放边界

- 新增独立实体页，并交叉链接 LeTools、VLA 与具身数据路线。
- 沿用既有 LeTools-Learning 来源档案，补记 KUAVO-VLA-1.0 发布与申请审核状态；截图评测指标按厂商自报值处理。
- 区分 GPL-3.0 工具代码、需审核的模型权重和配套数据许可。

## [2026-10-02] ingest | 补全 LT-Mem 官方 Demo、视频与论文资源入口

## [2026-10-02] ingest | OTRetarget 论文、项目页与 OMOMO 数据入口核查

## [2026-10-02] ingest | Eidon AI — 归档可穿戴硬件、仿真与 POV/IMU 数据，补充采集路线节点

- 官方归档页确认公司停止运营，公开 Tracker/Glove/Sim 仓库以及 CC-BY-4.0 数据集。
- 将配对集与纯视频集分开记数，并链接到数据采集产业地图。

## [2026-10-02] ingest | sources/sites/anthropic-claude-shaped-science.md 与 sources/repos/bootloops.md — 收录科研选题文章及固定版本计算工具箱，两个独立节点接入 AI Auto-Research；通用代码已开放，部分依赖与成果在仓外

## [2026-10-02] ingest | 智践行 Microduck 入门微信专辑 4688586645438726146；5/5 独立 overview 节点 + sources；④ Camoufox 全文；开源已核查

## [2026-10-02] ingest | Argus（Pantheon）机器人数据审计

- 意图：将训练前数据质量审计接入机器人学习主线。
- 开源结论：官方代码 Apache-2.0、标注 CC BY 4.0；原始录像与可选手部模型保留各自许可。
- 关键页：[Argus](../wiki/entities/pantheon-argus.md)，从 LeRobot 生态入口互链。

## [2026-10-02] structural | 图谱页 graph.html 不启用点阵随指针点亮（画布点阵与全站 body 底纹点亮均关闭）

## [2026-10-02] ingest | wiki/overview/humanoid-motion-intelligence-day1-data-retargeting.md — 为 Day 1 文章补独立综述节点和 30 篇论文入口

## [2026-10-02] ingest | 收录 React Bits 动效组件与官网入口，核查 MIT + Commons Clause 限制和 BlurText 源码，关联机器人展示界面的工程实践。

## [2026-10-02] structural | 点阵随指针点亮扩展到全站：深色主题 body 底纹（所有页面）、首页 Hero、图谱页画布点阵；有 prefers-reduced-motion 降级

## [2026-10-02] ingest | 接入 Serial Studio 硬件遥测仪表盘；核查 GPL 核心与专有 Pro 边界，补充单关节调试示例并关联 PlotJuggler。

## [2026-10-02] structural | 首页 P1 交互动效：Hero 点阵随指针点亮；知识图谱预览改为分帧离屏收敛，滚到区块时节点自中心向外扩散入场（终态与原先一致）；均有 prefers-reduced-motion 降级

## [2026-10-02] structural | 首页 P0 交互动效：首屏以下区块滚动入场、入口卡指针光斑、主题切换圆形扩散（View Transitions）、路线 / 公司展开高度过渡；均有 prefers-reduced-motion 降级

## [2026-10-02] ingest | sources/blogs/humanoid_motion_intelligence_day1_data_retargeting_2026_10_02.md — 逐篇映射30篇论文，补齐8个独立详情页；核查DexMV/TRAM代码与其余项目开放状态

## [2026-10-02] ingest | sources/papers/supermap_arxiv_2608_22896.md — 收录 CMU AirLab 时空 SLAM 与语言导航空间记忆；核查源码待发布，关联 VLN。

## [2026-10-02] ingest | sources/papers/dawn_arxiv_2609_29092.md — 接入深度去噪与潜变量对齐四足跑酷，核查训练/回放已开源及真机入口边界，关联感知运动中心

## [2026-10-02] ingest | NeoHorse-1（2609.08183）：routing harness agentic post-training 与 RSI 原型环；TokenRhythm 权重+推理示例已开源，入库实体页与交叉引用

## [2026-10-02] ingest | sources/papers/hiphi_arxiv_2608_16222.md — 接入 HiPHI 高精度动捕与同步 HOI，核实数据门控、镜像口径及 Viewer 开放边界，补齐数据选型与重定向/具身数据路线

## [2026-10-02] ingest | LocoWM：复用已有节点，补充成对检查点、初始掉落排除口径与预测式残差入口

## [2026-10-02] ingest | Sparse-WAM：新增动作引导稀疏加速节点，明确无需额外训练和同后端 1.56×，实现入口未核实

## [2026-10-02] ingest | Fiatlux：归档代码与仿真数据，区分八项可达和四项攀爬未验证，新增任务评测节点

## [2026-10-02] ingest | 用户指定 mp.weixin iNL8tMNdUA87 IROS2026 九篇获奖；9/9 独立节点（5 新建 4 升级）；开源 3 已开 4 待发布

## [2026-10-01] structural | 清单索引实体页读者化：rcl / painode / paper-pai / paper-as 共 740 页去掉「策展索引级详情节点 / 升格 / canonical」等维护者口径（与 sun254667 已改写页同措辞），4 个生成器模板同步；humanoid-locomotion 缩写表移到定义后；rcl 地图去「detail 节点」

## [2026-10-01] structural | top-50 hub 读者化：7 张 Awesome 清单技术地图详情列改为标题链接（含 4 个生成器）；Physical AI / AwesomeSim2Real 地图与 Paper Notebooks 索引去维护者口径；manipulation / humanoid-locomotion 关联页面拆出「按主题索引」

## [2026-10-01] ingest | sources/repos/awesome-gpt-6-astra.md 与 sources/sites/astra-games.md — 收录 AI 浏览器交互作品集，核查展示站和部分作品源码开放边界，提炼机器人教学展示参考。

## [2026-10-01] ingest | AgentsDock 项目页+双仓（ZhengyiLuo/AgentsDock·AgentsServer）；Apache-2.0 已开源；实体 agentsdock + 交叉 zhengyi-luo；自动合并 PR

## [2026-10-01] ingest | arXiv:2510.20808 Reality Gap 综述深读：项目页核查无代码，升格实体页 taxonomy/recipe，补 sources 与 Sim2Real 概念互链

## [2026-10-01] ingest | RoM-Nav arXiv:2609.19272 深读 ingest：sources/sites 核查待发布、论文实体页方法/实验/Poisson CBF 与 GTI 交叉引用

## [2026-10-01] ingest | Generate Track Improve（arXiv:2609.31577）— 2026-10-01 复核查项目页 Code 仍待发布，同步开源锚点日期

## [2026-10-01] ingest | TactileStep（2609.28959）入库复核：main 已有实体页；补 unitree-g1 关联；项目页仍无 GitHub（待发布）

## [2026-10-01] ingest | ChunkTrust arXiv:2609.39754 — AHS/QHA 动态 execution horizon；项目页核查 hf618/ChunkTrust + Niugan/ChunkTrust 已开源

## [2026-10-01] ingest | LocoWM（arXiv:2609.39179）：CASIA 等 WM 引导 preactive 残差高精度行走；Go2-W 真机三任务；GitHub 已开源；升格 paper-locowm 并交叉轮足/loco-manip。

## [2026-10-01] ingest | 微信公众号两篇：电机概念 FAQ + MIT 阻抗模式整定；新建 sources/blogs、wiki/concepts/electric-motor-fundamentals 与 wiki/methods/mit-joint-impedance-mode-tuning

## [2026-10-01] ingest | Grounded Action Model (2609.23863)：Northwestern/UW/NUS 3D grounding 操纵 foundation model；项目页+GitHub 核查为代码待发布；升格 paper-grounded-action-model-3d-grounding 并链入 vla.md

## [2026-10-01] ingest | 多模空间一周 VLA 第二篇（15 篇）；新建 12 实体 + 技术地图；复用 G0.5/StellaVLA/Neural Introspection Gating

## [2026-10-01] ingest | 微信公众号 LLM 推理显存估算长文；新建 concepts/llm-inference-vram-estimation + sources 归档；交叉 VLA 部署与国内租卡；自动合并 PR

## [2026-10-01] ingest | ImageNet 一手资料：CVPR 2009 建库论文、IJCV/arXiv ILSVRC 综述、image-net.org 站点归档；升格 dataset-imagenet 实体页

## [2026-10-01] ingest | Lot Vulture 项目页+GitHub lotvulture/lotvulture；源码可得 PolyForm Shield 衍生许可；实体 lotvulture + ORT 交叉；自动合并 PR

## [2026-10-01] ingest | Science Advances aed5603（FLEXOR 脊-腿协同，北理工×南大）；sources+实体页；项目页无代码

## [2026-10-01] ingest | LP-ACRL（arXiv:2601.17428 / RA-L）：ETH RSL 学习进度自动课程 + ANYmal D rough 2.5m/s；项目页无 GitHub，确认未开源

## [2026-10-01] ingest | Qwen-Audio-Agent（arXiv:2609.25195）：sources+实体页，开源全双工语音 harness 与座舱 mixed 评测；交叉更新人形语音交互方法页

## [2026-10-01] ingest | 写个 goodMan Modern Robotics 微信专辑 10 篇：C-space→twist/wrench→PoE FK；Camoufox 专辑同会话抓取

## [2026-10-01] ingest | Trendshift 批次：archify/gods-eye-view/ponytail/i-have-adhd/ECC/open-code-review/hindsight/OpenMAIC/paperclip/hyperframes 十项 Agent 技能与工具入库（7 新实体 + 3 热度更新）

## [2026-10-01] ingest | TactileStep（2609.28959）：清华 G1 足底压力四相位 RL；项目页 CoRL 2026 Spotlight，代码待发布；深化 paper-tactilestep 与 sources/sites

## [2026-10-01] ingest | LiveTalking（lipku）实时流式数字人引擎；项目页+GitHub 已开源、权重网盘；实体页 livetalking，交叉 MetaHuman/teleoperation
