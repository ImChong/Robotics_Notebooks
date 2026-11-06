# World Model for Robot Learning 综述通读（训练闭环与三线 taxonomy 视角）

> 来源归档（原文通读 · 支撑 `wiki/overview/robot-world-models-training-loop-taxonomy.md`）

- **标题：** World Model for Robot Learning: A Comprehensive Survey
- **作者：** Bohan Hou, Gen Li, Jindou Jia, Tuo An, Xinying Guo 等（NTU / UC Berkeley / Stanford / 东京大学 / Oxford / Microsoft / ETH / Princeton / Harvard；通讯 Jianfei Yang）
- **类型：** paper / survey / world-model
- **原文：** <https://arxiv.org/abs/2605.00080>（2026-11 通过 arXiv PDF v1，43 页，pdftotext 获取全文）
- **一句话说明：** 以机器人学习为中心，把世界模型按「与策略耦合」「作为模拟器（RL / 评估）」「机器人视频世界模型」三条线整理，并主张评价应看动作忠实、物理一致与下游控制效用，而非视觉逼真。
- **核对范围：** 通读摘要、§1 引言、§2 背景（世界模型 / 视频生成 / 策略定义）、§3 世界模型作为策略（解耦 IDM、单骨干、MoE/MoT、统一 VLA、潜空间）、§4 作为模拟器（RL 与评估）、§5 机器人视频世界模型四阶段、§6 导航与自动驾驶、§7 基准 / 数据集 / 结果（三类基准、表 5–6 的结论性文字）、§8 挑战与未来方向全文；表 1–6 的逐格数值与参考文献列表仅浏览，未逐项核对。本页引用的其他论文（WorldVLN、Ctrl-World 等）只依赖综述中的转述与 wiki 既有内容，未单独通读。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **定义（§2.1）：** 世界模型 = 机器人 / 具身语境下的智能体–环境动力学预测模型，给定当前状态 / 观测与动作预测下一状态或未来序列；「动作」既含低层 a，也含高层语言 / 任务指令 l。功能性定义：不是生成了看似合理的未来画面就算，必须捕获与机器人交互相关、能支撑策略相关计算的环境演化；最重要的子类是动作条件世界模型。状态可以是视觉观测、潜状态、结构化物理状态，甚至符号状态；实践中最常见的是视频世界模型。
- **三种能力（§1）：** foresight（执行前预见后果）、imagination-driven planning（想象 rollout 比较选择候选行为）、data amplification（合成示范 / 交互轨迹）。
- **概率视角（§3.1）：** 策略 p(a|o,l)、被动世界模型（视频生成）p(o'|o,l)、可控世界模型、逆动力学模型，都是同一联合分布 p(o',a|o,l) 的不同边缘 / 条件查询，因此世界模型与策略可自然耦合。
- **§3 策略耦合五类：** (a) 解耦 predict-then-act（UniPi、VPP、MimicVideo 等：视频模型预测未来，IDM 式模块反推动作，趋势是从像素 rollout 走向紧凑、结构化的中间表示）；(b) 单骨干联合建模（UVA、UWA、VideoVLA、Cosmos Policy、DreamZero 等；关键差别是推理时视觉分支保留多少）；(c) MoE/MoT 专家耦合（GE-Act、Motus、LingBot-VA、BagelVLA、DiT4DiT、Fast-WAM；Fast-WAM 结论是收益更多来自训练期视频共训而非推理期显式想象）；(d) 统一 VLA（GR-1、WorldVLA、DreamVLA、UniVLA、InternVLA-A1 等，未来预测内化于同一多模态骨干）；(e) 潜空间世界建模（FLARE、VLA-JEPA、JEPA-VLA、WoG、DIAL）及符号 / 规划器侧世界模型。作者强调视频预训练骨干是否优于同规模 VLM 骨干仍是开放经验问题。
- **§4 模拟器两用途：** RL 后训练（UniSim、World-Env、VLA-RFT、WMPO、World-Gymnast、RISE 等：世界模型输出 (ô, r̂, d̂)，用 GRPO 类目标优化策略；第二层是 World-VLA-Loop、VLAW、WoVR 的世界模型与策略共同演化，模拟器自身需被修正）；评估（GPC、IRASim、World-in-World 候选筛选；TD-MPC2、LeWorldModel 的 MPC；WorldEval、Veo 世界模拟器评估 Gemini Robotics 策略的离线评估）。警示：WoVR 指出幻觉与长程误差会直接污染评估信号。
- **§5 视频世界模型四阶段：** 想象式监督（Dreamitate、RoboDreamer、DreamGen、PhysWorld）→ 动作可控（IRASim、RoboMaster、Ctrl-World、EnerVerse-AC、EVA）→ 结构感知（Mask2IV、TesserAct、RoboVIP）→ 基础模型化（Vid2World、Genie Envisioner、DreamDojo、WoW、Cosmos Predict 2.5、GigaWorld-0）。结论：瓶颈不再是生成逼真未来，而是未来须与动作因果对齐、长程物理自洽、多视角 / 本体一致、可执行。
- **§6：** 导航（Pathdreamer、NWM、VISTAv2 等：把不可见空间变成可规划的预测底物）与自动驾驶（MILE、OccWorld、GAIA-1、DriveWM、DriveVLA-W0 等）。
- **§7 评测三分类：** 开环动作条件生成质量（RBench、EWMBench、DreamGen Bench、EVA-Bench）；闭环任务效用与策略评估（WorldArena、WorldEval、WorldGym、World-in-World；排序一致性、价值保真度、决策可靠性比像素精度更有信息）；物理一致 / 可控 / 可执行诊断（WorldSimBench、WoW-World-Eval 含 IDM 图灵测试、DrivingGen、WM-ABench）。结论：视觉合理只是控制效用的弱代理，没有单一指标足够。数据集按轨迹语料、跨本体、人类视频、触觉 / 力觉、合成数据多轴比较；失败恢复与决策敏感变化数据仍稀缺。
- **§7.3 结果：** LIBERO、RoboTwin 2.0、CALVIN、SIMPLER 上，解耦 / 共骨干 / 混合 / 统一 / 潜空间各范式都有强结果；长程（LIBERO-Long）是主要区分点；跨基准迁移弱、汇报不标准；作者据此认为有效具身控制不要求照片级视频生成。
- **§8 挑战：** 因果条件缺口（未来更多依赖历史与任务意图而非待执行动作）；效率（扩散去噪延迟、MPC 需反复 rollout）；多模态感知（触觉 / 力觉与视觉的异步、维度失衡）；经典控制整合（稳定性保证）；符号结构整合（混合世界模型）；缺少公认的功能导向评价指标，倡议以任务成功率、策略排序保真度、可执行性诊断为紧凑标准集。

## 对 wiki 的映射

- [`wiki/overview/robot-world-models-training-loop-taxonomy.md`](../../wiki/overview/robot-world-models-training-loop-taxonomy.md)
- 既有摘要归档：[`wm_robot_survey_arxiv_2605_00080.md`](wm_robot_survey_arxiv_2605_00080.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2605.00080>
