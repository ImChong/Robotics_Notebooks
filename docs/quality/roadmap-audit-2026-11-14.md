# 全路线知识审查 / Roadmap Knowledge Audit（2026-11-14）

## 基线与范围

- 起始基线：`main@5b261db31c1f893665d4ed87314b9983fe50eb4f`；检查期间快进并对合入差分补审，最终固定在 **`main@b10c0fbfa7744857fba927a4750ae22f4aa34634`**。
- 最终全局索引：`wiki/` 7,775、`roadmap/` 32、`tech-map/` 20、`references/` 20，共 **7,847 个 Markdown 文件**。目录页也计入文件数，不把它们全部称为独立研究成果。
- 路线范围：`roadmap/` 的 **30 条纵深 + 1 条主路线 + 1 个目录**，以及 `wiki/roadmaps/` 的 **1 条工程入门子路线**；共 32 条路线知识页和 1 个目录页。公司/实验室 JSON 是站点实体映射，另以既有测试核对 ID 引用。
- 近期差分：以 `2026-11-07 00:00 UTC` 前最后一个 main 一父提交 `5402ffa87da2150644272cc4df60bc095e1ec7c0` 为起点，至最终基线：wiki **新增 433、修改 567、删除 1**，roadmap **修改 31**。使用端点树差分，不重复累计 merge 与分支提交。
- 初查时开放的 PR #3726–#3730 用于区分待合入范围。期间 #3726–#3731 合入，已补审新增 26 个、更新 11 个 wiki 文件：OCR/HierText 与湍流方法由原有感知、CAE 枢纽承接；RoboHarness v3 同步到 VLA 部署入口；DAS 2.0 补入数据采集阶段。动作验证/门控/偏好、表征/数据、记忆时间偏差等新增研究，由既有 VLA、WAM、评测与记忆枢纽承接，不把每篇论文变成路线必修。未重复论文 ingest。

## 审查口径

对全局节点扫描路径、标题、frontmatter、分类和内链，按主题检索近期差分；逐条检查路线的入口、先修、阶段、例子、相邻路线与来源。精读发现冲突和新增入口所涉及的实体/方法/对比页，必要时回到官方项目、论文或 README。**本记录不是对全库 7,847 份材料或所有被引论文逐篇重新通读的声明。**

每条路线均核对覆盖范围、归属/跨主题关系、依赖顺序、内容时效、准确性、重复和遗漏。路线是精选学习路径，不要求把同主题的每篇新论文直接挂入；已有概念、任务、方法和专题枢纽能承接的内容不再铺一层清单。相同实体被多路线引用属于知识图复用，不因多重归属拆页或另造实体。

## 逐页结论

“保留”表示本轮未发现值得直接修改该页的路线问题，不表示相关领域已无开放问题。

| 页面 | 核对结论与处理 |
|---|---|
| [路线总览](../../roadmap/README.md) | 更新：原末段说不维护导航等独立路线，与已有目录矛盾；改为鸟瞰→目标纵深，并补非算法工程入门子路线入口。30 条纵深不增不减。 |
| [运动控制主路线](../../roadmap/motion-control.md) | 更新：L5.2 区分直接 locomotion、运动模仿和 MPC/RL 组合，取消混合控制的通用先修要求；L9 修正 FM 必然更快/轨迹必直、chunk 预测长度等于开环执行长度的表述及自测答案。保留 L0–L12 与锚点。 |
| [PCB](../../roadmap/depth-pcb-design.md) | 保留：电路→第一块板→设计→调试主干明确，功率/高速/量产允许按项目分支；与驱动、整机 EMC 的板级接口已互链。近期软件与机械小工具不足以改变学习主干。 |
| [遥操作](../../roadmap/depth-teleoperation.md) | 更新：运动学指引由 L2 改 L1；全身与手部映射并列进入下游，避免只做手部采集也被要求先完成全身控制。保留采集、部署接管与无机器人采集的边界。 |
| [火箭回收](../../roadmap/depth-rocket-landing.md) | 保留：GNC→LCvx→SCvx→内环→机载验证的主线与回边清楚，工业事实/推测、摘要核对/全文证据已有边界说明；本周机器人论文不构成替换航天基础资料的理由。 |
| [力矩电机](../../roadmap/depth-torque-motor-design.md) | 保留：指标、电磁热、驱动、电流环、标定与模组交付分层清楚，PCB 与整机约束互链；现有关节轴承、摩擦及总线知识仍由对应枢纽承接。 |
| [传统控制](../../roadmap/depth-classical-control.md) | 保留：浮动基/QP→简化模型→质心→MPC→WBC 覆盖完整；近期 GCS/轨迹优化等工作属于已有 footstep/TO 分支，未把最新单篇强加为基础先修。 |
| [整机硬件](../../roadmap/depth-humanoid-hardware-design.md) | 保留：质量/功率/时序预算→机械/配电/通信→验收的交付链完整，向 Sim2Real 输出模型和实测参数；不与电机、PCB 子路线重复展开。 |
| [安全控制](../../roadmap/depth-safe-control.md) | 更新：区分奖励惩罚、CMDP 期望代价与逐时刻约束，修正负半定 Lyapunov 导数自动收敛的暗示；传统控制上机不必先学 Safe RL。 |
| [RSI](../../roadmap/depth-rsi.md) | 保留：改进对象、验证锚与递归加速证据分层已明确，DGM 近期原文更新已有链接；新机器人自改进/评测工作可放入现有权重、harness、评测支路，不把单轮增益升级为 ignition。 |
| [接触操作](../../roadmap/depth-contact-manipulation.md) | 更新：去掉 IL/RL 二选一的预设问题，按示范与探索条件选择组合；chunk 实验分别记录预测长度、执行前缀与接触力。力/触觉、顺应控制与下游学习的分工保留。 |
| [导航](../../roadmap/depth-navigation.md) | 保留：状态估计、Nav2、学习导航、VLN 与操作衔接齐全；近期 SLAM 节点由已更新的 SLAM 选型页/开源栈总览承接，不把各 SLAM 实现重复铺进路线。 |
| [模型架构](../../roadmap/depth-model-architecture.md) | 更新：区分整序列复杂度与缓存增量推理，避免直接把 O(n²) 当单控制步耗时；感知编码、动作头、多模态与部署分层保持。 |
| [模仿学习](../../roadmap/depth-imitation-learning.md) | 更新：视觉操作与人体技能分支分开，DP 不再依赖人体重定向，AMP/ASE 补 RL 前置；DP 属于生成式 BC，技能潜变量不自动消除分布偏移。 |
| [RL 运动控制](../../roadmap/depth-rl-locomotion.md) | 更新：MPC/WBC 为选修支路，AMP/DeepMimic 回归运动模仿；用 rl-mpc-locomotion 与近期复核的 ReCo 对照相反的控制层级，明确四足实证边界；仿真泛化/sim2sim 不再记作 sim2real。 |
| [移动操作](../../roadmap/depth-loco-manipulation.md) | 更新：ω-0 的代码/数据 WIP 已落后于实体与官方 README，改为入口已公开且依赖外部资产；接入 Kimodo-Policy，并区分视觉条件运动策略与显式世界后果建模。 |
| [人形足球](../../roadmap/depth-humanoid-soccer.md) | 更新：RL 基础由错误的主路线 L3 指向 L5.1；感知、踢球与多机战术仍由任务/方法枢纽承接，未把近期单机守门员示例当成整队方案。 |
| [动作重定向](../../roadmap/depth-motion-retargeting.md) | 更新：FK 先修指向 L1；运动学、物理感知、工具链、交互保持的分工及人形/四足支线保留，近期工具与手物方法不另造同论文节点。 |
| [群控展演](../../roadmap/depth-humanoid-swarm-performance.md) | 更新：SONIC 是运动跟踪接口，不能直接称为原生音乐 token 输入；以 AHEADance 说明音频生成器→跟踪器，明确单机流式生成不等于机器人状态反馈或群体时钟同步。 |
| [跨本体](../../roadmap/depth-cross-embodiment.md) | 保留：五层差异→接口→架构→混训→人机迁移→新本体验收已区分；新共享意图/人视频工作沿已有表征分支阅读，不将统一 schema 等同于消除动力学差异。 |
| [Sim2Real](../../roadmap/depth-sim2real.md) | 保留：资产/执行器、训练鲁棒、视觉与部署分层完整，Stage 4 已允许跳过视觉分支；SIL/PIL/HIL 与近期校准更新已有入口。反向孪生路线的职责矛盾在 Real2Sim 页修正。 |
| [人形拳击](../../roadmap/depth-humanoid-boxing.md) | 更新：RL 先修由 L3 改 L5.1；潜空间、自博弈、真机减损仍各有阶段，未把遥操作联赛与自主策略实验合并为同类证据。 |
| [ICL](../../roadmap/depth-icl.md) | 保留：示范表征、同本体/人视频、快权重/记忆与未见集协议已有分类和对照；近期上下文/测试时适应材料不应仅凭“context”名字重复加页或另设模型家族。 |
| [BFM](../../roadmap/depth-bfm.md) | 保留：运动跟踪、Goal/Intrinsic/FB 预训练、适应与感知扩展覆盖主线；SuperMimic、MorphMimic、ScaleBFM 已在跟踪阶段，新感知 BFM 属于现有扩展分支，不要求全部读完才可部署。 |
| [具身测评](../../roadmap/depth-embodied-eval.md) | 更新：图、首屏和先修统一成按对象分支，认知分不再称所有策略的必要条件；补 BEHAVIOR 2026 的部分完成度/观测/实例协议与操作基准有效性审计。 |
| [感知越障](../../roadmap/depth-perceptive-locomotion.md) | 保留：盲走基线→地形表示→训练→越障→导航/降级链完整；近期内部模型、训练配方与失效降级仍由已链接的任务枢纽、PIE、VB-Com 等承接。 |
| [动作生成](../../roadmap/depth-motion-generation.md) | 更新：补 Kimodo-Policy 与生成到闭环对比页；区分参考生成、视觉条件适配和 SONIC 执行，不把 Kimodo 本身等同于闭环操作策略。 |
| [扩散与流匹配](../../roadmap/depth-robotics-diffusion-dit-flow.md) | 更新：阅读顺序不暗示直接继承，DiT 骨干与生成机制正交；Octo 的 MLP 是扩散去噪头；FM 不自动单步/更快。与 VLA 页同步八篇代表作口径。 |
| [VLA](../../roadmap/depth-vla.md) | 更新：专题入口的七篇旧计数改为含 PredActor 的八篇；RoboHarness 入口去掉已被 v3 更新的旧版单点成绩。RT/OpenVLA/π、数据与部署主干及近期模型由现有索引承接，不扩大成全量公司清单。 |
| [Real2Sim](../../roadmap/depth-real2sim.md) | 更新：episode 孪生也可包含执行器/控制响应，不能把执行器划为 Sim2Real 独占；VLM 说明链接改到分类页，避免点击却进入 VLA 动作模型。 |
| [具身数据](../../roadmap/depth-embodied-data.md) | 更新：ARSTAG 从采集装置移到合成支路，补操作数据/基准选型与 DAS 2.0 代际/模态选型；无动作视频仍可预训练，时钟同步不等于消除链路/执行延迟。保留失败数据、许可与配比维度。 |
| [WAM](../../roadmap/depth-wam.md) | 更新：Cascaded/Joint 分支并列；SLIM 移到训练期预测对照并分清 H100 模型计时，LeWM/LG-DKGP 作为独立世界模型+规划对照，不按 world-model 名称自动归类。 |
| [工程入门子路线](../../wiki/roadmaps/humanoid-practitioner-entry-roadmap.md) | 保留正文：局部切口→单机构闭环→作品集的定位与算法路线不同；从总览补入口解决导航遗漏，不复制为新的 root 路线。 |

## 关键变化的一手核验

| 证据 | 核验用途 |
|---|---|
| [AMP](https://xbpeng.github.io/projects/AMP/) / [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/) 官方项目 | 方法主线是运动模仿和 RL 奖励，不是显式 MPC/WBC 架构。 |
| [rl-mpc-locomotion README](https://github.com/silvery107/rl-mpc-locomotion) | 高层策略预测 MPC 权重，低层 MPC；README 明说真机迁移不在本仓。ReCo 的反向层级对照另据本库近期原文核读节点。 |
| [Diffusion Policy 官方项目](https://diffusion-policy.cs.columbia.edu/) / [Octo 官方 README](https://github.com/octo-models/octo) | DP 建模示范动作分布并滚动执行，Octo 自述 Transformer-based diffusion policy；区分回归头与去噪网络。 |
| [Flow Matching 原论文 §4](https://arxiv.org/html/2210.02747v2) | FM 可使用不同概率路径；条件 OT 直线不意味着边缘速度场是 OT 解，不能推出固定采样步数或必然提速。 |
| [SLIM 原论文 §3.3、§4.4、附录 A.5](https://arxiv.org/html/2608.09771v1) | 推理通过训练塑形的隐式槽出动作，不显式预测未来观测；H100 eager/BF16、单次动作块模型计时排除加载、仿真与 RPC。 |
| [ω-0 README 固定版本](https://github.com/gentlefress/Omega-0/blob/2ca35a3ff9363c54c9c206ca101cdf58028824cb/README.md) | 训练/推理/采集/部署与 checkpoint 发布已勾选，仍需外部 Qwen/T5/V-JEPA/Wan/SONIC 等资产。 |
| [Kimodo-Policy README](https://github.com/kimodo-policy/kimodo-policy) | 已有训练/评测/真机指南与模型数据入口，BiGym 2.0 和 HumanoidToolBench 支持仍待发布；原项目网页在本轮抓取不可用，改从与既有归档一致的作者仓库核查。 |
| [AHEADance 项目](https://ihcr.github.io/AHEADance/) / [论文 III-E、V](https://arxiv.org/html/2611.13370v1) | 流式音乐生成接 SONIC；生成器没有机器人状态反馈，项目仍标 Code soon。网页演示不能代表研究实现开放。 |
| [BEHAVIOR 2026 官方规则](https://behavior.stanford.edu/challenge/evaluation.html) | 100 任务主分是末状态 BDDL 谓词部分完成度，另有观测限制及固定公开/隐藏实例；不等同于完整成功率。 |
| [DAS Ego 2.0](https://www.genrobot.ai/products/ego2) / [Dex 2.0](https://www.genrobot.ai/products/dex2) / [Gen-HumanEgo 官方数据卡](https://huggingface.co/datasets/genrobot2025/Gen-HumanEgo) | 当前硬件的视觉、关节与触觉输出不能反推到旧数据集；数据卡仍列 1600×1300 六相机、重建手部/深度/语义字段，须按实际发布字段选训练监督。 |

仅做路线层的核验与索引更新，没有重复 ingest、改写论文实体或宣称独立复现实验。历史开放状态未全部在线重测；本次更新有版本/日期依据，其余入口仍需按对应实体的核查日期理解。

## 验证

- **结构与内链**：33 个路线/目录文件的 2,806 处站内路径引用全部存在，去重后连接 1,077 个知识页。对比最终 main 的 H1、既有 H2–H4 和解析出的阶段 ID，无删除/重命名；仅专题页新增参考来源节。新先修锚点指向既有章节。
- **Mermaid**：锁定的 Mermaid 12.3.0 全量检查 **7,650 张图 / 7,847 个 Markdown 文件，0 错误**。
- **Python**：路线页面、阶段/首屏/更新时间导出、wiki 子路线导出、页面正文、来源/健康度导出、公司与实验室映射的既有测试 **53 passed**（`pytest --no-cov` 指定对应 10 个测试文件）。
- **前端**：`npm run test:frontend` **130 passed**。首次在受限执行环境中，Mermaid CLI 子进程测试遇到 `spawnSync EPERM`；以允许子进程正常启动的执行权限重跑后全绿，没有改测试或解析器。
- **全库门禁**：`GITHUB_ACTIONS=true make ci-preflight` 通过，含身份去重、导出、Mermaid、图谱、wiki lint、搜索回归和导出一致性；0 孤儿，来源覆盖 7,772/7,772。保留 2 条主线既有信息项：NavWAM 缺对比节，HierText 缺评测枢纽回链。
- **常规本地门禁的基线限制**：起始 main 上运行普通 `make ci-preflight`，仅被 **11 条**“sources 比 wiki 新”阻塞；最终 `b10c0fb` 基线上单独重算该规则为 **16 条**，且本分支对 `wiki/`、`sources/` 的差分为空。这是 main 上待复核的 source→wiki 日期标记，未通过批量修改 `updated` 消掉。仓库 `_check_sources_health` 在 Actions 环境按既有规则跳过此日期检查；因此上述通过结果明确称为 **Actions 口径**，不宣称普通本地 preflight 全绿。
- 最终基线上重新生成导出、复跑全量 Mermaid 和路线相关 Python 测试，并核对源文件差分；派生 JSON、catalog、log 和站点统计不随 PR 提交。

16 条本地日期标记涉及：action-chunking、sonic-motion-tracking、2026-10-18 技术地图、diffusion-policy、manipulation、HWM、LPWM、INTACT、MPC、V-JEPA2、WAM 综述、HarnessVLA、行为树 VLA 编排、Graph-as-Policy、VLA 开源复现全景与 Open X-Embodiment。它们需要按各 source 的真实内容另做 wiki 复核；本轮未宣称这些节点已完成更新。
