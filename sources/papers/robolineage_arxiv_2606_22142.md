# RoboLineage — 原论文通读核验（arXiv:2606.22142v1）

> 一手论文归档：记录论文真实方法、实验协议、附录、适用边界；不把项目宣传或源码现状视为论文指标的独立复现。

- **英文标题：** RoboLineage: Agent-Native Data Lifecycle Governance Across Robot Policy Iterations
- **中文标题：** RoboLineage：跨机器人策略迭代的 Agent 原生数据生命周期治理
- **作者：** Qian Luo, Wentao Guo, Zhennan Qin, Nanchun Guo, Yunhan Zhao, Yi Ma, Yanchao Yang
- **机构：** TranscEngram（忆生科技）、The University of Hong Kong（香港大学）、Beijing Institute of Technology（北京理工大学）
- **论文：** <https://arxiv.org/html/2606.22142v1>
- **PDF：** <https://arxiv.org/pdf/2606.22142v1>
- **提交：** 2026-06-20，arXiv:2606.22142v1 [cs.RO]
- **会议：** CoRL 2026（作者项目页明确标注 Accepted；arXiv v1 的引用类型仍为预印本）
- **项目：** <https://robolineage.github.io/>
- **官方代码：** <https://github.com/robolineage/robolineage>
- **核对范围：** arXiv HTML 完整论文 §1 引言、§2 相关工作、§3.1–3.7 方法、§4.1–4.4 实验/消融、§5 局限；Appendix A（实现与 prompts）、B（13 类 agent）、C（schema/lineage）、D（VSA）、E（review）、F（实例）、G（训练适配）、H（前端）、I（实验协议与后端敏感性）、J（更多案例）、K（限制与未来工作）；重点对照 Figures 1–5、Tables 1–3、10–12。
- **通读：** 已完成
- **核心定位：** 不是一个端到端机器人控制策略，而是围绕既有采集/训练/评估栈、以 schema 验证的 artifacts 构成完整策略数据 provenance 和复采闭环。

## §1–§2：问题与差异

- 现有机器人模仿学习已积累视频、轨迹与策略权重，但 rollout 为何接受/拒绝、失败发生阶段、训练集如何形成、某 checkpoint 与数据版本之间的因果证据常分散在脚本、目录和口头经验里。
- 数据版本控制/MLOps 通常记录已经形成的数据集和模型；本文把**具身语义判断及数据准入决策**作为版本化记录中的一等公民，不替代 RoboMimic、ACT、Diffusion Policy 或 VLA learner。

## §3.1–3.2：生命周期与 Agent 边界

- 一轮抽象链条：机器人本体 b、任务合同 τ、采集方式 c_k → rollouts R_k → review/admission A_k → 数据集 D_k → 训练 T_k → 策略 π_k → 评估 E_k → 部署/复采决策 G_k。
- 类型化谱系图 L_k=(V_k,P_k)；节点含 robot profile、task config、rollout、VSA snapshot、review annotation、dataset decision/lock、train record、policy metadata、evaluation、deployment ticket、master review；边表示 observes/reviews/admits/freezes/trains/evaluates/recommends。
- Agent 读取已有谱系上下文并输出自身职责范围的 artifact；schema 校验、hash、父节点链接、operator overrides 使语言模型输出**不能直接**作为训练准入/部署命令。

## §3.3–3.7：实现主链

1. **Onboarding/Task Config：** 绑定 ROS2 摄像头/状态/动作话题和时间戳，形成 robot profile；任务合同规定 phase、success、risk、snapshot priority 与 admission policy。
2. **三轨 rollout：** 原始 rosbag/时间戳无条件优先保存；Online Visual Snapshot Agent（VSA）按事件取稀疏帧，返回 phase/progress/risk/confidence 及证据；录制结束后 Post-Rollout Review 异步汇总 terminal/post-terminal/temporal evidence packets，产出 final annotation 和 failure analysis。
3. **Data Governance：** 将 semantic result 与训练准入隔离；accepted_for_training、human review priority、failure pool、rejection 原因分别记录；失败轨迹可进入诊断池但默认不进入主要 imitation manifest。
4. **Data Health：** 用覆盖度、动作阶段均衡、缺失视觉证据与 OOD 失败位置形成复采目标；独立于 rollout-level admission。
5. **Framework Discovery/Adapter：** 发现现有训练框架的数据格式和运行命令；适配 accepted manifest，记录 dropped/remapped fields 与 adapter hash；锁定数据，训练后生成 checkpoint、训练日志、不可变 policy metadata、parent policy。
6. **Eval/Deployment/Master：** 按真实 evaluation trials 写成功/失败阶段/回归证据；建议 deploy、hold、rollback 或 collect more；Master 读取摘要 artifact 保留跨轮失败记忆，输出下一轮数据采集 brief，而不是擅自执行部署。

## §4：实验证据及口径

### Table 1：25 tasks / 500 rollouts，直接视频 VLM vs 完整系统（百分比）

| 指标 | Direct video-to-VLM | RoboLineage | 口径 |
|---|---:|---:|---|
| Outcome accuracy | 78.4 | 96.0 | 任务结果判定 |
| Failure-phase accuracy | 58.0 | 88.0 | 失败阶段定位 |
| Admission route accuracy | 79.6 | 96.4 | 500 rollouts 上与专家共识的精确路由一致率，非真实策略成功率 |
| Bad-data leakage | 16.0 | 2.1 | 194 个专家拒绝 rollout 被错误纳入主训练 manifest 的比例 |

- schema-only、VSA-only、post-review-only、移除 final observation 或移除 packetization 的消融均弱于完整系统；同一数据池、受控基线，不能把改善归功于单一模块。
- ARX / ACT-style、Realman / Diffusion Policy、GALBOT G1 / VLA-LeRobot-style，主要为视觉可判断的操作任务。

### §4.3 / Appendix I：策略质量与人工时间

- 八个不同于复采实验的任务；每任务共用 **100 个 candidate rollouts**，两套工作流各挑 **60 demonstrations**，相同 learner/命令/预算；60 次独立 evaluation trials/任务，平均真机成功率 **92.7%（RoboLineage）vs 93.1%（人工）**。仅能说明此匹配条件下近似，而非无差别更优。
- 12 组匹配的每轮 30-rollout session：专家 **130.2±10.7 min** → 系统 **54.3±5.9 min**，活跃人工时间减少 **58.3%**；collection 分别 40.5 与 40.6 min，几乎不变；优势来自 review（59.2→9.4 min）、onboarding（10.1→2.3）和 framework preparation（20.4→2.0）。
- **不包括**硬件初始 bring-up、第一次 adapter authoring、训练 GPU 时间；不是机器人实际执行速度提升 2.4 倍。
- Table 2 的 API 成本 **约 $11.60/30-rollout cycle** 为论文记录的调用配置估算，不涵盖全套硬件/人工成本。

### §4.4 / Appendix I.6：跨轮复采

- Cube stacking 初始成功 **25.0%**；三轮复采后独立 final test **83.3%**，random **36.7%**、phase heuristic **66.7%**、no-memory **70.0%**、human **86.7%**。
- Drawer opening 初始 **26.7%**；final **78.9%**，random **37.8%**、phase **62.2%**、no-memory **66.7%**、human **81.1%**。
- 每方法三次独立运行、每轮 **30 accepted demos** 和 **30 diagnostic trials**，最后每 run **60 independent trials**；固定接受样本数**不代表**采集尝试次数相同。
- 叠方块失败从 OOD 抓取位置→释放前中心对齐→限时释放，复采意图随错误分布逐轮变化。

## Appendix A–K：关键约束

- **A/B/C：** 13 类 agent；每个输出 schema_version、producer、依赖/父节点 ID 和 hash，支持 superseded artifact 与 operator override 留痕；无需图数据库。
- **D/E/F：** Raw capture 不可被 VLM 超时阻塞；invalid JSON/parse 失败保守标注 needs_review；VSA phase 可有条件向前跳跃或出现带证据回退；Post-review 通过 terminal / temporal packets 与确定性聚合避免模型直接决定 admission。
- **G/H：** framework adapter 支持直接 manifest、用户转换命令或生成 converter；前端为人机共同操作界面，部署建议仍需操作者/既有安全流程批准。
- **I.5：** 200-rollout 后端敏感性 replay：Claude Sonnet 4.6 route admission 96.0、leakage 2.0；Gemini 3.1 Pro Preview 95.0/2.5；qwen3-vl-plus 86.5/4.5。不同视觉模型导致可观偏差；schema 不是语义准确性保证。
- **K：** 主要对实验室内既有视觉操作与任务定义有效；高接触/触觉/音频依赖、柔性物体、泛化到跨实验室 federated provenance 未充分证明；本系统不提供独立硬件安全保证。

## 对 wiki 的映射

- [RoboLineage 唯一项目实体](../../wiki/entities/paper-robolineage.md)
- [官方主页核验](../sites/robolineage-project.md)
- [官方 GitHub 实际源码核验](../repos/robolineage-official.md)
