# RoboFinals RL Environments：官方产品技术页全文核验（2026-11）

> 来源归档（Lightwheel 官网官方产品页 + 官方发布公告），归纳而非转载全文。

- **英文标题：** Introducing RoboFinals RL-Env
- **中文标题：** RoboFinals 强化学习环境：从真实任务与实测物理构建可训练仿真
- **主要一手来源：** https://lightwheel.ai/robofinals-rl-env
- **产品页 Early Access：** https://lightwheel.ai/robofinals-rl-env/early-access
- **官方公告：** https://cn.linkedin.com/company/lightwheel2026 （「Today, Lightwheel is introducing RoboFinals RL Environments」）
- **官方 X：** https://x.com/LightwheelAI/status/2120350216679493836
- **发布时间：** 官方公告对应 2026-11-11（CoRL 2026 期间）；产品页未单独标示上线日
- **入库日期：** 2026-11-13；**全文复核：** 2026-11-14
- **核对范围：** 已直接打开并通读官方独立产品页 https://lightwheel.ai/robofinals-rl-env 所有正文区块（Overview、Why the Environment、Real-World Simulation、Task Demos、Measured Physics、Expert Review、Training to Evaluation、Scaling with EgoSuite、Environment Library、Join Us）、Figure 1–6 的文字图注及 Agent/Expert 四阶段表；打开并检查 Early Access 申请页字段；与光轮官方 LinkedIn 全文及既有 RoboFinals 评测产品页交叉核对。图中精确数值未提供可独立提取的表格，未臆算曲线误差。
- **通读：** 已完成
- **代码：** 官网 RL-Env 产品页未列专属训练环境源码仓库／可运行入口／SDK；[Newton](https://github.com/newton-physics/newton)、[LW-BenchHub](https://github.com/LightwheelAI/LW-BenchHub) 是相关但独立的开源基础设施，不等于 RL-Env 开源。
- **数据／任务包：** 官方产品页无直接下载链接或公开任务清单；需 Early Access 申请。
- **沉淀到 wiki：** [RoboFinals RL-Env](../../wiki/entities/lightwheel-robofinals-rl-env.md)

## 官方产品页技术细节

### 1. 任务来源与场景重建

- **输入为人类真实作业：** EgoSuite 用第一人称视角记录人在生产任务中使用自己的手和工具，产品页首发环境为 **汽车维修（automotive repair）**。
- **Agent 解析：** 从 EgoSuite 采集解析任务结构、子目标、涉及物体及操作先后顺序。
- **重建：** 将零部件、工具和灵巧手本体构建为 **SimReady OpenUSD** 场景资产；Figure 1、2 展示从第一视角任务到仿真场景的对应关系。
- **难点：** 多阶段任务持续数分钟、前后阶段依赖；本体是**灵巧手而非平行夹爪**；软管、线缆、垫片以可变形体形式表示。这三点是页面针对同一环境的说明，而非三套单独发布的数据集。

### 2. 测量、CAE 与 Newton 物理

- **参数：** 摩擦（friction）、刚度（stiffness）、阻尼（damping）来自对真实物体和材料的测量；不是仅依赖视觉形状或经验手调。
- **可变形物：** 软管／线缆／垫片响应经过工业级 **CAE（Computer-Aided Engineering）** 比较或验证后进入环境。
- **耦合求解：** 官网称刚体、可变形体、颗粒体接触在**同一个时间步**通过耦合求解器交换接触力，后端为 **Newton-native**。
- **图表边界：** Figure 4 为软管模拟响应与实测响应对照示意；页面正文**未提供可引用的误差百分比、拟合指标、样本数或测试协议**。不能把其他 Lightwheel 新闻中独立任务的「5%」当成本产品统一精度。

### 3. Agent 构建与人工专家逐环节审查

官网 Agent/Expert 四阶段对照表：

| 阶段 | Agent 工作 | Expert 工作 |
|---|---|---|
| Human Demo / EgoSuite | 从录像解析子目标、物体与操作次序 | 核对任务结构符合实际作业 |
| Scene / SimReady OpenUSD | 重建部件、工具、灵巧手 | 审查场景重建结果 |
| Calibration / Measured Physics + CAE | 设置实测参数、执行可变形响应验证 | 审核物理保真度 |
| RL-Env + Eval / Newton · RoboFinals | 装配环境并执行评测 | 审批任务发布 |

此表为官方设计／流程陈述，不等于公开自动生成算法、验收阈值或可复现的测试脚本。

### 4. 训练与评测口径

- **RoboFinals RL-Env 是训练侧**；既有 [RoboFinals](../../wiki/entities/lightwheel-robofinals.md) 是评测侧。官网明确称两者采用**相同任务、场景、物理**，共享任务定义；训练完的策略可依照相同任务规范参加评测。
- **完整基准＋公开排行榜：** 官方称 **“later this year”**（以 2026-11 页面对外表述，即**计划**在 2026 年后续开放）；不是已上线的既成事实。
- **未公开：** RL 算法（PPO／SAC 等）、奖励函数、观测与动作张量格式、episode reset、机器人控制器、并行环境数、训练吞吐、baseline 成功率、训练曲线及 Sim2Real 成功率。

### 5. 随 EgoSuite 扩展的环境分布

官方使用**同一套四轴分类**整理 EgoSuite captures 和仿真环境：

| 分类轴 | 官网示例 |
|---|---|
| Scene（场景） | Domestic、Industrial、Retail、Hospitality、Healthcare |
| Operation and Difficulty（操作与难度） | Pick-and-place、alignment、tool use、force regulation、precision work under closed-loop control |
| Asset Type（资产类型） | Rigid、articulated、deformable、fluid |
| Task Horizon（任务时域） | 多阶段／长时域 |

- Agent 随数据增长按分类生成、标定环境，由专家审核；目标是让训练环境的采样分布跟随真实人类工作任务分布。
- **Figure 6 展示例子：** Cooking Rice（家庭／流体及颗粒／多阶段）、Assembling an Engine（工业／刚体及铰接／工具精密操作）、Organizing the Operating Room（医疗／刚体及柔性／长时域）、Buffer Installation（工业／对齐／装配）。这是官网展示的环境例子，**不能据此认定已经公开下载**。

### 6. 入口与开源核查

- 官方官网入口 [Join the Waitlist](https://lightwheel.ai/robofinals-rl-env/early-access) 已直接打开，表单字段：Name、Email、Organization/Lab、Model Type（可选）、RL-Env 使用目标、Project Link/Additional Details、预计开始时间（可选）。
- 使用目标勾选项包括：Training a Robotics Policy Using RL、Evaluating an Existing Policy、Sim2Real Policy Transfer、Research on Real2Sim2Real、Design and Explore Environments and Tasks、Others。
- 官网另有「Partner with Us」入口，面向工厂、医院、零售、物流、机器人厂商共同构建 benchmark 任务；入口在本次抓取工具中不可直接读取具体申请正文。
- 官网**未发现该 RL-Env 产品自身**的公开代码、安装指南、训练脚本、任务包下载、许可证或直接 API 文档。不能以 [LW-BenchHub](https://github.com/LightwheelAI/LW-BenchHub) 代码替代或据此画 RL-Env 源码运行时序图。
- 官网产品页**未点名 SimFoundry 的具体集成路径**。光轮官方 LinkedIn **另一个**演示提及 SimFoundry 对 Newton 中米粒／纸袋／电饭煲混合物理模拟的求解器库与实测材料参数支持，但不可据此推定 RL-Env 的内部必经模块。

## 已确认与尚不能证明的区别

| 主题 | 已直接核实的官方披露 | 仍无独立证据 |
|---|---|---|
| 任务真实来源 | EgoSuite 采集 → Agent 解析 → SimReady 场景 | 采集数据标注错误率与真实使用规模 |
| 物理标定 | 真实摩擦、刚度、阻尼 + CAE 验证 + Newton 耦合 | 误差统计、物理求解性能、第三方标定复现 |
| 任务复杂性 | 首发汽车维修；长时域、灵巧手、可变形 | 全部环境的完整下载清单与任务逐项难度 |
| 环境扩张 | 四轴 taxonomy，Agent 构建专家审查 | 自动构建成本、成功率、审核 SLA |
| 训练/评测 | 两产品共享任务／场景／物理定义（官方陈述） | 已公开 RL 基线、开放 leaderboard、真机迁移结果 |
| 可获取性 | 官网 Early Access 表单 | 开源许可证、可下载代码与训练环境 |

## 推荐后续核验项

- 申请 Early Access 后索取环境 SDK、任务及版本清单、obs/action/reward/reset 定义、并行吞吐与机器人本体。
- 索取 Figure 4 对应实验数据（真实软管多次测试、误差区间、参数与 solver 版本）、CAE 配置与真实接触测量。
- 评测策略时要求与训练任务共享的场景版本、随机化参数、评测协议、成功判据及防数据泄漏方案。
- 仅在官方产品实际开放运行代码后，阅读 README/入口源码并更新运行时序图。

## 与 wiki 的对应关系

- [RoboFinals RL-Env 训练环境](../../wiki/entities/lightwheel-robofinals-rl-env.md)
- [RoboFinals 评测平台](../../wiki/entities/lightwheel-robofinals.md)
- [Lightwheel 公司](../../wiki/entities/lightwheel.md)
