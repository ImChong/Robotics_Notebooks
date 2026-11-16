# 全路线知识审查 / Roadmap Knowledge Audit（2026-11-16）

## 基线、范围与方法

- 固定基线：`main@83b2277a3682a4b183bbb87bd4dd26f2fa7e2fd3`。
- 全局扫描：`wiki/` 7,801、`roadmap/` 32、`tech-map/` 20、`references/` 20，共 **7,873 个 Markdown 文件**，含目录文件，不等同于独立论文数。
- 路线逐页核对：**1 条主路线 + 30 条纵深 + 1 条 wiki 工程入门子路线**，另核对 `roadmap/README.md` 目录，共 33 个文件。
- 近期端点差分：以上次全路线审查合入提交 `07bd89cfd70d8a0cfbc3a4db4427c92ed7092598`（PR #3733）为起点，至本次基线，wiki **新增 26、修改 483**，roadmap **修改 9**，无删除。按树差分计数，路径使用 `git diff --name-status -z` 解码，避免 Unicode 文件名漏计及 merge 重复累计。
- 对全库路径、H1、frontmatter、分类、身份与内链做扫描；结合近期差分按主题定位，再逐路线核对入口、阶段、先修、练习、相邻路线和来源。对发现的问题精读相关概念/方法/实体页，并核验下表中的一手依据。
- **本轮不是对 7,873 份文件及其所有引用论文逐篇重新通读或独立复现实验的声明。** 路线是精选路径，不要求每篇新论文都成为必修。公司/实验室 JSON 另用既有映射测试检查，没有重编路线或实体身份。

## 逐页结论

每页均按覆盖范围、节点归属/关系、依赖顺序、分类/交叉链接、时效、准确性、重复/遗漏检查。“保留”指本轮未发现需要直接改该页的路线问题；新节点可由已有枢纽承接。

| 路线/目录 | 结论与处理 |
|---|---|
| [总览](../../roadmap/README.md) | 保留：30 条纵深、主路线和工程入门子路线入口完整，无需新增目录或路线。 |
| [运动控制主路线](../../roadmap/motion-control.md) | 更新：修正 MR 第 7 章/力控章节、LQR/MPC 的适用条件与终端代价、凸 QP 的唯一性与实时性、浮动基广义接触力/驱动力矩区别。分清策略查询/跟踪/内环频率，修正重定向 FK/IK 层级及 90 天路径中的迁移证据口径。同步自测文字和图。 |
| [PCB](../../roadmap/depth-pcb-design.md) | 更新：原文允许功率/高速/量产按项目选读，图和 Stage 7 先修却要求串行。统一首屏、图、先修与练习，保留全部 Stage。 |
| [遥操作](../../roadmap/depth-teleoperation.md) | 保留：全身/手部并行进入下游；近期 HEFT、TeleGate、CLOT、手术等原文边界更新仍由已有对应阶段承接。 |
| [火箭回收](../../roadmap/depth-rocket-landing.md) | 保留：GNC→LCvx→SCvx→内环→机载验证与全剖面主干完整，工业推测/公开资料边界已有说明。新增机器人材料未改变其基础依赖。 |
| [力矩电机](../../roadmap/depth-torque-motor-design.md) | 保留：指标→电磁热→电流环→驱动板→标定→模组验收，与 PCB/整机输出接口清楚。 |
| [传统模型控制](../../roadmap/depth-classical-control.md) | 保留：动力学/QP→简化步态→质心→MPC→WBC；Go2 感知 NMPC 在感知越障作对照，新橄榄球轨迹优化工作由 WBC 枢纽承接，不改通用先修。 |
| [整机硬件](../../roadmap/depth-humanoid-hardware-design.md) | 更新：去掉把“单点地/单端屏蔽”推广到所有接口的暗示；按回流、隔离、等电位及厂商要求设计，并用 EtherCAT 官方屏蔽说明作反例。同步图、问题与练习。 |
| [安全控制](../../roadmap/depth-safe-control.md) | 更新：正文允许传统 WBC/MPC 直接进入安全部署，首屏和图仍把 Safe RL 作为必经阶段；改为选读分支。 |
| [RSI](../../roadmap/depth-rsi.md) | 保留：Self-Rewarding 已在打分器/验证锚阶段；新科学预测评测与 AI 系统课程由 auto-research/编码基础枢纽承接，不据新单点结果升级 ignition 结论。 |
| [接触操作](../../roadmap/depth-contact-manipulation.md) | 保留：接触控制→示范/探索条件下选型→触觉与真机，原有枢纽覆盖近期手物、残差与硬件更新。 |
| [导航](../../roadmap/depth-navigation.md) | 保留：SLAM/Nav2/学习导航/VLN 分层；新增 BEV 由空间表征与相关感知节点承接，不将鸟瞰表征等同于导航策略。 |
| [模型架构](../../roadmap/depth-model-architecture.md) | 保留：StarVLA 框架与 α 受控基线已分开；AIInfraGuide 已接性能阶段。可解释神经元/OptiMer 属底层机制或语言模型研究，不强设为机器人架构先修。 |
| [模仿学习](../../roadmap/depth-imitation-learning.md) | 更新：Stage 2 只产参考轨迹，不能称其“训练的策略”直接部署；练习改引用 Stage 3/4。区分仿真训练迁移与真机示范策略的接口/泛化验收。 |
| [RL 运动控制](../../roadmap/depth-rl-locomotion.md) | 保留：直接策略→Sim2Real 与可选 MPC/WBC 分支已一致；运动模仿归进阶，未把最新推理模型当 locomotion 训练前置。 |
| [移动操作](../../roadmap/depth-loco-manipulation.md) | 保留：基座/数据/技能/统一模型/任务层清楚，近期 WARP、手物与交互工作由任务/参考/触觉枢纽承接。 |
| [人形足球](../../roadmap/depth-humanoid-soccer.md) | 保留：全向走/恢复→感知→踢球→多机战术；新美式橄榄球投掷不是足球踢球节点，不因同含 football 而错误挂载。 |
| [动作重定向](../../roadmap/depth-motion-retargeting.md) | 更新：关系段仍误写“L2 的 FK/IK”，与 Stage 0 的 L1 指引矛盾；统一指向 L1。新 EgoMo3R/MacroHands 已在数据源 Stage 2，并明确不是重定向器。 |
| [群控展演](../../roadmap/depth-humanoid-swarm-performance.md) | 更新：Stage 1 已分清音乐生成与 SONIC 跟踪，Stage 5 却仍称“音乐 token 实时条件”；改为生成器→参考运动→跟踪器，补回多机时钟/时间轴职责与官方来源。 |
| [跨本体](../../roadmap/depth-cross-embodiment.md) | 保留：差异画像→接口→架构→混训→人机迁移→新本体验收；近期统一动作/历史与人视频更新由现有枢纽承接，不重复造实体。 |
| [Sim2Real](../../roadmap/depth-sim2real.md) | 更新：Stage 4 已允许跳过视觉 Stage 3，首屏和图仍串行；同步为按输入选读外感知分支。PEAR 已在仿真训练阶段，有独立推理策略边界。 |
| [拳击](../../roadmap/depth-humanoid-boxing.md) | 更新：潜空间先验降低探索难度，不等于直接电机空间不可训练或任意状态下都天然可行。结合原文对照及训练机制，改为记录采样/扰动下的摔倒和越界，不把有限检查当成普遍保证。 |
| [ICL](../../roadmap/depth-icl.md) | 保留：示范归纳/历史记忆/快权重对照与未见集协议已有入口，近期记忆和模型合并资料不据名称自动升格为 ICL。 |
| [BFM](../../roadmap/depth-bfm.md) | 保留：ILA 已在适应阶段，位置是冻结 WBT 前的参考修正；无需在跟踪预训练或跨本体阶段重复挂载。 |
| [具身测评](../../roadmap/depth-embodied-eval.md) | 保留：按认知/世界模型/任务/运控对象分支，NimCritic 已进入长程过程评测；统一 VLA 评测工具可由新生态对照页和已有评测基础设施枢纽到达。 |
| [感知越障](../../roadmap/depth-perceptive-locomotion.md) | 保留：表征→训练→越障→系统整合；Go2 OCS2 感知 NMPC 已作为模型控制对照，注明足端区域约束默认关闭，未把 RViz 区域当闭环证据。 |
| [动作生成](../../roadmap/depth-motion-generation.md) | 保留：参考生成/物理化/在线规划接口完整，Kimodo 与 Kimodo-Policy 分工已明确；EgoMo3R 是重建上游，不另列通用文本动作骨干。 |
| [扩散与流匹配](../../roadmap/depth-robotics-diffusion-dit-flow.md) | 保留：8 篇代表作与 PredActor Stage 8 均存在；DiT 与生成机制两轴、阅读顺序与继承关系已区分。 |
| [VLA](../../roadmap/depth-vla.md) | 更新：Stage 4 补新开源生态对照页，给 Hy/Turbo、StarVLA、统一评测与后训练工具一个按职责选复现的入口；不逐篇铺成必修，也不把工具框架当 VLA 模型。 |
| [Real2Sim](../../roadmap/depth-real2sim.md) | 保留：几何/外观→物性→场景/episode→回训评测，与执行器/Sim2Real 共享标定；近期可视化与重建更新已有入口。 |
| [具身数据](../../roadmap/depth-embodied-data.md) | 保留：采集→清洗→格式→合成→配比闭环；近期设备、参考数据与 OXE 口径更新由既有阶段承接，没有将公开数据量与总训练量混用。 |
| [WAM](../../roadmap/depth-wam.md) | 保留：Cascaded/Joint 并行及训练期预测对照已一致；近期 V-JEPA2/LpWM/INTACT/开放状态更新均有现存入口。 |
| [工程入门子路线](../humanoid-practitioner-entry-roadmap.md) | 保留：背景→局部切口→单机构闭环→作品集，属于非算法工程入门，不复制为第 31 条纵深。 |

## 近期 26 个新增节点的承接

- **已直接挂载**：AIInfraGuide、MacroHands、EgoMo3R、Go2 OCS2、ILA、NimCritic、PEAR、Self-Rewarding、StarVLA-α、StarVLA 框架。核对现有 Stage 的职责，不重复加链接。
- **本轮补共同入口**：VLA 生态三转变对照页承接 Hy-Embodied-0.5-VLA、TurboVLA、verl-vla、vla-evaluation-harness；分别是策略与基础设施，不是五个新模型家族。
- **由专题承接**：BEV→空间表征；Neurons as an Interpretable Basis→可解释性；OptiMer→模型算术/持续预训练；Science Prediction Universe→auto-research 评测；橄榄球投掷→WBC/轨迹优化；Harvard CS2680、Modern Web Guidance→编码工程基础。
- **保持领域边界**：Moshi、PACE、XBot-Omni 属语音/人格/表达交互，由语音交互或多模态枢纽承接；AesFormer 的 action 是图像编辑指令，不挂到机器人动作策略路线。

可达性只用于发现导航遗漏，不以“图上有任意一条路径”证明语义归属正确。对上述节点同时核对了职责和路线已有入口，未新建或合并论文实体。

## 关键变化的一手核验

| 一手依据 | 核对范围与用途 |
|---|---|
| [Modern Robotics Ch 7](https://modernrobotics.northwestern.edu/nu-gm-book-resource/kinematics-of-closed-chains/)、[§11.5](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-5-force-control/) | 官方目录/讲解确认 Ch 7 是闭链运动学，力控在 Ch 11；纠正先修阅读入口。 |
| [MIT LQR 讲义](https://underactuated.mit.edu/lqr.html) | 有限时域、时变与轨迹跟踪各节及 Riccati 递推；稳态解作为终端代价的等价条件另做数值反例核验。 |
| [Boyd & Vandenberghe §4.2–4.4](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf) | 凸性/严格凸、全局最优与 QP；不能由半正定推出唯一解，也不能由凸性推出 1 kHz。 |
| [Herzog et al. §2.1，式 (1)–(2)](https://arxiv.org/pdf/1410.7284v2) | 动力学与接触约束：广义接触力含未驱动基座分量，需和驱动力矩联合求解。 |
| [RoboStriker v1 §3.3、表 2、附录 A/B](https://arxiv.org/html/2601.22517v1) | 原始电机空间存在可训练对照，潜空间由技能蒸馏、先验和归一化形成；将其表述为经验性改进，不扩展成任意状态/接触下的硬约束保证。 |
| [SONIC 官方项目](https://nvlabs.github.io/GEAR-SONIC/) 与[部署源码固定版本](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/1e1ec4f5f27e6c3e03048f693bd7c60cce04afda/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp#L2186) | Multi-Modal Control 明确 GEM 先生成运动；源码 `control_dt_(0.02)` 与多线程职责确认 50 Hz 跟踪，不将其写成音乐原生输入或 1 kHz 策略。 |
| [Beckhoff EtherCAT/Ethernet 屏蔽说明](https://infosys.beckhoff.com/content/1033/ethernetcabling/1952414987.html) | 双端大面积低阻抗端接、避免长尾辫、等电位/RC 环流处置；只作对应接口实例，不把它推广为所有传感器统一接法。 |

以上为路线纠错所需的具体章节/源码核对，不声明重新通读全部论文。其他历史条目的开放状态仍以各实体标注的日期和版本理解；本轮没有重新测量硬件、跑模型训练或做 EMC 认证。

## 验证

- **结构与导航**：33 个路线/目录文件的 2,893 处本地路径引用、1,116 个唯一目标均存在；保留全部既有 H1–H4、显式锚点和导出的阶段 ID/标题。10 条路线有内容修改，未新建路线或重复实体。
- **普通本地预检**：已运行 `make ci-preflight`，因 **64 条既有 wiki/source 日期先后提示**退出 1；其余 lint 项为 0。这些提示是待 review 信号，不是本轮认定的 64 条内容错误。`wiki/` 与 `sources/` 均无改动，没有通过批量改日期消除提示。
- **Actions 规则的全量预检**：`GITHUB_ACTIONS=true make ci-check` 通过。生产 lint 在 Actions 环境原本就跳过上述日期比较，其余身份、内链、格式、图谱与导出检查照常执行；未修改规则或配置。全量图谱 7,831 节点、65,323 条边、0 孤儿；ingest 来源覆盖 7,798/7,798。
- **导出与图表**：最终文字修改后重新生成导出，`check_export_quality.py` **15/15**；Mermaid **12.3.0** 全库 **7,785 图 / 7,873 文件，0 错误**。派生 JSON、统计、`catalog.md` 和 `log.md` 均未提交。
- **回归测试**：路线页面、阶段/首屏/更新时间、正文/来源/健康度导出及公司/实验室映射的 Python 测试 **54 项通过**；`npm run test:frontend` **140 项通过**。
- **搜索**：现有回归 **39/40（98%，门槛 80%）**通过；查询“RL 训练 原地踏步 关节抖动”仍未把 `locomotion-failure-modes` 排入前五。未修改搜索实现或降低门槛。
- **数学反例复核**：对 2 个标量 LTI 系统、4 种有限时域检验稳态 Riccati 终端代价的首步等价；半正定 `diag(1,0)` 构造多个最优解，核对唯一性误区。这是辅助数值核验，不替代理论条件。
- **浏览器**：本地 Chromium 实际打开安全控制、Sim2Real、PCB 和运动控制页面，展开章节/自测；阶段数分别 5/6/8，三张分支图及 L2/L3 答案图可渲染，KaTeX 错误和页面 JavaScript 异常均为 0。测试以本机同版本 Mermaid 12.3.0 替代 CDN 脚本，未改变站点源码；不将此声明为公网 CDN 可用性测试。
- **已知显示限制**：安全控制图的个别节点下缘有 SVG 边界裁切；用基线原图在相同页面复现，属于既有标签扩框后的边界问题。本内容 PR 未改渲染器。
- `git diff --check` 通过；未修改 Python、前端代码或依赖。节点健康度表 **N/A**：没有 wiki 页变更。
