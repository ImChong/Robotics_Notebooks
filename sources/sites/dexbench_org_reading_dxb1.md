# DexBench 规范站通读（dexbench.org/en/ 全文）

> 来源归档（原文通读 · 支撑 `wiki/entities/dexbench.md`）

- **标题：** DexBench — Defining the Benchmark for Industrial Dexterity（Documentation v1.0）
- **作者 / 机构：** RLWRLD 发布，NVIDIA 为 Platform Partner；页脚 “Published by RLWRLD · 2026”；列出 9 家 Industry Partners
- **类型：** site / benchmark-spec
- **原文：** <https://dexbench.org/en/>（2026-11 拉取；页面正文由 SPA 从 `https://dexbench.org/data/contents_en.json` 加载，本次读的是该 JSON 全文，约 140 KB）
- **一句话说明：** 工业灵巧操作的开放任务规格：OSC 六轴诊断难度、五种 Dexterity Regime 规定能力，18 任务 / 55 用例 + 采购清单与定制夹具。
- **核对范围：** 通读 `contents_en.json` 的 home（问题陈述、四原则、分类框架）、osc（六轴定义与度量、敏感度、行业相关性）、dex（五 Regime、规则 A–E、传感/数据表、流程）、全部 18 个 task 及 55 个 case 的 definition / initial / final / failure、objectReference 与 customObjects 的结构（5 套定制套件及其 usedIn）；采购清单 45 行与对象清单 48 行只核对条目数与前若干行，未逐条点开亚马逊链接。站点无论文 / PDF，无逐任务分数，故无实验章节可读。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题（home.problem）：** 四点——无共享任务词汇（物流 pick 与电子装配 pick 同词不同法）、难度不可比（插 USB 与插活塞杆同称 insertion）、基准不可复现（物体 / 初态 / 成功判据各异）、缺少研发优先级地图。
- **四原则（home.principles）：** State Transition（只定初态与目标态，方法开放）、State-Based Judgment（终态可验证且安全不破，不比轨迹相似度）、Real Objects（可买、给尺寸重量材质）、Breakdown Curves（不看成功率，看从可靠到失败的参数区间）。
- **OSC（osc）：** `OSC = (C_geom, C_force, C_contact, C_obs, C_deform, C_dyn)`，六轴各自独立打分；三个关键问题：为何失败（70% 成功率不说明原因）、何时需要高灵巧、传感 / 控制 / 学习如何设计。关键原则：OSC 升高不是状态空间变大，而是容差包络 ε 变得极窄或集中在特定坐标。
- **六轴度量：** geom（对称阶 / 曲率、对准容差 mm 或角度、有效构型占比）；force（力矩包络宽度、所需力分辨率、越限后果）；contact（模态切换次数、接触片几何多样性、摩擦变化）；obs（被遮挡的关键状态比例、传感模态缺口、名义状态估计不确定性）；deform（有效自由度、历史依赖、材料参数不确定）；dyn（状态演化时间尺度 ms、不连续事件、最大可容忍控制延迟）。
- **行业（osc.industry*）：** 制造 / 服务 / 物流三大板块；四条“为何 OSC 是工业核心问题”：结构性部分可观测、单次执行内 OSC 动态切换、多轴复合（插入 = geom+force+obs，开包 = contact+deform+obs）、产线时间压力抬高 C_dyn。同一任务在不同行业的 OSC 画像可不同（电子装配 vs 食品加工的力敏感度）。
- **Dexterity 定义（dex）：** “灵巧不是手的属性，而是解的属性”；两指夹爪用指步态解决高接触复杂度任务，在语境下比解不了的五指手更灵巧。操作被表述为部分可观测的混合动力学（粘 / 滑 / 滚 / 卡 / 脱模态切换；视觉看不到内部接触与摩擦；动态场景里端到端延迟与控制带宽决定成败）。
- **五 Regime（E1–E5）：** Grasp Diversity（瓶颈：硬件运动学与接触面；度量：解空间覆盖）、Spatial Precision（传感分辨率 / 运动学精度 / 柔顺调参；度量：位姿 sigma mm、°）、Temporal Precision（感知延迟 / 控制频率 / 预测精度；度量：时延 τ ms）、Contact Precision（力传感带宽 / 驱动反驱性 / 阻抗控制；度量：力 / 阻抗差 N、Nm）、Context Awareness（遮挡下状态估计 / 任务图复杂度 / 重规划；度量：阶段 / 分支正确率）。“Dexterity = 敏感度带宽”：在成功条件敏感度集中的坐标上给出足够带宽与稳定性。
- **瓶颈规则（dex.mappingRules）：** A：C_geom→Spatial；B：C_dyn+C_obs→Temporal；C：C_force+C_contact+C_obs→Contact；D：C_deform+C_obs→Context；E：C_geom+C_contact→Grasp Diversity。
- **传感 / 数据策略表：** Grasp：RGB-D、触觉阵列 / 形状多样性；Spatial：高分辨率立体、激光轮廓 / 位姿真值；Temporal：高帧率相机、IMU / 时序与延迟剖析；Contact：F/T、触觉 / 力曲线与阻抗日志；Context：多模态融合 / 阶段标注与失败分类。流程：Real Object → OSC Analysis → Regime Classification → Task Design → Case Evaluation。
- **18 任务 / 55 case：** T00 Special Picking 4、T01 In-Hand Reorientation 4（禁双手）、T02 Bimanual Regrasping 3、T03 Precision Insertion 6、T04 Hand Fastening 3（禁工具）、T05 Constrained-Axis 5、T06 Control Interface 4、T07 Force-Regulated Wiping 2、T08 Flowable Material 4、T09 Fabric Folding 2、T10 Cable Winding 2、T11 Package Handling 3、T12 Selective Sorting 1、T13 Heterogeneous Bin Packing 2、T14 Box Sealing 1、T15 Precision Arrangement 3、T16 Tool-Use 4、T17 Moving Object 2。每个 case 写 Initial / Final / Failure（常见失败：外物辅助、强行插入、物体损坏、掉落、体积误差超限）。
- **附录：** Object Reference 含对象清单（48 行）与采购清单（45 行，亚马逊链接）；5 套定制套件：Outlet（3-B/3-C）、Hole（3-D–F、4-A/B、16-A–D）、Hexagonal Bulb（4-C）、Valve（5-A–C）、Switch（6-A–D），均由市售件加固定组成。
- **未见内容：** 站点不含 GitHub 链接、论文、逐任务基线分数或 breakdown curve 数据。

## 对 wiki 的映射

- [`wiki/entities/dexbench.md`](../../wiki/entities/dexbench.md)
- 站点编译归档：[dexbench-org.md](./dexbench-org.md)

## 参考来源

- 项目页：<https://dexbench.org/en/>
- 内容数据：<https://dexbench.org/data/contents_en.json>
