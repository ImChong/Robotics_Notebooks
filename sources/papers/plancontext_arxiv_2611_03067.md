# PlanContext：成对导航干预评测上下文驾驶决策（arXiv:2611.03067v1）

> 一手论文摘录与核对记录；以下数字来自论文，不代表本仓库独立复跑。

- **正式标题：** PlanContext: Benchmarking In-Context Driving Decision-Making
- **中文标题：** PlanContext：上下文驾驶决策基准
- **作者：** Ruikai Li、Naiyu Gao、Hao Shan、Yizhe Fan、Yanhao Wu、Han Jiang、Yilong Ren、Zhenning Li、Haiyang Yu、Yanhu Shan、Rui Wu、Hao Zhao、Hongyang Li、Zhiyong Cui
- **机构：** 北京航空航天大学（Beihang University）；地平线（Horizon Robotics）；澳门大学（University of Macau）；北京智源人工智能研究院（BAAI）；清华大学智能产业研究院（AIR, Tsinghua University）；香港大学 OpenDriveLab（The University of Hong Kong）
- **论文：** <https://arxiv.org/abs/2611.03067>
- **版本：** arXiv:2611.03067v1，2026-11-02，cs.RO / cs.CV；目前属于预印本，不以此推定已正式录用。
- **官方全文 HTML：** <https://arxiv.org/html/2611.03067v1>
- **论文 PDF：** <https://arxiv.org/pdf/2611.03067v1>
- **项目主页：** <https://plancontext.github.io/>
- **类型：** paper / autonomous-driving / benchmark / evaluation / NAVSIM
- **入库及核对日期：** 2026-11-10
- **核对范围：** 逐节阅读 arXiv v1 官方 HTML：摘要、§1–5（含 §3.1–3.4 方法、§4.1–4.4 实验、公式 1–9 和表 1–5）、附录 A（场景挖掘与排除规则、表 6）、B（路径拓扑、成对变轨和 LC 连续性）、C（车辆足迹、安全、评分）、D（样本组成、统计、实现口径、因素分析、稳健性、专家复核）及 E（定性实例）；官网核对 Code / Datasets 开放状态。使用与 PDF 同版本的官方 HTML 阅读正文与附录；没有独立执行 benchmark。
- **通读：** 已完成

## 要回答的问题

传统 NAVSIM 的 PDMS 等指标能判断规划轨迹的行驶质量，但在「车况与道路完全相同，仅改变高层导航命令」时，未必能解释模型是否做出了合理决策。PlanContext 因此把**意图—场景—决策一致性**从一般轨迹质量中独立出来评测；不是新控制策略。

## 方法核验（§3，图 1–2，公式 1–9）

1. 固定场景传感器观测与车辆历史，分别输入原导航意图 c^N 和替代意图 c^F；每个模型生成两条预测轨迹 T^N、T^F。输出归一为未来 4 秒的 8 个自车位姿（后轴中心）。
2. 从 NAVSIM navtest 挖掘高置信路线分叉、车道可达性、驾驶阶段和方向约束；模棱两可的几何候选剔除。记录轨迹用于构造路网与阶段证据，**不是唯一正确轨迹**。
3. 将实例分为 A / B1 / B2 / C / L / LC 六类，对应 Follow / Wait / Reject / Reject / Follow / Preserve；按类别判定原、替代轨迹在语义上是否合理，以及成对行为是否适当改变或保持。
4. 每个实例的成功条件为 q_i = C_i ∧ R_i^F ∧ E_i^F：上下文决策正确、替代条件下整车足迹始终符合道路区域、替代条件下动态安全符合 NAVSIM 碰撞/可行驶区域/逆行/交通灯/TTC/舒适性要求。LC 有 uncertain 状态，排除于最终分母。
5. 每类先计算成功率 S_k；NAS=S_A；CPS=(S_B1+S_B2+S_C)/3；NCS=sqrt(NAS×CPS)；LAS=S_L；MCS=S_LC。**NCS 是路线层聚合指标，车道层 LAS/MCS 单列，且 NCS 是几何均值而非全样本微平均。**

## 六类条件、数量与期望行为（表 1、附录 A、表 8）

| 类别 | 触发条件 | 应对行为 | 样本 |
|---|---|---|---:|
| A | 备选路线在 2 秒内可达 | 跟随可执行替代路线 | 334 |
| B1 | 备选路线在 4.5–8 秒后可达 | 等待，当前不可抢转 | 251 |
| B2 | 局部没有满足意图的可行目标 | 拒绝该方向指令 | 5,285 |
| C | 已越过路线决策/承诺点 | 拒绝事后改道 | 580 |
| L | 邻接目标车道几何及拓扑可达 | 执行有效换道 | 2,716 |
| LC | 横向操作已经开始 | 保持当前操作连续性 | 395 |

- 样本筛选：12,840 个初始 scene–condition 候选，运动筛选后 12,242，最终 **9,561 条**；对应 **5,911 个**互不重复的物理场景。B2 比例大，不能直接按样本比例得出总体 NCS。
- A/B1 需要稳定分叉几何与可靠到达时间；C 依据过去转弯与承诺点距离；LC 根据历史横向运动及保护窗口检查不合理的逆转。完整几何阈值与边界情形见附录 A/B。

## 实验与统计核验（§4、附录 D）

- **23 个公开端到端驾驶模型**，跨 VA（vision/state-to-action）、VLA、WA（world-model-based action）三种范式；作者依照统一的 4 秒轨迹输出表示重新评测。
- PDMS 是全部 NAVSIM navtest 的常规质量指标；mPDMS 是同 **5,911 场景**上、原导航指令下的匹配指标；NAS/CPS/NCS 是改变导航意图后的新评测。它们不是同一物理含义。
- mPDMS 与 NCS 的 **Spearman 相关系数 −0.542**（跨 23 模型排序），说明常规指标模型排名不能代替上下文决策排名；是相关性描述，不等于因果检验。
- 最优 NAS **22.46%（OneVL）**、相应 CPS **68.79%**、NCS **39.30%**；OneVL 的 mPDMS 为 **80.79**（论文表 2）。
- DrivoR：mPDMS 83.56 / NAS 19.76 / CPS 61.23 / NCS 34.78；高 mPDMS 的 CLOVER（94.35）却只有 NAS 0.60 / NCS 5.86，Drive-JEPA（94.18）为 NAS 0 / NCS 0。这些是作者同一表内对比，不等同于全场景表现优劣。
- 六类跨 23 模型平均成功率（论文表 3）：A 10.01%、B1 62.91%、B2 73.28%、C 34.98%、L 0.45%、LC 85.51%。**可执行替代路线/车道意图最难**，保持已有横向动作相对容易。
- **统计：** 以 5,911 个 scene token 为 bootstrap 抽样单位，10,000 次重采样；23 模型两两比较共 253 组，采用 Benjamini–Hochberg 校正；193 组 NCS 差异在 q<0.05 下显著。OneVL 对 ChainFlow-VLA 的 NCS 差为 4.17 个百分点，95% CI [0.74,7.43]、校正 q=0.0252；论文也提示相邻多个模型的区间重叠。
- **人工核验：** 三位自动驾驶背景审阅者在规则确定后盲审 500 场景条件和 11,500 次模型-实例评估，讨论后约 0.7% 被判与自动结果不一致（§3.2 / 附录 D.9）。这是抽样规则核验，不是证明所有判定无误。

## 原文适用边界与审读提醒

- 基于已有 NAVSIM 日志、固定观测与 **4 秒短时轨迹** 的成对、偏开环/非反应式评价；不能直接声称真车闭环道路安全或长期导航能力。
- 第 A 类要求在可执行分叉内体现替代意图，有可能与某些安全延后策略发生指标取舍；阈值依赖地图/参考几何、轨迹归一与场景筛选，适合跟补充诊断一起读。
- 各类别极不平衡，尤其 A 仅 334、B1 仅 251；宏平均和 bootstrap 场景级聚类重要。比较模型时须固定同批场景、相同条件、相同版本。
- 项目页显示 **Code Coming soon / Datasets Coming soon**（2026-11-10）；论文给出评测规则但尚无可核验官方运行脚本，故**当前不能独立复跑 PlanContext 作者全套结果**。
- 关于 OneVL：PlanContext 表 2 的 PDMS/mPDMS/新指标，与 OneVL 原论文自己的 NAVSIM PDM-score（88.84）并非同一比较口径，不能跨表混为一谈。

## 对 wiki 的映射

- [PlanContext 唯一论文项目实体](../../wiki/entities/paper-plancontext.md)
- [OneVL 的驾驶潜推理模型](../../wiki/entities/paper-xiaomi-onevl.md)
- [NAVSIM 原始基准导航页](../../wiki/entities/paper-rcl-ref-e25271fa6f028e5611cf-navsim-data-driven-non-reactive-autonomous-vehic.md)
