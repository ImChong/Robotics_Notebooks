# Agility Robotics（agilityrobotics.com / Research + Analysis）官网、技术栈与开源核查

> 来源归档（site / Agility Robotics 官方 + IEEE Spectrum + WEF + NVIDIA + SEC/Nasdaq 新闻稿 + GitHub/HF 核查）

- **入口 URL：** <https://www.agilityrobotics.com/resources?tab=research-%2B-analysis#tab-group>（同步核查 `/company-overview`、`/company/vision`、`/solutions`、`/solutions/digit-5`、`/press-releases`、`/latest-press`）
- **公司全称与上市代码：** Agility Robotics, Inc.（总部 Salem, Oregon；研发/部署中心 Pittsburgh, PA 与 Fremont, CA；2026-11-05 于 Nasdaq 上市，代码 **AGLT** / 认股权证 **AGLTW**）
- **成立时间：** 2015 年（Spin-out from Oregon State University Dynamic Robotics Laboratory；创始人 Jonathan W. Hurst、Damion Shelton 等，见 `/company-overview` 与 IEEE Spectrum 2019 署名）
- **GitHub 组织：** <https://github.com/agilityrobotics>（9 个公开仓库；仅 `cassie-doc` 与机器人相关且标 `End of Life`，无现役 Digit 源码）
- **Hugging Face：** <https://huggingface.co/agilityrobotics>（API 返回 `[]`，0 个公开模型）
- **开源状态：** **官方现役商业软件栈、WBC 模型与硬件 CAD 均未开源**（历史双足平台 Cassie 文档仓 `cassie-doc` 已标注 End of Life；无现役 Digit 源码、权重或公开训练数据集）
- **核对日期：** 2026-11-20
- **核对范围：**
  1. 完整展开并读取 `/resources?tab=research-%2B-analysis#tab-group` 全部 5 个标签页条目索引（Insights 4 页、Executive Perspectives 4 篇、Research + Analysis 7 篇、eBooks 3 本、Videos 6 页）。
  2. 完整抓取并通读 Research + Analysis **全部 7 篇正文**（含 *Science Robotics* 2026-08 作者版对照页、*Safeguarding America's Humanoid Future*、WEF 2026-04 PDF 原文抽取文本 `28,366` 字符、以及 IEEE Spectrum 2019/2021/2023/2026 四篇原文）。
  3. 完整抓取并通读公司简介、Vision、Solutions、Digit 5、Latest Press/Press Releases，以及 12+ 篇官方工程/AI/安全/部署文章（含 *Crossing the Sim2Real Gap With NVIDIA Isaac Lab*、*Training a Whole-Body Control Foundation Model*、*Agility and AI*、*Digit's Next Steps*、*Built for the Real World*、*Agility's Humanoid Deployment Process*、*Beyond the Hype*、*Jonathan Hurst answers your questions about building Digit 5*、*GXO RaaS / 100K Totes*、*Nasdaq AGLT 上市公告* 与 NVIDIA Case Study 页）。
  4. 通过 GitHub REST API（已认证）及 Hugging Face API 逐仓核查公开代码与模型资产。
  5. **未展开边界：** `/solutions/digit/product-sheet` 为表单门禁下载页（仅采集页脚说明）；eBooks 为表单门禁 PDF 与 Video 条目内嵌视频流未转录音轨。
- **覆盖 wiki：**
  - [敏捷机器人（Agility Robotics）](../../wiki/entities/agility-robotics.md)
  - [Agility WBC Foundation Model（Digit Motor Cortex 与 Isaac Lab Sim2Real）](../../wiki/entities/agility-wbc-foundation-model.md)
  - [Physical AI × 机械硬件（SciRob 2026）](../../wiki/entities/paper-physical-ai-mechanical-hardware.md)
  - [机器人基础模型与通用人形：公司技术路线对照（2026）](../../wiki/comparisons/robot-foundation-model-company-paths-2026.md)
- **通读：** 已完成

---

## 一、开源与公开学术资源核查（步骤 2.5）

通过 `https://api.github.com/orgs/agilityrobotics/repos` 与 `https://huggingface.co/api/models?author=agilityrobotics` 实测核查：

| 平台 / 仓库 | 状态 | 创建 / 最近推送 | 许可证 | 说明与边界 |
| --- | --- | --- | --- | --- |
| `github.com/agilityrobotics/cassie-doc` | 非 fork 公开文档仓 | `2017-09-06` / `2023-02-17` | `NOASSERTION` | 61 stars；README 明确标注 **`# NOTE: End of Life`**（Cassie 已停止官方支持与更新，不含现役 Digit 运行码） |
| 其余 8 个 `agilityrobotics/*` 仓库 | 全部 `fork: true` | `2023-11`–`2026-07` | MIT / Apache-2.0 / Null | 均为 CI/基础设施 fork：`cache`、`setup-jfrog-cli`、`rules_nixpkgs`、`jfrog-credential-helpers`、`cluster-api-provider-aws`、`fluent-bit-helm-charts`、`sequin`、`terraform-provider-temporalcloud`，不含机器人控制、仿真或 AI 模型 |
| `huggingface.co/agilityrobotics` | 空清单（`[]`） | — | — | 0 个公开模型权重与数据集 |
| 外部学术界基于 Cassie / Digit 硬件的工作 | 第三方仓库 | — | 各异 | 例如 UC Berkeley [`paper-digit-humanoid-locomotion-rl`](../../wiki/entities/paper-digit-humanoid-locomotion-rl.md)（arXiv:2303.03381）与 OSU 学术研究（arXiv:2103.15309 / 2506.02206）；**不属于** Agility 官方现役商业软件栈 |

**开源判定总结：** Agility Robotics 现役系统（**Digit 控制栈**、**WBC Motor Cortex**、**Agility Arc** 车队调度云平台、**Halos / 安全逻辑**）与 **训练/现场数据湖** 全部闭源。网站与论文仅公开架构思想、物理建模经验总结与商业安全标准框架。

---

## 二、Research + Analysis 标签页全部 7 篇条目逐篇归档

入口页面：<https://www.agilityrobotics.com/resources?tab=research-%2B-analysis#tab-group>（倒序展示以下 7 篇研究与行业分析）：

| 日期 | 标题 | 作者 / 出版方 | 原文链接 | 核心论点与可核查技术事实 |
| --- | --- | --- | --- | --- |
| 2026-08-12 | *Physical AI Is Enabled By Mechanical Hardware* | Jonathan W. Hurst（Agility / OSU）；*Science Robotics* 11(117) eaee2921 | [Agility 作者版](https://www.agilityrobotics.com/research-analysis/physical-ai-is-enabled-by-mechanical-hardware) / [DOI](https://doi.org/10.1126/scirobotics.aee2921) | 本库已独立建页归档（见 [paper_physical_ai...](../papers/physical_ai_mechanical_hardware_scirobotics_aee2921_2026.md) 与 [agility_physical_ai...](./agility_physical_ai_mechanical_hardware.md)）：主张在视觉–语言与 RL/IL 迅速普及的同时，**机械被动动力学、执行器柔顺与接触能量传递**才是 loco-manipulation 性能与安全的真正上限 |
| 2026-06-12 | *Safeguarding America's Humanoid Future* | Agility Robotics（政策白皮书） | <https://www.agilityrobotics.com/research-analysis/safeguarding-americas-humanoid-future> | 提出 6 大美国联邦政策支柱：① 激励美国本土制造与供应链（披露 Salem RoboFab 设计峰值产能 **10,000 台/年**、Fremont CA **60,000 sq ft** 设施、Pittsburgh PA 办公研发及 **371 名员工**）；② 对华人形机器人战略竞争与关税/反倾销/数据安全；③ 依托 **OSHA NRTL**、NIST 人形基准与 NTIA 推动自愿标准；④ 劳动力增强与技工培养；⑤ **Agility Arc + RaaS + NVIDIA Isaac Sim** 真实部署数据湖商业加速；⑥ 联邦采购与国防部指令 **DoW Directive 5000.94** 后勤维持应用（明确排除武器化） |
| 2026-04 | *Next-Generation Physical Autonomy: Advancing Intelligent Systems for a Thriving and Resilient Society* | World Economic Forum Global Future Council on Autonomous Systems（2025–2026，CEO Peggy Johnson 为贡献成员 p.12） | <https://reports.weforum.org/docs/WEF_Next_Generation_Physical_Autonomy_2026.pdf> | 以「技术创新速度 × 社会接受度」两轴提出面向 2031 年的 **4 种情景**（Integrated progress、Proven deployment、Tech disillusionment、Divided deployment），指出「过度承诺与早期事故导致公众幻灭（Tech disillusionment）」或「受控场景循证落地（Proven deployment）」比理想化的全面融合更为现实，提出 8 项治理与安全透明度目标 |
| 2026-05-20 | *Will Robotics Have a ChatGPT Moment?* | Jonathan W. Hurst & Hans Peter Brøndmo（前 Google X Everyday Robots 负责人）；IEEE Spectrum | <https://spectrum.ieee.org/robotics-ai-breakthrough> | 总结机器人 AI 落地的 **5 条硬事实**：① **YouTube-to-Reality Gap**（春晚武术/舞蹈等编排表演本质上只是低层防摔控制+预编排，远非开放环境自主劳动）；② **数据荒漠**（高维配置空间缺乏互联网级对齐语料；Everyday Robots 2022 年曾靠 **2.4 亿**次仿真样本才训出一个垃圾分拣模型）；③ **无单一通用模型**，胜出架构将是高层 **Agentic AI** 协调多专有小模型与多机协作；④ **硬件与执行器仍极端困难**（刚性位置伺服撞击破坏大，需力敏感、低惯量或被动/主动柔顺以利用环境接触，如「触边扭动插钥匙」）；⑤ **真正价值源于「简单重复任务」**（Moravec 悖论；真实客户现场迫使 Agility 将**安全架构**置于优先级首位） |
| 2023-07-22 | *It's Totally Fine for Humanoid Robots to Fall Down* | Evan Ackerman（专访 Jonathan Hurst、Bambi Brewer 及 Boston Dynamics CTO Aaron Saunders）；IEEE Spectrum | <https://spectrum.ieee.org/falling-robots> | 针对 ProMat 2023 Digit 摔倒事件解释工程真相：起因为手臂编码器罕见 bug 误触发急停（已修改固件策略为**冻结故障臂而保持站立**）。Hurst 提出 **「Digit 是为跌倒而设计的」**——若靠厚海绵防护电子器件，机器会肿成「米其林轮胎人」，装在身体两侧的机械臂同时承担「防摔减速支架 + 双臂操作 + 偏航摆动平衡」；跌倒控制还必须学会优先保护周围人员与手持载荷（Brewer 补充现场须明确告诫旁观工人「后退让它倒下，不要伸手扶」） |
| 2021-09-01 | *Agility Robotics' Digit Is Getting Back to Work* | Evan Ackerman（专访 Jonathan Hurst & 时任 CEO Damion Shelton）；IEEE Spectrum | <https://spectrum.ieee.org/digit-agility-robotics> | ① Hurst 阐明双臂形态学起源：最初为解决 Cassie 抬腿摆动带来的**偏航（yaw）无控制力矩**问题（反作用轮太重，单尾多用于俯仰，鸵鸟翼型**左右对称双尾**既提供偏航角动量、又兼顾跌倒护盾与操作末端——「不是为了模仿人类外形，而是物理最优解的收敛」）；② Shelton 说明商业路线收敛：**放弃**与 Ford 合作的户外自动驾驶送货上门（last-mile delivery 太复杂且商业 ROI 周期过长），转向半结构化仓库与物流中心的托盘/料箱搬运（与专机比，Digit 优势在于一机多用、不闲置以及可用人类垫脚梯等人工工具） |
| 2019-02-26 | *Building Robots That Can Go Where We Go* | Jonathan W. Hurst；IEEE Spectrum 封面专题 | <https://spectrum.ieee.org/building-robots-that-can-go-where-we-go> | 完整追溯 OSU Dynamic Robotics Lab 物理基础：受珍珠鸡无感知突踩纸蔽深坑启发，建立**弹簧–质心（Spring-Mass / SLIP）被动动力学**设计哲学。对比三代原型：**ATRIAS**（OSU/CMU/UMich 合作，`72.5 kg`，四连杆+玻璃纤维弹簧，`COT = 1.13` 接近人类 `0.2` 量级对比传统液压机器）→ **Cassie**（双腿各 **5 个电机 + 2 个被动弹簧**，`31 kg / 68 lb`，行速 `5 km/h`，功耗 `100–300 W`，5h 续航）→ **Digit**（在 Cassie 下肢基础上增加集成激光雷达感知的直立躯干与双臂） |

---

## 三、Insights / Executive Perspectives / Solutions 技术与商业条目详录

### 1. Sim2Real、WBC Motor Cortex 与三层 AI 架构

| 发布日期 | 标题与作者 | 链接 | 核心技术细节与数字 |
| --- | --- | --- | --- |
| 2020-10-23 | *Our Approach to Merging Machine Learning with Engineered Locomotion*（Jonathan Hurst） | <https://www.agilityrobotics.com/content/our-approach-to-merging-machine-learning-with-engineered-locomotion> | 早期 ML 结合方法：围绕降阶模型与周期时钟奖励（clock-based periodic rewards）、步态库（gait library）与力曲线优化，**用代价函数（cost function）而非僵化参考轨迹**定义双足运动目标 |
| 2021-04-20 | *How Full-Stack Autonomy Helps Digit Navigate Indoor Spaces* | <https://www.agilityrobotics.com/content/how-full-stack-autonomy-helps-digit-navigate-indoor-spaces> | 无需预先建图或光学/数字地标，靠机上感知在未见过的室内环境自主导航 |
| 2024-08-12 | *Explained: Data Collection & Humanoid Robots*（CTO Pras Velagapudi） | <https://www.agilityrobotics.com/content/explained-data-collection-humanoid-robots> | 定义部署现场 **三层数据飞轮**：① 原始环境感知（相机 / LiDAR / 接触与惯导，需脱敏隐私保护）；② **100–1000 Hz** 低层高频响应数据（足端落点、手部接触力与控制误差对照）；③ 任务级 **Telemetry**（成功/失败率统计，如 400 次抓取失败 5% 下钻定位破损料箱等长尾异常） |
| 2024-10-31 | *Crossing the Sim2Real Gap With NVIDIA Isaac Lab*（CTO Pras Velagapudi） | <https://www.agilityrobotics.com/content/crossing-sim2real-gap-with-isaaclab> | **核心工程教训：** Digit 具有 **28 DoF**；早期 Isaac Lab 策略在仿真走得稳但在真机打滑乱晃。团队拒绝堆砌抑制滑步的「**胶带奖励（duct-tape rewards）**」，花费 **6 个月**排查物理差异根因：① 脚尖踏板（toe-plate）触地碰撞几何简化；② 电机/传动系统**能量传播**建模失真；③ 脚趾/跗骨推杆构成的**闭链运动学约束求解稳定性**；④ 并改用 **4 接触点脚底设计**替代传统鞋底纹路，在 MuJoCo 与 Isaac Sim 上交互验证，使最终奖励精简回能耗与对称性等直观项 |
| 2025-08-25 | *Agility Robotics Powering the Future of Robotics with NVIDIA Jetson Thor* | <https://www.agilityrobotics.com/content/agility-robotics-powering-the-future-of-robotics-with-nvidia-jetson-thor> | 宣布自第 4 代起迁移算力至 NVIDIA Jetson Thor（后续 Digit 5 进一步标准化至 IGX Thor），统一训练仿真到机上端侧 AI 推理管线 |
| 2025-08-28 | *Training a Whole-Body Control Foundation Model* | <https://www.agilityrobotics.com/content/training-a-whole-body-control-foundation-model> | **Digit 「Motor Cortex」：** **小于 1M 参数的轻量 LSTM 网络**，在 **NVIDIA Isaac Sim** 中于 **3–4 天内累积数十年等效仿真时间**，**纯仿真训练、零样本迁移真机**。三大关键设计选择显著区别于学界常规人类动捕模仿方案：<br>1. **全工作空间均匀随机采样**（vs GMT/TWIST 依赖 AMASS/LAFAN1 离线人运动库导致的末端极端位姿稀疏）；<br>2. **根位置跟踪（Root Position）**（vs VMP/AMOR/FALCON/ExBody/ExBody2/HumanPlus/GMT/R2S2/MHC 等纯速度跟踪，需机上浮动基座里程计，对标 CLONE/OmniH2O/HOVER，外力推挤后能自主回位并主动迈步）；<br>3. **自由空间 SE(3) 任务空间提示**（双手与躯干位姿/朝向，而非关节角构型空间提示），便于与状态机、3D 开放词表检测器（GTC 2025 杂货购物演示）、Gemini 研究预览或模仿学习直接对接 |
| 2025-10-02 | *Digit's Next Steps* | <https://www.agilityrobotics.com/content/digits-next-steps> | GXO 全职商用一周年升级：重写导航栈，求解**全局最优最小步数路径（minimum-step path）**并由 **MPC** 精准追踪；新增下游拥堵时的**动态料箱暂存/回取策略**，减少原地调整步数并提升重负载狭窄通道吞吐 |
| 2026-03-16 | *Agility and AI* | <https://www.agilityrobotics.com/content/agility-and-ai> | 明确 **3 层软件控制栈**：① **Cognition**（最慢层：LLM/VLA 语义与任务拆解）；② **Skills**（实时中层：VR 头显 + 双手柄遥操作采集相机/关节/力/末端位姿的 **Learning from Demonstration** 技能策略）；③ **Controls**（亚毫秒快环：Isaac Lab + MuJoCo 训练的 RL WBC Foundation Model 物理直觉层）。文中亦超链接 Isaac ROS `nvblox` 场景重建 |

### 2. 安全演进、Digit 5 整机规格与标准制定

| 发布日期 | 标题与来源 | 链接 | 核心事实与规格 |
| --- | --- | --- | --- |
| 2025-11-24 | *Beyond the Hype* | <https://www.agilityrobotics.com/content/beyond-the-hype> | 披露 Digit 成为首个获得 FCC 认证并通过 **OSHA 认可国家测试实验室（NRTL）**现场安全检验的工业人形工作站（Digit 4 在物理隔离工位围栏内作业） |
| 2026-06-22 | *Built for the Real World*（Pras Velagapudi + Kevin Reese） | <https://www.agilityrobotics.com/content/built-for-the-real-world> | **安全四代演进：** Digit 3（外部 E-stop + 外置安全控制器，断联易跌倒）→ Digit 4（**机载安全 PLC + Category 1 Stop 停机 + 控制吊舱 pendant + 围栏工作站 workcell**）→ **Digit 5（协作安全 Cooperative Safety）**。牵头或核心参与 **ISO TC 299 WG12（ISO 25785-1）**、**ANSI/A3 R15.08（TR R15.108）** 动态稳定工业移动机器人标准，以及 AI 安全标准 **ISO/IEC TS 22440**（可量化 AI 失效检测 + 上线漂移监控），并对接将于 2027 年初取代机械指令的欧盟 **EU Machinery Regulation 2023/1230** |
| 2026-09-15 / 10-23 | *Digit 5 产品页* & *Jonathan Hurst Digit 5 工程 AMA* | [Digit 5 页](https://www.agilityrobotics.com/solutions/digit-5) / [Hurst AMA](https://www.agilityrobotics.com/content/jonathan-hurst-answers-your-questions-about-building-digit-5) | **整机与关键工程参数（已公开，Product Sheet PDF 需留资下载）：**<br>• 整机重量 **129 kg**（含协作安全传感器防护罩、孤岛计算单元带来的质量代价），负载 **23 kg (50 lb)**，臂展高度 **2.2 m (7.2 ft)**；<br>• 运行/充电比 **10:1**，每日 **20 小时/24 小时**作业；**9 分钟**极速快充（合作 **Nyobolt**，且按高温使用 1 年衰减后容量设计）；<br>• 驱动与力学：**自研摆线传动（cycloidal transmissions）**替代低减速比行星 QDD 以支撑 50 lb 负载并降低齿轮摩擦、配合电流环实现透明力矩控制；采用**分级冲击柔顺架构（scaled impact）**：小块柔顺脚跟吸收初撞 → 低惯量小踝关节电机 → 缓冲并加速膝关节大功率电机；<br>• 下肢与末端演化：**Digit 3** 叶簧（leaf springs，利于跑步缓冲但站立操作动力学复杂）→ **Digit 4** 刚性连杆（面向 loco-manipulation）→ **Digit 5** 接近人类比例腿型配置以兼顾下蹲低位举重与冲压铸造/钣金等可制造性（DFM）大改版；坚持**可快换工装夹爪**，认为高复杂五指灵巧手在工业跌倒耐受与成本上面临取舍；<br>• 安全计算：**NVIDIA IGX Thor + NVIDIA Halos** 安全框架内建物理孤岛安全计算机（取代独立黄色安全盒），配合 AI 人体检测与灯光/声音意图提示（所有停机决策完全本地化、不依赖 WiFi/云端 `Agility Arc`）；<br>• 市场与交付节奏：商业部署定位以双足稳态适应狭小过道与货架交互，探索团队亦规划面向家用静态稳定场景的轮式路线 |

### 3. 商业部署流程（CAP → RaaS）与 2026-11 Nasdaq 上市里程碑

| 日期 | 里程碑 / 文档 | 链接 | 可核查业务与财务口径 |
| --- | --- | --- | --- |
| 2024-06-05 → 2025-11-20 | GXO Logistics Flowery Branch（GA）商用 RaaS | [2024-10 GXO RaaS](https://www.agilityrobotics.com/content/digit-deployed-at-gxo-in-historic-humanoid-raas-agreement) / [2025-11 10万箱](https://www.agilityrobotics.com/content/digit-moves-over-100k-totes) | 从固定围栏 AMR 卸箱上输送线跑通 **RaaS**；2025-11-20 宣布在单厂累计搬运超 **100,000 个料箱（totes）**；截至 2026-11 已扩展 Schaeffler、Toyota Motor Manufacturing Canada（TMMC），并签约 2027 年初新增 2 座 GXO 美国设施与 AGCO 试点 |
| 2026-08-04 | *Agility's Humanoid Deployment Process*（Pras Velagapudi） | <https://www.agilityrobotics.com/content/agilitys-humanoid-deployment-process> | 披露正式 **Customer Acceleration Program（CAP）** 三阶段：① **POT（Proof-of-Technology）**：现场需求筛选（排除更适合 AMR/固定机械臂的伪需求），在 Agility 己方场馆 1:1 仿真与实体复现微调；② **POC（On-site Validation）**：工程师驻场开箱联网接入 **Agility Arc** 车队管理云与安全围栏，开展客户运维培训与 NRTL 验收；③ **Operational Impact（90 天产线试运行）**：量化节拍、OEE/开机率并转入 **RaaS** 多年期订阅（SEC S-1 曾披露典型 CAP 试点费约 50 万美元，详见 [robot-fde 核查](./robot-fde-delivery-primary-evidence-20261119.md)） |
| 2026-11-04 / 2026-11-05 | 与 Churchill Capital Corp XI 完成合并并于 Nasdaq 交易 | <https://www.agilityrobotics.com/content/agility-becomes-first-u-s-publicly-listed-pure-play-humanoid-robotics-company-begins-trading-today-on-nasdaq-as-aglt> | 首家美国公开上市的纯人形机器人企业（**Nasdaq: AGLT / AGLTW**）：① 合并及融资总募集约 **4.65 亿美元**（交易费用前），含 **Foxconn 领投 PIPE** 与交割前 1 亿美元股权融资；② 披露近 3 年商业运营累计超 **65,000 小时**；③ **Digit 5 获逾 3 亿美元多年期承诺订单**（明确附带前瞻性声明限定：**受限于合同里程碑、产品功能与规格验收达成**；目标 **2027H1 早期客户试用、2027 年底全面商用**）；④ 美国本土设施含 Salem RoboFab（**70,000 sq ft**，设计峰值年产 **10,000 台**）、Fremont（**60,000 sq ft** 硬件与物理 AI）及宣布扩展的 Pittsburgh **50,000 sq ft**；另 3 家付费制造试用客户与 30+ 潜在客户管线 |

---

## 四、对本知识库的映射建议

1. 新建公司主实体页 [`wiki/entities/agility-robotics.md`](../../wiki/entities/agility-robotics.md)（`project_id: agility-robotics-company`）：承接 `Research + Analysis` 7 篇文献、ATRIAS→Cassie→Digit 1–5 机械动力学演化、安全 PLC/Halos 与 ISO 协作安全框架、CAP/RaaS 商业闭环与 Nasdaq 上市口径。
2. 新建技术专页 [`wiki/entities/agility-wbc-foundation-model.md`](../../wiki/entities/agility-wbc-foundation-model.md)（`project_id: agility-wbc-foundation-model`）：提炼闭源体系下最具可借鉴价值的工程方法——Isaac Lab 六个月根治闭链脚尖撞击 Sim2Real 偏差、`28 DoF` 到 `<1M` LSTM Motor Cortex 的全空间采样/根位置跟踪/自由空间 SE(3) 接口选择、以及全局最小步数 MPC 混合导航栈。
3. 联动更新 [`wiki/entities/paper-physical-ai-mechanical-hardware.md`](../../wiki/entities/paper-physical-ai-mechanical-hardware.md)、[`wiki/overview/notable-commercial-robot-platforms.md`](../../wiki/overview/notable-commercial-robot-platforms.md)、[`wiki/comparisons/robot-foundation-model-company-paths-2026.md`](../../wiki/comparisons/robot-foundation-model-company-paths-2026.md) 及 [`roadmap/depth-sim2real.md`](../../roadmap/depth-sim2real.md) / [`roadmap/depth-loco-manipulation.md`](../../roadmap/depth-loco-manipulation.md) / [`roadmap/depth-humanoid-hardware-design.md`](../../roadmap/depth-humanoid-hardware-design.md)。
