# From Passive Perception to Active Interaction: A Survey of Affordance Learning for Embodied AI

> 来源归档（ingest）

- **短名：** Affordance Learning for Embodied AI Survey
- **类型：** paper / survey
- **预印本：** [Preprints.org 202610.0392](https://www.preprints.org/manuscript/202610.0392)
- **项目主页：** <https://jingliangli.com/affordance-survey/>
- **论文与数据集清单：** <https://github.com/hq-King/Awesome-Affordance-Learning>
- **版本：** v1；2026-10-07 发布，未经过同行评审
- **作者：** Gen Li、Hanqing Wang、Jingliang Li、Yifan Han、Jindou Jia、Tao Lin、Yuanzhe Liu、Yutong Wang、Bo Zhao、Fangqiang Ding、Anh Nguyen、Laura Sevilla-Lara、Huazhe Xu、Gregory S. Chirikjian、Marc Pollefeys、Oier Mees、Hui Xiong、Jianfei Yang
- **机构：** Nanyang Technological University；HKUST (Guangzhou)；Shanghai Jiao Tong University；University of Illinois Urbana-Champaign；The University of Sydney；University of Liverpool；The University of Edinburgh；Tsinghua University；MBZUAI；ETH Zurich
- **一句话说明：** 将具身 affordance 学习统一为感知（能做什么、在哪里）、推理（当前任务与本体下哪些机会相关）和 affordance 引导动作（如何变成计划与策略）三部分，梳理从被动视觉识别走向动作接地学习的路线。
- **核对范围：** Preprints.org 抓取返回 403；改由项目主页链接的作者托管全文 PDF（<https://hq-king.github.io/Awesome-Affordance-Learning/Survey__Affordance_Learning_in_Embodied_AI.pdf>，pdftotext 提取）通读摘要、第 1-7 节正文、图表 1-4 的文字内容与结论；参考文献列表（[1]-[234]）仅核对与正文引用的对应，未逐条核对原文，无独立附录。
- **通读：** 已完成

## 核心摘录

### 三个互补角色

1. **Affordance perception：** 识别可执行的交互及其空间位置，可表现为部件、区域、mask、heatmap、keypoint 或 3D 位置。
2. **Affordance reasoning：** 根据任务、环境上下文和 agent 状态筛选相关交互机会，涵盖关系建模、意图理解及可行性判断。
3. **Affordance-guided action：** 将接地的 affordance 表征转为动作原语、规划或策略，也可把 affordance 作为显式输入、隐式表征或优化信号。

### 领域演进

综述概括五条变化：由密集标注转向多样证据；由 2D 图像平面转向 3D 度量空间；由静态区域转向接触与后续动作动态；由封闭类别转向开放词汇泛化；由独立模块转向在 diffusion / VLA 策略中集成。

### 资源价值与边界

- 配套清单整理 affordance 相关论文、数据集与基准，适合作为持续维护的检索入口。
- 本文是综述，没有提出新的策略或机器人实验结果；不能将代表方法表当作统一协议下的性能排名。
- 具体方法、数据集和基准以论文正文及清单原条目为准；本文的 taxonomy 是阅读坐标，而非唯一分类。

## 全文核对笔记（作者托管 PDF，已通读）

- 问题形式化：F = fθ(o, l, c) = (a, r, η)，a 为动作/affordance 标签，r 为接地区域，η 为执行相关规格（6-DoF 位姿、动作分数、运动方向、接触后轨迹）；l、c 可选。分层执行写作 u = gexec(F, Crobot)，策略集成写作 u = πϕ(o, l, p, F)。
- 范围与规模：纳入 affordance 在问题定义、监督、模型输出、规划或评估中起显式作用的工作；统计至 2026-07 共 193 篇，其中感知与推理类占 68%，说明与物理执行的衔接仍是开放问题。
- 五条演进（第 2.2 节）：密集标注到多样证据；2D 到 3D；静态区域到动作动态；封闭集到开放集（开放集方法占比 2025-2026 为 42/57，即 74%，2023-2024 为 14/35）；模块化流水线到集成策略。
- 感知四视角（第 3 节）：视觉（2D）、空间（3D 点云/网格/高斯）、交互驱动（HOI 视频与图像）、可泛化（示例迁移与开放集）。表 1 比较 33 个代表方法。
- 推理三范式（第 4 节）：关系型（Markov 逻辑网络、图网络）、语言中心（SayCan、AffordanceLLM、RoboPoint 等；含序列式 SeqAfford）、智能体式（预定义工作流如 A4-Agent、AffordBot；自适应工作流如 A3R、A-Harness）。
- 引导动作（第 5 节）：分层（原语式、检索式、规划式）与策略集成（显式输入、隐式表征含潜变量条件与辅助监督、优化信号含奖励/过滤/目标）。
- 数据与评估（第 6 节）：表 4 列代表数据集；指标分热图/分布（KLD、SIM、NSS、MAE、MSE）、分割（IoU、gIoU、cIoU、P@50、ROC-AUC、mAP、aIoU）、关键点与运动（欧氏误差、ADE、DTW）三类，并明确不覆盖推理与操作成功率指标。
- 挑战（第 7 节）：表示与动作集成（需在匹配条件下对比显式/潜变量/辅助/优化信号，含无/预测/oracle affordance 对照）；可扩展学习；可靠自主接地；闭环建模与规划；具身评估与安全。文中指出辅助预测变好不一定带来更高执行成功率。
- 边界：本文为综述，不提供统一协议下的性能排名；表中会标注 Venue & Year（含 2026 年会议），以原表为准。

## 对 wiki 的映射

- 主页面：[具身 AI Affordance 学习综述](../../wiki/entities/paper-affordance-learning-embodied-ai-survey.md)
- 交叉：复用已有 affordance、机器人操作、VLA 与具身感知页面；不为综述列出的每篇代表作另建空壳。

## 来源核查

- 预印本页面标记为 Review，v1 于 2026-10-07 发布且未经过同行评审。
- 项目主页提供可浏览的综述框架与交互式文献索引；配套 GitHub 仓库自述为论文、数据集和基准的策展清单。
