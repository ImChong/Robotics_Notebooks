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

## 对 wiki 的映射

- 主页面：[具身 AI Affordance 学习综述](../../wiki/entities/paper-affordance-learning-embodied-ai-survey.md)
- 交叉：复用已有 affordance、机器人操作、VLA 与具身感知页面；不为综述列出的每篇代表作另建空壳。

## 来源核查

- 预印本页面标记为 Review，v1 于 2026-10-07 发布且未经过同行评审。
- 项目主页提供可浏览的综述框架与交互式文献索引；配套 GitHub 仓库自述为论文、数据集和基准的策展清单。
