# Robot-Powered Data Flywheels 论文通读（数据飞轮视角）

> 来源归档（原文通读 · 支撑 `wiki/concepts/data-flywheel.md`）

- **标题：** Robot-Powered Data Flywheels: Deploying Robots in the Wild for Continual Data Collection and Foundation Model Adaptation
- **作者：** Jennifer Grannen, Michelle Pan, Kenneth Llontop, Cherie Ho, Mark Zolotas, Jeannette Bohg, Dorsa Sadigh（Stanford / TRI）
- **类型：** paper / data-flywheel / vlm / in-the-wild-deployment
- **原文：** <https://arxiv.org/abs/2511.19647>（2026-11 通过 arXiv PDF v1 获取全文）
- **一句话说明：** 把「部署—策展—微调—再部署」写成形式化迭代框架（RPDF），以图书馆盘点机器人 Scanford 为实例，证明野外部署数据能同时提升域内与域邻近能力。
- **核对范围：** 通读摘要、§I–§VI（框架算法 1、Scanford 系统、数据策展、模型适配、实验、讨论与局限）及参考文献列表；该文无附录。未复现实验、未核对项目页与代码。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **框架（§III，算法 1）：** D0=∅，预训练模型 FM0；第 t 轮由 FM(t−1) 驱动机器人部署（受电池等限制的最长时长 T）得到原始数据，经 Curate 得到 Dt，累加后**始终从 FM0 出发**重新微调得到 FMt。策展可人工或自动。
- **Scanford 系统（§IV-A）：** Franka FR3 + TidyBot++ 移动底盘，腕部 RealSense D435 与底座 Unitree L2 激光雷达；每站停多个预设高度拍照，底座前进 0.3 m（约半幅重叠）；里程计漂移大、SLAM 在重复几何的书架过道失效，用点云拟合两侧书架平面做启发式纠偏。
- **自动标注与策展（§IV-A/B）：** 按分类号范围缩小候选书目，用 RAG 把候选作为 VLM 上下文；再用字符串相似度（Ratcliff-Obershelp）与相邻书序一致性检查过滤，超阈值才入库。目录并不完全准确（借出、离馆），故不能开环直接当标签。
- **训练（§IV-C）：** Qwen2.5-VL 7B，5 个 epoch，AdamW，学习率 2e-7，有效批量 16，单卡 H200，数据 487–5,019 张时训练 1–7 小时。
- **数据量（§IV-D/V-A）：** 部署 6 小时得 8,232 张带标签图，策展后留 5,019 张。
- **域内结果（§V-A）：** 71 张人工标注留出集（10 个书架）：Qwen2.5 预训练 32.4%、Gemini 43.7%，微调后 71.8%（+39.4 点）。摘要/结论写作 32.0%→71.8%，正文与图为 32.4%，原文自身存在小数不一致。增益大多在前约 1.5 小时（约 1,352 张）内取得，之后趋于平台。
- **域邻近结果（§V-A）：** 取各基准最难子集——英文 OCR（644 张）Qwen 24.8%、Gemini 30.7% → 微调后 46.6%；中文 OCR（500 张）Qwen 30.8%、Gemini 3.4% → 38.0%。作者「假设」Gemini 中文差源于预训练中中文较少（推测，未验证）。
- **部署效益（§V-B）：** 10 天、每天 4 小时（共 40 小时）扫描 2,103 个书架，馆员估计节省 18.7 小时；共 26 次人工干预（日均 2.6 次，每次不到 5 分钟，多为纠正漂移）；第二周周二书架仅 3 层（常规 7 层），激光雷达定位更噪，当天干预达 12 次。
- **设计经验（§VI）：** 任务应落在当前模型与机器人能力的「最近发展区」（太易则无提升空间，太难则需大量人工标注/干预）；选非结构化、网络语料欠代表的领域；任务须对利益相关者有直接价值。
- **局限（§VI）：** 仅验证了 VLM；需要一定任务特定工程；两类任务均未达 100%，仅靠微调可能不足以保证鲁棒；未来方向包括 LLM / VLA、并入预训练混合、改进策展、把数据汇编成评测基准。

## 对 wiki 的映射

- [`wiki/concepts/data-flywheel.md`](../../wiki/concepts/data-flywheel.md)
- [`wiki/entities/paper-scanford-robot-powered-data-flywheel.md`](../../wiki/entities/paper-scanford-robot-powered-data-flywheel.md)
