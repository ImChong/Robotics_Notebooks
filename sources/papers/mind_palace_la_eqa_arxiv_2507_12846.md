# Enter the Mind Palace：长期主动具身问答（LA-EQA）原文归档

- **英文标题：** Enter the Mind Palace: Reasoning and Planning for Long-term Active Embodied Question Answering
- **中文标题：** 走进记忆宫殿：面向长期主动具身问答的推理与规划
- **类型：** paper / embodied-question-answering / long-term-memory / active-exploration
- **正式发表：** CoRL 2025；PMLR 305:5072–5106
- **作者：** Muhammad Fadhil Ginting, Dong-Ki Kim, Xiangyun Meng, Andrzej Marek Reinke, Bandi Jai Krishna, Navid Kayhani, Oriana Peltzer, David Fan, Amirreza Shaban, Sung-Kyun Kim, Mykel Kochenderfer, Ali-akbar Agha-mohammadi, Shayegan Omidshafiei
- **机构：** Field AI；斯坦福大学（Stanford University）
- **arXiv：** <https://arxiv.org/abs/2507.12846>（v2，2025-09-25）
- **全文 HTML：** <https://arxiv.org/html/2507.12846v2>
- **CoRL / PMLR 正式论文：** <https://proceedings.mlr.press/v305/ginting25a.html>
- **正式 PDF：** <https://raw.githubusercontent.com/mlresearch/v305/main/assets/ginting25a/ginting25a.pdf>
- **项目：** <https://mind-palace-laeqa.github.io/>
- **代码与评测：** <https://github.com/mind-palace-laeqa/benchmark>
- **入库日期：** 2026-11-17
- **核对范围：** arXiv v2 全文 §1–8；附录 A（任务形式化）、B（分层场景图、规划、VoI 与 Algorithm 1）、C（五个场景与题型）、D–F（对照方法、评测与案例）、G–I（真机任务、失败与逐题案例）；对照 PMLR 正式元数据与项目主页；官方源码另见仓库归档。
- **通读：** 已完成

## 原文要点（按论文组织）

### 1. 问题定义（§3；Appendix A）

LA-EQA = (Q, M, E, x0, A*)：问题 Q、跨时段记忆 M、变化中的当前环境 E、初始机器人姿态 x0、真实答案 A*。
记忆为多次巡检采集的位姿+图像观测；动作集包括历史检索（retrieve）、当前探索（explore）和自然语言回答（answer）。与仅靠历史单次轨迹的 EM-EQA、仅现场探索的 A-EQA 不同，要求组合不同日期的信息及当前场景。

### 2. Robotic Mind Palace 构建（§4.1；Appendix B.1）

将跨日/跨周观测按宏观时间切分成多个 world instance，每个 instance 为分层场景图。viewpoint 节点包含位置、图像、检测物体和文字 caption；根据空间/语义相近聚成 area 节点，图中保留相邻及归属关系。历史图为 G1…GN；当前图 G0 初始化为区域结构并随着实时探索更新。仿真每 3 个仿真步采样视点，实景每 3 秒采样并去除静止重复；利用 GPT-4o 生成 caption / 对象列表，实景可借助平面图分组。

### 3. 推理、规划与提早停止（§4.2–4.3；Appendix B.2–B.7）

- VLM 依据工作记忆 h 与问题决定是否已能回答；否则 LLM 识别下一目标对象/空间概念 y。
- 先选择相关 world instance（可跨多个历史时段），再以 LLM 估计区域包含目标的概率；forward-search 优化搜索代价，附录实现展望步数为 3。
- 最后选 viewpoint：若来自历史图则检索图像；若属于 G0 则导航并拍摄当前图像；写入 h 和当前世界图，循环直至足够证据。
- 在要转向当前探索时，若预测集仅剩一个区域，或新的历史检索不会改变下一步探索区域，则根据 Value of Information（VoI）为零的直觉停止继续回忆；不是把所有历史图片都读一遍。

### 4. 数据与结果（§5–6；Appendix C–I）

- 数据：Habitat HM3D 两个家庭、Isaac 大仓库、真实办公室、真实高层施工现场，合计 **3 个仿真+2 个真实场景**；150 道题，每场景 30 题，均匀覆盖 past / present / multi-past / past-present / past-present-future 五种题型。真实部分有 6 个月跨度的 11 条 30–60 分钟轨迹。
- 同样 GPT-4o 与预算的 Table 1：**Mind Palace 65.0%**、探索效率 0.45、历史检索图像均数 22.86；**Multi-Frame VLM 52.9%**、检索 100；ReMEmbR 46.1%；Active EQA w/ Frames 43.7%、探索效率 0.29。
- **Mind Palace w/ stopping**：正确率 **61.8%**、探索效率 **0.42**、检索图像均数 **15.73**。减少检索有准确率/探索效率代价，不能描述为指标完全不变。
- 真实部署：超过 1,000 平米、27 个区域的办公室腿式机器人，10 段历史巡检、7 道需要主动探索的问题。论文报告利用历史记忆平均节省 3–10 间房间搜索；未由本次运行复测。

### 5. 局限（§8；Appendix H）

- 记忆仅覆盖机器人真正巡视过的时空：办公室空工位曾因巡检采样稀疏而错误判断仍无人占用。
- VLM 检测/计数/功能与跨房间空间归属可能出错，难题须跨多对象/多次观察推理。
- 基准依赖人工设计长期场景变化、放置物体与题目，自动构造尚不成熟；实景长期巡检成本高。
- 回答正确率采用 LLM 评判；不等于端到端真实机器人全年无故障能力。

## 对 wiki 的映射

- [独立论文实体](../../wiki/entities/paper-mind-palace-laeqa.md)
- [HUMEMBR 长期记忆导航对照](../../wiki/entities/paper-humembr.md)
- [视觉–语言导航任务](../../wiki/tasks/vision-language-navigation.md)
- [导航纵深路线 Stage 5](../../roadmap/depth-navigation.md)
