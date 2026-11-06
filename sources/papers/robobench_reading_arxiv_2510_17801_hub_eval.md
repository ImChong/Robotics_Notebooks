# RoboBench 论文通读（具身评测基准选型闭环视角）

> 来源归档（原文通读 · 支撑 `wiki/overview/hub-embodied-eval-benchmark.md`）

- **标题：** RoboBench: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models as Embodied Brain
- **作者：** Yulin Luo, Chun-Kai Fan, Menghang Dong, Jiayu Shi, Xiangju Mi 等（北京大学 / BAAI / 复旦 / 北科大 / 北京人形机器人创新中心），通讯 Shanghang Zhang
- **类型：** paper / benchmark / mllm / embodied-brain / evaluation
- **原文：** <https://arxiv.org/abs/2510.17801>（2026-11 通过 arXiv PDF v2，2026-07-05 版获取全文并用 pdftotext 提取）
- **一句话说明：** 沿操纵流水线把 MLLM 作为「具身大脑（System 2）」做五维评测，并以「MLLM 当世界模拟器」评估开放式长程规划；是本 hub ① 层（认知评测）的代表性一手来源，也示范了「代理指标须与下游 VLA 相关性校验」的方法学。
- **核对范围：** 通读正文 §1–§6 与附录 A–F 全部（相关工作、模型清单、人评协议、评测器鲁棒性 ICC、隐式指令 CoT 消融、标注噪声鲁棒性、与 CALVIN/LIBERO 相关性、错误分类、数据统计）；附录 G 的数据构造与评测 prompt 仅通读世界模拟器打分 prompt（Fig. 29），其余构造 prompt（视频转步骤、函数实例化等）只浏览标题。未阅读本 hub 其余基准（EWMBench、RoboDojo 等）的原文，它们不在本归档范围内。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **定位（§1）：** 现有基准要么只看执行成功，要么认知维度残缺、依赖仿真、规划评测停留在选择题 / BLEU / LLM 打分。RoboBench 沿「意图理解 → 感知 → 规划 → affordance → 失败诊断」流水线定义 5 维、14 能力、25 任务、6092 QA（4336 个条目；选择题 1895，规划 Q1/Q2/Q3 = 1973/842/1192，导航 190）。
- **数据（§3.2）：** 来自大规模真机数据（含 RoboMIND 的真实 VLA rollout 失败）与自采；高层规划失败按 REFLECT 式协议人为扰动（错物体 / 缺步骤 / 错顺序 / 错条件）；隐式指令由 MLLM 生成后人工核查；质检含多模型多数投票（全对剔除、全错人审），验证子集残余分歧 < 4%。
- **规划评测（§4）：** Q1 长程规划用 DAG 编码偏序，分数 =（NodeCorrectness + TaskCompletion）/20；NodeCorrectness 是 skill / object / parameter 精确一一匹配；TaskCompletion 由 MLLM 依首帧、参考动作与 DAG 做 rollout，统计达成的关键物体状态里程碑比例。Q2 下一步规划按 skill / object / param 打分，Q3 任务状态估计为二分类准确率。感知、affordance、失败分析为选择题准确率。
- **评测器可靠性（§5.3 / 附录 D.2、D.4）：** 与人评 Pearson r = 0.83（基线 LLM 打分 0.73）；4 个评审模型 × 3 种 prompt 的 NodeCorrectness ICC(2,k) = 0.948，但 composite 层 ICC 约 0.85–0.90；标注节点被修改 5%/10% 时排名保持，删除节点会机械抬高完成度分。
- **结果（§5，18 个 MLLM）：** 闭源平均领先开源约 20 点；隐式指令明显差于显式（最强模型 79.94 → 61.38）；感知里 robot-view（最好 53.62）与时序 grounding（最好 50.38）最弱；Gemini-3.1-Pro 感知 67.32、affordance 85.70；执行层失败诊断最好仅 43.71（人类也仅 47.30），规划层失败诊断最高 80.74。纯文本 GPT-5.4 在感知 / affordance 约 27.12 / 25.61，说明题目确需视觉证据。规划失败占比：执行类 45%、常识类 25%、识别类 24%、格式类 6%。
- **隐式指令 CoT 消融（附录 D.3）：** 先显式改写意图的 CoT 脚手架在 3 个前沿模型上无稳定增益，且结果对输出格式指令敏感；仅覆盖前沿模型。
- **与下游 VLA 的相关（§5.3 / 附录 D.5）：** 按 VLM4VLA 协议把 8 个 VLM 转成 VLA，在 CALVIN ABC-D 与 LIBERO-10 上对照。CALVIN 与 object-centric / scene-centric 感知相关 r = 0.884（p = 0.0035）/ 0.833（p = 0.0103）；LIBERO 与静态 + 动态 affordance 相关 r = 0.677，但 p = 0.0652 未达常规显著；导航 affordance 与 LIBERO 为 −0.680；低层执行误差诊断与两者相关都弱。样本仅 8 个模型，作者自述最相关的能力随下游基准而变。

## 对 wiki 的映射

- [`wiki/overview/hub-embodied-eval-benchmark.md`](../../wiki/overview/hub-embodied-eval-benchmark.md)
- [`wiki/entities/robo-bench.md`](../../wiki/entities/robo-bench.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2510.17801>
- 既有入库归档：[robo_bench_arxiv_2510_17801.md](robo_bench_arxiv_2510_17801.md)
