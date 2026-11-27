# RoboVista: Evaluating Vision Language Models for Diverse Robot Applications

- **作者：** Shuangyu Xie, Kaiyuan Chen, Ziyang Chen, Simeon Adebola, Yixuan Huang, Zehan Ma, Tianshuang Qiu, Wentao Yuan, Dhruv Shah, Pannag R. Sanketi, Ken Goldberg
- **机构：** University of California, Berkeley; Princeton University; Google DeepMind
- **版本 / 载体：** `arXiv:2607.04610v1` [cs.RO]（2026-07-06 提交；arXiv 备注 Accepted to RSS 2026）；arXiv HTML 全文 + 项目页标题核对
- **原文：** [arXiv:2607.04610 (HTML)](https://arxiv.org/html/2607.04610) / [arXiv:2607.04610 (Abs)](https://arxiv.org/abs/2607.04610) / [项目页](https://berkeleyautomation.github.io/robovista/)
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，项目页仅返回标题页，论文正文与项目页未提供 RoboVista 数据集或评测脚本的官方代码仓库链接；正文中唯一出现的 GitHub 地址 `https://github.com/open-h-embodiment/data-collection` 属于参考文献 [26] 手术清创源数据集，候选比对仓库 `open-h/open-h-embodiment` 对应另一篇论文 `arXiv:2604.21017`，均非本论文官方仓库）
- **核对范围：** 已通读 `arXiv:2607.04610v1` HTML 全文（Abstract、§I–§VII、图注 Fig. 1–Fig. 11、正文表 Table I–Table IV、附录 Appendix A–F 及 Table V–VIII）；核对项目页静态标题 `RoboVista | Evaluating Vision-Language Models for Diverse Robot Applications`
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [RoboVista: Evaluating Vision Language Models for Diverse Robot Applications](../../wiki/entities/paper-agentic-2607-04610.md)
- **核心问题：** 真实世界机器人应用（农业、工业制造、医疗手术、家庭、自动驾驶等）跨多种本体构型与复杂视觉环境，且往往缺乏大规模遥操作端到端轨迹数据；现有基于大规模示教数据的 VQA 基准容易偏向桌面操作并遗漏模块化决策节点。
- **RQA 框架（§III）：**
  - 将机器人系统抽象为四个功能层（Fig. 2）：感知（Perception）、高层决策（High-level decision making）、运动/动作感知（Motion / action awareness）与失败恢复（Failure recovery）。
  - 模块四元组抽象（式 1）：$\mathcal{M} = (\mathcal{E}, \mathcal{X}, \mathcal{U}, \mathcal{C})$（本体配置、状态空间、任务输出空间、可行性约束）。
  - Robot-VQA 五元组数据结构（定义 1–3）：$\mathcal{Q} = (\mathcal{V}, q, a^{\star}, \mathcal{A}, r)$，视觉输入受机器人视角约束 $\mathcal{V} \in \mathbb{V}_{\text{rv}} \cup \mathbb{V}_{\text{ob}}$（机器人可见或机载相机），其中候选答案 $|\mathcal{A}| = 5$（A–E），$r$ 为专家人工撰写的推理解释。
  - 映射机制（式 2）：由领域专家人工完成 $(\mathcal{E}, \mathcal{X}, \mathcal{U}, \mathcal{C}) \to (\mathcal{V}, q, a^{\star}, \mathcal{A}, r)$，确保问题仅凭给定图文即可单义作答并与原模块输出一致，再经独立专家多轮质量复核。
- **RoboVista 基准规模与构成（§IV, Fig. 4, Appendix A Table V）：**
  - 共 **474** 道五选一多选题、**730** 张图像（平均 1.54 张/题；单图 355 题占 74.9%、2 图 66 题占 13.9%、3+ 图 53 题占 11.2%），来自 **62** 个发表文献/数据集来源。注：摘要与 §IV 记为 **39** 种独特任务类型，Appendix A Table V 统计表记为 **33** 个独特任务。
  - 六大领域分布（Table V）：开放数据集（150 / 31.6%，含 Agibot 14、DROID 69、Open X-Embodiment 67）、工业制造（144 / 30.4%，含 1D 可变形物体 57、料箱抓取 43、装配 19、3D 可变形物体 15、缺陷扫描 10）、农业（62 / 13.1%，除草 28、园艺 18、植株巡检 16）、家庭（52 / 11.0%，居家整理 49、衣物操作 3）、手术机器人（46 / 9.7%，打结 30、清创 16）、自动驾驶（20 / 4.2%，Mcity 20）。
  - 能力维度分布：Fig. 4 二分为感知 318 / 规划 156；Table V 细分为场景理解 317（66.9%）、高层决策 79（16.7%）、底层运动感知 68（14.3%）、恢复/重规划/鲁棒性 9（1.9%）、物理交互 1（0.2%）。
- **主要实验结论（§V–VI, Appendix B–E）：**
  - **Zero-Shot 评测（Table I）：** 随机基线 20.0%，纯文本 Qwen 3-8B 25.1%。闭源模型中 Gemini 2.5 Pro 总体最高（**56.5%**，手术 76.1%、农业/工业均 48.4%），GPT-4o 49.6%，GPT-5 48.1%；开源模型随规模上升，Qwen 3-235B-A22B 达 **51.3%**，Qwen 3 VL-32B 49.2%，机器人专用模型 RoboBrain 2.5-8B-NV 为 47.0%、RoboBrain 2.5-8B 为 45.8%、Robo2VLM-ER 为 42.6%。农业与工业整体准确率显著低于家庭与手术领域。
  - **Chain-of-Thought（Table II & Appendix C）：** CoT 呈现不对称效应——在多数模型上**持续拉低场景理解/感知准确率**（如 Qwen 2.5 VL-3B 下降 11.9%、Qwen 2.5 VL-7B 下降 11.3%、Robo2VLM-ER 下降 8.8%），但在部分模型的多步规划任务上有所提升（如 Qwen 3 VL-32B 规划 +8.3%、Qwen 2.5 VL-72B +5.1%）。
  - **In-Context Learning（Table III）：** 即使提供同域同任务的 1-shot 示例，所有 8 个评测模型准确率反而全线下降（-1.7% 至 -6.5%），且校准误差（Calibration Error, CE）全部上升（+3.1% 至 +9.7%），表明长上下文空间示例易诱发过自信与幻觉。
  - **失败归因（§V-D）：** 多数错误源自视觉感知而非逻辑推理。Qwen 2.5-VL-7B 上误识别（misidentification）达 143 例（30.2%）；扩增至 Qwen 3-235B-A22B 后正确率提升（36.5% → 48.5%）、误识别降至 20.3%，但空间推理与任务语义误判依然突出。
  - **物理/闭环机器人关联实验（§VI, Table IV, Appendix D–E）：**
    1. ALOHA 双臂夹爪对齐（5 种几何先验 × 5 构型，间距 10–40 cm）：RoboVista 得分与最终位置误差呈强负相关（Pearson $r=-0.78$，Spearman $\rho=-0.93$），与距离估计误差呈负相关（Pearson $r=-0.70$，Spearman $\rho=-0.75$）。
    2. dVRK 共享自主手术打结（16 阶段序列问题）：无人工干预下三模型均在第 2 题（针体空间点选）即受阻（2/16）；允许最多 3 次干预时，推进阶段数与 RoboVista-Surgery 得分高度一致（Qwen-2.5 32B 58.7% → 9/16，ChatGPT-5.0 63.0% → 9/16，Gemini 2.5 Pro 76.1% → 15/16）。
- **推理配置说明（§V vs Appendix B Table VI）：** 正文 §V 简述温度 0.7、最大补全 4096、上下文 10240；附录 B 与 Table VI 详列复现评测配置为 8× A100（80GB）+ SGLang、贪心解码 temperature 0.0、top-p 1.0、最大输出 10,240、最大上下文 32,768、图像最长边限制 720 px（JPEG quality 85）。
