# Are Frontier VLM Agents Ready to Be Robot Generalists? An Empirical Study with the Embodied Agent Arena

- **作者：** Haojian Huang, Pukun Zhao, Zexi Li, Yehang Zhang, Yangkai Wei, Wenqian Li, Han Yang, Kaiwen Zhou, Ying-Cong Chen, Yinchuan Li
- **机构：** 香港科技大学（广州）（HKUST-GZ）、香港中文大学（CUHK）、Knowin AI
- **版本 / 载体：** arXiv:2610.00854v3（cs.RO，2026-10-01 提交 v1，2026-10-23 修订 v3，73 页含附录），arXiv HTML 全文 + 官方项目页
- **原文：** [arXiv:2610.00854](https://arxiv.org/abs/2610.00854)（HTML 全文：[arXiv HTML](https://arxiv.org/html/2610.00854)）
- **项目页 / 代码：** [项目页](https://embodied-agent-arena.github.io/embodied-agent-arena/) ｜ 官方仓库：[embodied-agent-arena/embodied-agent-arena](https://github.com/embodied-agent-arena/embodied-agent-arena)（MIT 协议，**已开源**：评测 harness 代码、1,000-case 任务索引、评测结果 CSV 与分析脚本均已在 GitHub 公开；Pinned 数据集 `uuu-Quant/Embodied-Agent-Arena` 已在 Hugging Face 公开）
- **核对范围：** 通读 arXiv:2610.00854v3 全文（摘要、Sections 1–5、Tables 1–2、Figures 1–8 及全部附录 Appendices A–J、Tables A1–A46、Figures A1–A16），并核对官方项目页全文、GitHub 仓库 README 与文件列表样本。
- **通读：** 已完成

## 原文核对笔记

### 1. 研究动机与核心问题
- 核心问题：前沿视觉语言模型（VLM）智能体是否已经具备成为通用机器人智能体（robot generalists）的能力？作者指出，机器人不仅需要理解场景并执行动作，还必须在有限时间与交互预算内同时满足任务的所有终态约束条件；单一平均分（如局部预测精度或终态成功率）容易掩盖中间子目标达成后的控制失效与目标绑定错误。

### 2. 基准构建：1,000 个案例与五大能力域（Section 3.1 & Appendix A）
- 总规模：1,000 个案例（660 个固定输入案例 + 340 个交互式回合），整合 32 个既有具身/视觉来源与团队自建的 168-case 几何诊断基准 **GeoProbe**（基于 130 张主要为 Blender 渲染的 RGB 图像 + 4 对真实桌面照片，覆盖 10 种协议）。
- 五大能力域分布（Table A1）：
  1. **Geometry（370 例，37.0%）**：InFlux-Real（30，其中 4 例参考无效在连续误差计算中排除）、Map-free（30 相对位姿）、MultiSPA-derived（30）、GeoProbe（168：固定/小视角变化/大视角变化物体位移各 12、真实桌面位移 4、完整内参 24、焦距与主点居中 12、相对相机位姿 24、稀疏深度 12、物体尺寸 12、基于参考的尺度 44）、TraceSpatial-Bench（56 场景共 112 例 2D/3D 轨迹追踪）。
  2. **Spatial Reasoning（220 例，22.0%）**：3DSRBench（50，28 条 flip 标注样本按未翻转展示并修正 5 例参考答案）、MindCube（50）、VSI-Bench（50，初始给 8 帧）、RoboSpatial-Home（30）、BOP-ASK（20）、MMSI-Bench（20）。含 183 道分类题、27 道数值题、10 道点集题。
  3. **Affordance（70 例，7.0%）**：UMD（30，5 类工具、6 种功能）、ReasonAff-style（20，基于 3DOI）、RAGNet-3DOI（20）。同时测点有效性、原生边界框与直接多边形输出（无外部分割器）。
  4. **Task Planning（157 例，15.7%）**：ALFRED（20）、ALFWorld（20）、DiscoveryWorld（10）、HumanCLAW（13：5 导航 + 8 坐下任务）、ScienceWorld（73 文本实验）、VirtualHome（21 符号程序）。
  5. **Manipulation（183 例，18.3%；含 182 二值任务 + 1 连续奖励 AnymalC-Spin）**：CALVIN（10 个 5 子目标序列）、CLIPort（18）、RLBench（10）、RoboCasa（10）、RoboCasa365（10）、RoboWits（10）、RoboTwin 2.0（50）、VIMA-Bench（10）、VLABench（25）、ManiSkill（10：9 二值 + 1 连续）、LIBERO-PRO（10）、robosuite（5）、BEHAVIOR-1K（5）。

### 3. 统一极简评测框架与协议（Sections 3.2–3.3 & Appendix B）
- **极简 Harness**：保留各来源的原生观测与原生接口/辅助函数，不额外外挂感知模型或规划器模块（除 Section 4.4 / Appendix E.2 的补充多轮审查实验引入 SAM 3、SuperPoint+LightGlue、GeoCalib 之外），以评测模型的「原生能力（native capability）」。
- **运行时机制**：Geometry / Spatial / Affordance / Planning 运行在持久 Python 会话中（变量与环境状态跨轮保留，绘图或裁剪图像在轮末送入下一轮请求）；Manipulation 沿用各基准原生 agent loop，Opus 5.5 / DeepSeek-1017 / Gemini 4 Argon 在 Manipulation 中每轮自动获得最新 RGB 观测。
- **独立判分**：模型主动终止不代表成功；评测器独立检查提交结果或底层物理环境状态。API 中断、代码执行错误、超预算或超时均计为失败（留在固定分母中）。
- **评测模型（10 个）**：7 个闭源视觉服务（GPT-6 Astra、GPT-6 Sol、Claude Fable 5.1、Claude Opus 5.5、Gemini 4 Argon、Gemini 3.8 Flash、Qwen 3.8 Max [其开源仅限文本权重，视觉为闭源 API]）与 3 个开源视觉权重模型（Qwen 3.5 397B-A17B、Qwen 3.5 27B、DeepSeek-V4.1-Flash-1017）。
- **重复次数**：非追踪 Geometry、Spatial、点/框 Affordance 测 5 轮（Fable 为 3 轮）；Tracing、直接多边形 Affordance、Planning、Manipulation 为单轮。

### 4. 核心实验结果与诊断结论（Section 4 & Appendices C–J）
- **主表表现（Table 2）**：闭源模型在全部 5 个域领先；Gemini 4 Argon 在平移误差（105.6 cm）、Spatial Pass@1（71.6%）、Box-mask IoU（0.356）和 Polygon IoU（0.441）领先；GPT-6 Astra 在旋转误差（17.2°）和 Task Planning（80.3%，126/157）领先；Gemini 3.8 Flash 数值 Spatial AbsRel 最低（0.161）；Claude Opus 5.5 在 Manipulation 领先（47.8%，87/182），紧随其后的是 Gemini 4 Argon（44.0%，80/182）和 Astra（41.8%，76/182）。开源权重模型在 Spatial Pass@1 达到 48.0–51.6%，但 Planning（19.1–31.2%）与 Manipulation（9.9–20.9%）显著落后。
- **感知与执行的背离**：Qwen-Max 的 Spatial Pass@1（68.9%）接近 Astra（69.4%），但 Planning 仅 36.9%（vs Astra 80.3%）、Manipulation 仅 9.3%。在 Manipulation 中，按 13 个来源等权平均时，Gemini 4 Argon（47.4%）反超 Opus 5.5（44.5%），因为占比 50/182 的 RoboTwin 2.0 偏向 Opus（23 vs 14）。
- **细粒度误差拆解（Section 4.3）**：
  - *Geometry*：GeoProbe 深度上 Gemini 4 原始 AbsRel（0.299）与 Qwen-397B（0.312）接近，但做尺度对齐后 Gemini 4 降至 0.065 而 Qwen-397B 仍达 0.203（揭示系统性尺度偏差 vs 形状畸变）；TraceSpatial 3D 上 Gemini 4 路径 RMSE 最低（0.310 m），但 56 条路径仅 24 条（42.9%）同时满足起止点与碰撞约束。
  - *Affordance*：直接多边形预测在 9/10 模型上高于框掩码 IoU，但交互场景（RAGNet/ReasonAff）仍远低于工具部件（UMD）；Qwen-397B 常「点对但框不对」（96 次仅点通过 vs 18 次双通过），Gemini 4 出现 16 次「框过但点落出功能掩码外」。
  - *Task Planning*：157 任务中 142 个至少有 1 模型完成，15 个全部失败（含全部 8 个 HumanCLAW 坐下任务）。ALFRED 洗肥皂案例揭示部分模型会放到外层 `Bathtub` 而非 `BathtubBasin`（1/3 条件）；HumanCLAW 坐下任务中 Astra 与 Gemini 4 在 5/8 与 7/8 回合中到达 20 cm 内并调用 `sit`，仍全部失败并自报完成。
  - *Manipulation*：100 个 CALVIN 回合中 60 个达成第 1 子目标，仅 9 个完成全部 5 步；65 个出现代码错误的回合中有 62 个恢复执行无错代码，但仅 3 个最终通过任务验证器。ManiSkill 中 Opus 5.5 完成 PickCube/PlaceSphere/StackCube，但在 PegInsertionSide 与 PlugCharger（要求 ≤5 mm 且 ≤0.2 rad）全败；DeepSeek-1017 在 PlaceSphere 悬停误差仅 2.8 mm 却 `is_grasped=false`。
- **反馈、审查与计算成本（Section 4.4 & Appendix J）**：
  - 自审/裁剪审查在修复坐标越界（6/9 越界框全部修回图像内）的同时，也可能把原本正确的空间关系改错（Qwen-27B 18 例空间题自审改对 1 错 2、裁剪改对 2 错 4）。
  - 失败回合消耗了 Planning 总 Token 的 55.5%–94.3%（Astra 31 个失败任务占 55.5% Token，单次失败均值 591,162 vs 成功 116,435，为 5.1 倍；其中 HumanCLAW 占 Astra 失败 Token 的 81.7%）；且在 7 条 HumanCLAW 轨迹中，98.23%–99.46% 的 Token 来自重复上下文输入的 prompt token。

- **对 wiki 的映射：** [Are Frontier VLM Agents Ready to Be Robot Generalists? An Empirical Study with the Embodied Agent Arena](../../wiki/entities/paper-agentic-2610-00854.md)
