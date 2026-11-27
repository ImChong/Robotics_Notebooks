# EgoTools: Towards Tool-Centric Reasoning in Real-World Egocentric Videos

- **作者：** Shulin Tian*, Junsu Kim*, Shuai Liu*, Hao Li*★, Yujiao Shen, Sihan Li, Zhe Yang, Yeongon Kim, Feiyu Li, Jialin Wu, Yichi Zhang, Wenhui Wang, Runmao Yao, Yuhao Dong, Zhaoxi Chen★, Fangzhou Hong★, Antonino Furnari, Jingkang Yang, Hongyuan Zhu, Ziwei Liu†★
- **机构：** S-Lab, Nanyang Technological University；A*STAR；KAIST；PKU；FDU；School of Biological Sciences, Nanyang Technological University；University of Catania（硬件与标注支持：Ropedia, Inc.）
- **版本 / 载体：** arXiv:2609.39378v2 [cs.CV]（2026-10-02，v1 提交于 2026-09-30），arXiv HTML 全文（32 页、7 图、10 表、附录 A–H）
- **原文：** https://arxiv.org/html/2609.39378 、 https://arxiv.org/abs/2609.39378 、 https://ropedia.github.io/egotools
- **项目页 / 代码：** 部分开源（官方代码仓 https://github.com/Ropedia/egotools 已开源评测、数据准备与 SFT 训练脚本；数据集入口 https://huggingface.co/datasets/ropedia-ai/egotools-data 与模型入口 https://huggingface.co/ropedia-ai/egotools-8b 已建，但按 Appendix E.2 原始鱼眼流与内部预处理工具不公开，原声旁白音频、精标工具定位与 3D 轨迹分批发布；仓库 README 说明完整论文实验尚未以当前公开发布资源全量复现）
- **核对范围：** 通读 arXiv HTML 全文（Abstract、§1–§5、Table 1–5、Figure 1–6 说明、Appendix A–H 含 Table 6–10 与 Figure 7 说明）、项目页（ropedia.github.io/egotools）文本、官方仓库（Ropedia/egotools）README 与目录文件清单
- **通读：** 已完成

## 原文核对笔记

- 对 wiki 的映射：[EgoTools：真实世界第一人称视频工具中心推理套件与诊断基准](../../wiki/entities/paper-agentic-2609-39378.md)

### 1. 研究动机与数据采集
- 现有第一人称数据集（EPIC-KITCHENS-100、Ego4D、Ego-Exo4D、Assembly101、MECCANO、HOI4D）与评测基准（EgoSchema、EgoPlan-Bench、EgoThink 等）多以动作、物体或高层计划为中心，工具及其引发的目标状态变化与因果关系常被掩盖。
- EgoTools 使用 Ropedia 的 **HOMIE** 头戴式多模态采集设备（4 路同步鱼眼相机 + 音频 + 运动/同步信号），选取前左视角经校正作为规范第一人称视角（canonical view），输出 **1024×1024、20 FPS** 同步音频视频（附录 D.1 示例参数：zoom 0.78125、FOV 90.0°、pitch -5.0°、yaw/roll 0.0°）。
- 完整语料 **EgoTools-Data** 共 **646 段视频、100.37 小时**，覆盖 7 个真实场景：**Kitchen**（276 视频 / 32.37 h）、**Research Lab**（61 / 30.92 h）、**Repair Workshop**（84 / 12.34 h）、**Craft**（67 / 10.17 h）、**Classroom**（74 / 6.86 h）、**Household**（54 / 4.88 h）、**Office**（30 / 2.83 h）。
- 采集结合两类模式：**结构化任务引导（structured task-guided）**与**开放式参与者驱动（open-ended participant-driven）**；并在构建指令微调集与基准前在**源视频级别**严格切分，将 **40.34 小时**高工具密度视频保留为基准专用池，训练集与评测集零源视频重叠。

### 2. 多模态标注管线与 3D 追踪
- **层级稠密字幕（361,332 条）：** 用 Gemini-3-Flash 将视频切分为 5 分钟片段，先取 32 帧生成全局上下文，再按 5 秒块顺序生成字幕（首块用全局字幕、后续块以前一块字幕作记忆），最后聚合为 1 分钟窗口与 5 分钟总结。
- **工具中心旁白（6,519 条）：** 参与者按参考模板（`I am using [property] [tool] rather than [alternative tool] to [action] on/for [target] because [reason]`）口述/撰写工具选择、替代方案、意图及因果影响，支持英/韩/中三语，经翻译规范化为英文并记录语言标签；部分旁白附关键帧 2D 锚点。
- **3D/4D 标注管线：** 基于 HOMIE 深度/位姿与 Holi-Spatial 流程训练 **3D Gaussian Splatting (3DGS)** 渲染时空一致深度图；用旁白 2D 点初始化 **SAM2** 掩码传播，并由 VLM 周期校验语义/漂移、失败时重初始化 SAM2（递归 VLM 引导追踪）；最后将掩码提升至 3D 得到 3D 包围盒与长时程 4D 物体轨迹，用于生成空间 QA。

### 3. 指令微调语料与诊断基准构造
- **SFT 训练语料（184,679 条，仅来自非基准源视频池）：**
  - 直接对应四赛道（69,760 条，37.8%）：AC 14,175（7.7%）、PG 14,755（8.0%）、PD 22,943（12.4%）、SR 17,887（9.7%）。
  - 片段级多选（69,554 条，37.7%，含 64,554 条全局动作排序/总结/目标推理与 5,000 条配合当前帧的下一步动作预测）、通用视频 QA（31,281 条，16.9%）、稠密字幕与旁白补全（9,084 条，4.9%）、单图开放问答（5,000 条，2.7%）。
- **EgoTools-Bench（1,000 题，8 选 1 单选，随机基线 12.5%）：**
  - 由 **900 条人工编写题**与 **100 条 3D 管线生成并经人工核验的空间题**组成，每题对应局部时间窗口（多数 ≥3 分钟，平均 4.19 分钟）。
  - 四赛道题量：**Affordance & Causality (AC)** 363 题、**Perception & Grounding (PG)** 236 题、**Procedural Dynamics (PD)** 222 题、**Spatial Reasoning (SR)** 179 题。领域分布长尾，Kitchen 占 837 题，Research Lab 99 题，其余 5 域共 64 题。
  - **Fix-First 质控与去捷径：** 22 项二元结构审计、Gemini-2.5-Flash-Lite（`temperature=0`）最小保留原意修复、统一规整为 8 选项、去元选项/子串泄露/文本捷径与长度偏差检查（最终 $|z| \le 1.5$，$\bar{z} \approx -0.15$，修复早期缓存快照 $\bar{z} \approx +2.21$ 的长度偏置）及确定性选项打乱。
  - **独立信度研究（250 题分层抽样 × 2 名独立评审 = 500 判定）：** 与原答案一致率 **89.8%**（AC 86.5%），评审间一致率 **88.4%**，**94.4%** 判定为存在唯一由视觉证据支持的正确答案；5.6%（14 题）存在歧义并已人工修订/替换。

### 4. 核心实验、微调与消融结论
- **主评测（Table 2）：** 人类专家自由回看 **83.2%**（AC 82.1 / PG 85.2 / PD 83.3 / SR 82.7）；闭源 Gemini-3.1-Pro（1 fps + 音频）总分 **66.9%**，但在 **PG 仅 51.7%**（低于 AC 69.7、PD 72.1、SR 74.9）；开源 Instruct 模型（64 帧）38.9%–50.0%（Qwen3-VL-8B-Instruct 50.0%、InternVL3.5-8B-Instruct 48.7%）；开源 Thinking 模型最高 GLM4.1V-Thinking 50.7%，而 Qwen3-VL-8B-Thinking（512 帧）45.7% 低于同系列 Instruct。
- **训练效用（Table 3）：** 对 Qwen3-VL-8B-Instruct 仅全参数微调语言模型部分（冻结视觉编码器与多模态对齐器，lr `2.3e-6`，1 epoch，8 GPU 有效 batch 128，1,443 步）得到 **EgoTools-8B**：总准确率 **50.0% → 60.9%（+10.9 pp）**，其中 AC **+9.3**（46.1→55.4）、PG **+13.8**（45.9→59.7）、PD **+11.0**（57.1→68.1），但 **SR 下降 11.0 pp（55.4→44.4）**；跨基准上 EgoThink 61.0→64.0（+3.0），EgoSchema 69.0→67.6（-1.4），EgoPlan 42.3→35.2（-7.1）。
- **输入证据消融（Table 4，Qwen3-VL-8B-Instruct 帧列表协议）：** Ordered 64 frames **51.7%** > Shuffled 64 frames **49.3%** > Middle frame **46.9%** > Text-only **42.7%**。PG 最依赖视觉证据（vs Text-only +13.9 pp），PD 最依赖时序顺序（Ordered vs Shuffled +3.2 pp），SR 最依赖多帧覆盖（Ordered vs Middle +11.9 pp，Ordered vs Shuffled 仅 +1.1 pp）。
- **显式/隐式工具线索与错误重叠（Table 5、Figure 6）：** 问题显式提及工具线索比隐式推断更容易（Gemini-3.1-Pro 72.9% vs 65.2%，+7.7 pp；Qwen3-VL-8B-Instruct 53.1% vs 49.1%，+4.0 pp）；同一模型族 Instruct 与 Thinking 错误 Jaccard 重叠最高（0.63），跨模型族（Gemini Pro/Flash vs Qwen3-VL-8B）重叠仅 0.35–0.38。
- **伦理与隐私边界（Appendix B & E.2）：** 自愿无偿参与并签署知情同意，**未申请正式 IRB 伦理审查**；公开发布经脱敏与隐私复核的校正视频、字幕、修正英文旁白、1,000 题基准、切分信息，分批发布部分定位与 3D 轨迹资产，不公开原始鱼眼视频流与个人身份元数据。
