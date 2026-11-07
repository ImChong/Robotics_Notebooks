# ESI-Bench: Towards Embodied Spatial Intelligence that Closes the Perception-Action Loop

> 来源归档（ingest）

- **标题：** ESI-Bench: Towards Embodied Spatial Intelligence that Closes the Perception-Action Loop
- **类型：** paper
- **来源：** arXiv abs / HTML；项目页与 GitHub README 交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2605.18746>
  - <https://arxiv.org/pdf/2605.18746>
  - <https://arxiv.org/html/2605.18746>
  - <https://esi-bench.github.io/>
- **作者：** Yining Hong*, Jiageng Liu*, Han Yin, Manling Li, Leonidas Guibas, Fei-Fei Li, Jiajun Wu, Yejin Choi（* 共同一作；机构 Stanford / UCLA / Northwestern）
- **入库日期：** 2026-05-22
- **论文版本：** arXiv:2605.18746v2（2026-05-25）
- **核对范围：** arXiv v2 完整正文 §1–5、图表与实验协议，以及附录 A–Q（任务生成细节、人工核查、3D 重建失败、元认知分析、主动探索提示词、动作与步数消融）；对照项目页及官方 README/运行入口。
- **通读：** 已完成
- **一句话说明：** 在 **OmniGibson** 上构建 **10 类 / 29 子类 / 3081** 实例的 **具身空间智能** 基准，按 Spelke 四类核心知识组织；智能体须选择 **感知 /  locomotion / 操作** 并主动探索以闭合 **感知–行动环**，系统评测 SOTA **MLLM** 在被动、主动与 oracle 视图下的表现，揭示 **行动盲**、不完美 3D 重建与 **元认知** 缺口。

## 核心论文摘录（MVP）

### 1) 问题：从被动空间感知到「观察者即行动者」

- **链接：** <https://arxiv.org/abs/2605.18746>
- **摘录要点：** 空间智能在 **感知–行动环** 中展开：智能体通过行动获取观测，并推理观测如何随行动变化。仅被动处理所见无法解决 **遮挡、动力学、容纳、功能** 等；既有空间智能基准常强调被动感知或假设 **oracle 观测**。ESI-Bench 将观察者重 cast 为 **actor**，要求决定部署何种能力以及如何排序以积累任务相关证据。
- **对 wiki 的映射：**
  - [ESI-Bench（具身空间智能基准）](../../wiki/entities/esi-bench.md) — 定位与三处相对既有基准的超越（能力选择、选择性传感、消解感知幻象）。

### 2) 任务体系与仿真底座

- **链接：** <https://esi-bench.github.io/>（Task Taxonomy）
- **摘录要点：**
  - **规模：** 10 任务类、29 子类、**3081** 任务实例；场景基于 **OmniGibson**，素材来自 **BEHAVIOR-1K** 场景库。
  - **Spelke 四系统：** 物体表征、布局与几何、数量表征、目标导向行动。
  - **十类（项目页编号）：** Physical Capacity、Physical Dynamics、Specular Reflection、Perceptual Grounding、Metric Comparison、Enumerative Perception、Spatial Relations、Cognitive Mapping、Temporal Scene、Action Sequencing；每类含多子类（如 Rigid Containment、Partial Occlusion、Structural Enclosure、Long-Term Navigation 等）。
  - **共性：** 正确答案通常不出自单帧，而来自 **选择性行动 + 对行动结果的推理**。
- **对 wiki 的映射：**
  - [ESI-Bench](../../wiki/entities/esi-bench.md) — 任务 taxonomy 表与 Mermaid 感知–行动环。

### 3) 实验协议与三条主发现

- **链接：** <https://esi-bench.github.io/>（Key Findings）；arXiv 摘要
- **摘录要点：**
  - **Finding 1 — 行动盲 > 感知盲：** 主动探索显著优于被动；无显式指令时模型可自发 **move-behind / top-down / pick-up / pour-out** 等策略；oracle 视图下感知并非瓶颈（例：Gemini 3.1 在 Partial Occlusion **14.6% → 95.1%**）；被动多视角常 **加噪**（GPT-5 在 Spatial Distance **53.9% → 49.1%**）；差行动 → 差观测 → 更差行动的级联失败。
  - **Finding 2 — 3D 双刃剑：** 真值 3D + Gemini 在 Material Transparency **60.4% vs 44.0%**；**VGGT** 等不完美重建可 **低于 2D**（Geometric Configuration **9.9% vs 27.5%**），扭曲空间关系。
  - **Finding 3 — 元认知缺口：** 人类倾向 **证伪性** 视点与矛盾下修正信念；模型高置信 **过早承诺**，重复同向运动；非单靠更好感知或更多交互可闭合。
- **对 wiki 的映射：**
  - [ESI-Bench](../../wiki/entities/esi-bench.md) — 评测范式表（passive / active / oracle）与发现节。
  - [3D 空间 VQA](../../wiki/concepts/3d-spatial-vqa.md) — 补充「被动多视图 ≠ 具身主动探索」对照。

### 4) 公开资源

- **代码：** <https://github.com/ESI-Bench/ESI-Bench>
- **数据：** <https://huggingface.co/datasets/esi-bench/ESI-Bench>
- **运行：** `behavior` conda 环境；`src/active_explore` 加载 OmniGibson 场景、逐步截图、调用 GPT/Gemini，输出 `answer.json`（见 [sources/repos/esi_bench.md](../repos/esi_bench.md)）。
- **对 wiki 的映射：**
  - [ESI-Bench](../../wiki/entities/esi-bench.md) — 复现入口与依赖（OmniGibson / BEHAVIOR-1K 引用）。

### 5) 任务形式化、动作空间与四协议评测

- **链接：** <https://arxiv.org/abs/2605.18746>（§3.1、§4.1、Table 1）
- **摘录要点：**
  - 每实例 \((\mathcal{S}, p_0, q, y^*)\)；环境 \(\mathcal{E}=\langle\mathcal{S},\mathcal{A},\mathcal{O},T\rangle\)；步数预算 **\(T_{\max}=30\)**。
  - **动作空间：** locomotion（六向平移）、perception（四向转头）、manipulation（pick/put/fill/pour）、`answer(ŷ,c)`。
  - **四协议：** Passive Single-View、Passive Multi-View（随机 30 视角）、Active Exploration、Ground-Truth Passive（oracle 轨迹分离行动/感知误差）。
  - **相对 VSI-Bench / EmbodiedBench / CHAIN：** 唯一同时覆盖 **主动 L+P+M**、**隐藏空间状态** 与 **Spelke 四系统** 十类任务的具身空间基准（Table 1）。
- **对 wiki 的映射：**
  - [ESI-Bench](../../wiki/entities/esi-bench.md) — 动作空间表、基准定位表、评测范式节。

## v2 原文补读：构造链路、实验边界与版本差异

### 任务构造（论文 §3.2；附录 H、Q）

1. **Task Proposal**：围绕 Spelke 核心知识系统设计需要主动获取额外证据的空间问题；部分题目由 GPT-4o 辅助提出，不能将生成器输出直接视为经人工证实的真值。
2. **Scene Instantiation**：利用 OmniGibson / BEHAVIOR-1K 的物体状态、几何和物理配置构造具有遮挡、容量、镜像、运动与拓扑条件的实例。
3. **Agent Trajectory Collection**：采集能够揭示隐藏信息的交互轨迹，作为 Ground-Truth Passive（oracle 视图）评测上界对照；不是学习好的低层电机控制器。
4. **Metadata Saving & Human Verification**：保存问题、目标答案、场景、初始位姿、轨迹与实例元数据，并对正确性、可回答性、非平凡性做人工审核及生成偏差审计。

### 评测协议（§4.1）

| 设置 | 观察获取方式 | 能回答的问题 |
|------|----------------|----------------|
| Passive Single-View | 固定初始视图 | 被动视觉理解基线 |
| Passive Multi-View | 随机额外视角（30 张） | 更多图像是否真正带来有效信息 |
| Active Exploration | MLLM 在最多 30 步内选择感知、移动、操作，再回答 | 主动行动选择是否形成有效信息增益 |
| Ground-Truth Passive | 提供参考轨迹的有信息量视图 | 排除行动选择失误后的感知能力上界 |

### 容易误读的实验结果（§4；附录 J、K、L、M、O、P）

- **对照受限**：论文所称“多视角可能加噪”指其 **随机视角**协议，不等于所有主动选视角、多视角融合方法都无效。
- **重建误差有方向性风险**：在 Geometric Configuration，文中 Gemini + VGGT 3D 为 **9.9%**，原始 2D 基线 **27.5%**；Ground-truth 3D 在 Material Transparency 为 **60.4%**，2D 为 **44.0%**。
- **高层交互而非运动控制**：视点变换、抓取和倒水等是仿真中的抽象行动接口；不能据此宣称真实机器人低层控制、Sim2Real 或操作技能成功率已被验证。
- **人类元认知对照**：附录 K 通过反证视角、视角多样性与信念修正刻画模型过早确信的问题；统计口径与具体任务条件不可脱离附录泛化。
- **领域与复现限制**：任务在合成的 OmniGibson 环境中评测；真实世界传感噪声、接触失败、资产和 API 成本是外推限制。官方代码仓 README 仅明确 smoke test 已覆盖单步加载/渲染/API/结果写出，并非所有任务的完整多步准确率复现。

### 论文 / 项目站 / 数据目录的命名差异

- 项目站把一类称作 **Physical Capacity**、另一类称作 **Temporal Scene**；arXiv v2 Table / 附录及发布 JSON 的任务目录分别用 **Physical Structure**、**Temporal Understanding**。这两个名称对按路径运行任务的读者需要显式映射；不属于新增的独立类别。
- 官方运行器以 `src/main.py` 为 CLI 入口，注册表基于 `big_task`/`small_task` 定位 29 个任务模块；README 给出的 `--task spatial_segmentation` 是有效的小任务名，旧别名 `counting` 仍可兼容但会映射为 `category_ambiguity`。

**源码一手核对：** [`src/main.py`](https://github.com/ESI-Bench/ESI-Bench/blob/main/src/main.py)、[`src/active_explore/pipeline.py`](https://github.com/ESI-Bench/ESI-Bench/blob/main/src/active_explore/pipeline.py)、[`src/active_explore/tasks/_registry.py`](https://github.com/ESI-Bench/ESI-Bench/blob/main/src/active_explore/tasks/_registry.py)、[`models/gemini.py`](https://github.com/ESI-Bench/ESI-Bench/blob/main/src/active_explore/models/gemini.py)。
## 当前提炼状态

- [x] arXiv 摘要、项目页 taxonomy 与 Key Findings 已摘录
- [x] GitHub README 环境与目录结构已对齐 [sources/repos/esi_bench.md](../repos/esi_bench.md)
- [x] §3.1 任务形式化、动作空间与 Table 1 基准对比已摘录（2026-06-07 补强）
- [x] 2026-11-07 通读 arXiv v2 正文与附录，补全 §3.2 构造流程、§4 四协议与工程复现边界，并核对 CLI、registry、模型接口和模拟器的源码路径
- [x] wiki 映射：`wiki/entities/esi-bench.md`；交叉 [3d-spatial-vqa](../../wiki/concepts/3d-spatial-vqa.md)、[vision-language-navigation](../../wiki/tasks/vision-language-navigation.md)
