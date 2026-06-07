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

## 当前提炼状态

- [x] arXiv 摘要、项目页 taxonomy 与 Key Findings 已摘录
- [x] GitHub README 环境与目录结构已对齐 [sources/repos/esi_bench.md](../repos/esi_bench.md)
- [x] §3.1 任务形式化、动作空间与 Table 1 基准对比已摘录（2026-06-07 补强）
- [x] wiki 映射：`wiki/entities/esi-bench.md`；交叉 [3d-spatial-vqa](../../wiki/concepts/3d-spatial-vqa.md)、[vision-language-navigation](../../wiki/tasks/vision-language-navigation.md)
