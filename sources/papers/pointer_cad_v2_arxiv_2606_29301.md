# Pointer-CAD v2: Plan-Then-Construct CAD Generation with Dimension-Aware Parametric Precision（arXiv:2606.29301）

> 来源归档（ingest）

- **标题：** Pointer-CAD v2: Plan-Then-Construct CAD Generation with Dimension-Aware Parametric Precision
- **缩写：** **Pointer-CAD v2**
- **类型：** paper / CAD program 生成 / 命令序列 / LLM / 参数精度
- **arXiv：** <https://arxiv.org/abs/2606.29301>（HTML：<https://arxiv.org/html/2606.29301>；PDF：<https://arxiv.org/pdf/2606.29301>）
- **会议：** ECCV 2026（Poster #3855）
- **代码：** <https://github.com/Snitro/Pointer-CAD-v2>（截至入库日 README 为 **Code coming soon**）
- **前作：** Pointer-CAD v1（arXiv:2603.04337）
- **作者：** Dacheng Qi, Chenyu Wang, Jingwei Xu, Yi Ma, Shenghua Gao（齐大成、王晨宇、徐经纬、马毅、高盛华）
- **机构：** 香港大学（HKU）；深圳河套研究院（Shenzhen Loop Area Institute）；忆生科技 / TranscEngram；莫纳什大学（Monash）；加州大学伯克利分校（UC Berkeley）
- **入库日期：** 2026-09-18
- **一句话说明：** 在 Pointer-CAD 逐步建模范式上引入 **Plan-Then-Construct**：先用 LLM 生成带 **公制单位** 的结构化设计计划，再通过 **指针机制** 从参数字典检索连续数值写入命令序列，消除量化误差；配套 **OmniCAD-Plan** 数据集与 **顶点 / 边 / 面** 三级几何精度指标。

## 摘要级要点

- **动机：** 现有 Text-to-CAD 与命令序列方法（DeepCAD、Pointer-CAD v1 等）多在 **归一化 + 量化** 参数空间自回归，CD/IoU 等形状指标对 **毫米级偏差** 不敏感，难以满足工业公差。
- **Plan-Then-Construct：** 每步拆为 **计划阶段**（文本域符号参数，`<>` 包裹，L/A 类型 + 单位 + 可引用先前参数）与 **构造阶段**（从计划提取长度/角度字典 → 频率编码 + 门控 MLP 嵌入 → 余弦相似度指针检索 → 执行命令更新 B-rep）。
- **相对 v1：** v1 用指针引用 B-rep 实体；v2 **直接预测连续值**，参数不再离散化到词表；计划与命令在同一次 forward 内用双解码头顺序生成。
- **数据集：** 基于 Recap-OmniCAD / Recap-OmniCAD+（含 chamfer/fillet），用 **Qwen3** 自动生成并校验 plan 级标注 → **OmniCAD-Plan**（202K 有效模型）、**OmniCAD-Plan+**（209K）。
- **评测指标：** **Vertex / Edge / Face Accuracy**（相对 GT 包围盒 min 边长 ×0.001 容差）、**RMR@3**（错误面数 ≤3 视为可修复）；对比 **Pointer-CAD**、**CADmium** 及 Qwen3/Gemini/GPT/Claude 写 CADQuery 代码。
- **骨干与训练：** 默认 **Qwen2.5-0.5B**；H800 上 10 epoch；OmniCAD-Plan 上 1.5B 模型平均精度较 CADmium-1.5B **+13.49%**，RMR@3 **91.25%**。
- **形状指标对照：** Line/Circle F1 接近饱和时，Arc F1 仍可从 51%→63.59%（0.5B）；CD 改善有限，说明 **参数精度增益未被 CD 充分反映**。

## 对 wiki 的映射

- 沉淀实体页：[`wiki/entities/paper-pointer-cad-v2.md`](../../wiki/entities/paper-pointer-cad-v2.md)
- 互链：[`wiki/concepts/text-to-cad.md`](../../wiki/concepts/text-to-cad.md)、[`wiki/entities/gencad.md`](../../wiki/entities/gencad.md)
