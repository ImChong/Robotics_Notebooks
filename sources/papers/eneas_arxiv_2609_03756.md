# ENEAS（文本提示实例跟踪与语义发现）

> 来源归档（ingest）

- **标题：** ENEAS: Embedding-guided Neural Ensemble for Adaptive Segmentation
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2609.03756>
- **项目页：** <https://speridlabs.com/research/eneas>
- **机构：** SperidLabs
- **作者：** Javier del Pino、Salvador Rodríguez、Alejandro Garabito、Javier Álvarez、Chema Garabito
- **代码：** <https://github.com/speridlabs/eneas>
- **演示：** <https://huggingface.co/spaces/speridlabs/eneas>
- **入库日期：** 2026-09-09
- **一句话说明：** 统一文本可提示的实例跟踪与语义发现：扩展 SeC 时序记忆做唯一实例跟拍，并用嵌入匹配 + 条件 VLM 验证层过滤雕像/画作/反射等「分身」误检；面向 3D 重建与无序图像集。

## 核心摘录（MVP）

### 1) 问题：SAM 3 等仍有时序幻觉与本体误分

- **摘录要点：** 文本可提示分割（含 SAM 3）在目标离屏时仍报告存在、极端近景时只分割局部纹理、把雕像/画作/反射当真实实体。ENEAS 用 **语义验证层** 区分真实例与 doppelganger。
- **对 wiki 的映射：**
  - [ENEAS](../../wiki/entities/paper-eneas.md) — 问题设定。
  - [SAM 3](../../wiki/entities/paper-sam3.md) — 主要对照基线。

### 2) 双模式统一方法

- **摘录要点：**
  - **Instance Tracking**：自然语言或点提示跟踪 **唯一实例**；扩展几何鲁棒 **SeC** 架构（原仅点交互）加文本 adapter + 时序记忆；目标离屏再入画可重识别，极端缩放保持完整掩码。
  - **Semantic Discovery**：文本查询发现 **该类全部实例**；高速视觉嵌入匹配 + **仅对歧义候选** 调用 VLM 精修，压低本体错误同时控延迟。
- **对 wiki 的映射：**
  - [ENEAS](../../wiki/entities/paper-eneas.md) — 方法。
  - [视觉–语言特征融合](../../wiki/concepts/vision-language-feature-fusion.md) — 验证层分工。

### 3) SA-Co/VEval 评测（官方 SAM 3 评测器，项目页 Table 1–2）

- **摘录要点：**
  - **Instance Tracking**：HOTA **26.70** vs SAM 3 **26.51**；AssA **90.77** vs **90.18**；TETA **17.86** vs **16.65**。
  - **Semantic Discovery**：HOTA **9.23** vs **9.19**；AssA **14.28** vs **13.85**；TETA **9.53** vs **9.10**。
  - 输入可为有序视频或 **无序图像集合**。
- **对 wiki 的映射：**
  - [ENEAS](../../wiki/entities/paper-eneas.md) — 评测读法。

### 4) 开源状态（截至 2026-09-09，项目页核查）

- **摘录要点：** **已开源** Apache 2.0。`speridlabs/eneas`：`UniqueInstanceSegmenter` / `GenericCategorySegmenter` CLI；SeC-4B 与 grounding 模型 HF 自动下载；generic 模式依赖 **Ollama** 本地 VLM。
- **对 wiki 的映射：**
  - [eneas 仓库](../repos/eneas.md)
  - [SperidLabs ENEAS 项目页](../sites/speridlabs-eneas.md)

## 当前提炼状态

- [x] arXiv / 项目页 / README 已对齐摘录
- [x] 仓库 / HF Space 已交叉核查（**已开源**）
- [x] wiki 映射：`wiki/entities/paper-eneas.md` 新建
