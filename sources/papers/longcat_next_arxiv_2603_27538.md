# LongCat-Next: Lexicalizing Modalities as Discrete Tokens（arXiv:2603.27538v1）

- **论文：** <https://arxiv.org/abs/2603.27538v1>
- **全文 HTML：** <https://arxiv.org/html/2603.27538v1>
- **代码：** <https://github.com/meituan-longcat/LongCat-Next>
- **模型：** <https://huggingface.co/meituan-longcat/LongCat-Next>
- **机构：** 美团 LongCat 团队
- **类型：** 原始论文
- **版本：** v1（2026-03-29）
- **入库日期：** 2026-11-10
- **核对范围：** 通读 [arXiv HTML v1](https://arxiv.org/html/2603.27538v1) 的 Abstract、§1–§8（含 §2 方法/公式、§3 主结果及消融、§4 数据和训练、§5 基础设施、§6 RL/局限）、§9 附录 9.1–9.6（含 Fig.18–25、Table 9 和定性图例），交叉复核官方 GitHub README 与 SFT 运行入口。审阅论文及文字/表格不等于独立运行作者实验。
- **通读：** 已完成

## 关键机制

1. **DiNA（Discrete Native Autoregression）：** 语言、视觉、音频转成离散 token，在同一个自回归 next-token 目标下训练。
2. **语义完整视觉词表：** 语义对齐编码器（SAE）和多级残差矢量量化（RVQ）在保留语义的同时编码细节，降低离散视觉建模的理解/生成冲突。
3. **dNaViT：** 原生分辨率视觉 Transformer，将输入视觉信息转换为可与语言联合建模的“视觉词”。
4. **统一理解与生成：** 图像理解对应视觉 token 条件下的文本预测；图像生成对应文本条件下的视觉 token 预测，生成端还需要视觉解码器。
5. **语言/视觉/音频统一训练：** 论文讨论多模态预训练与 SFT 组合，以及不同模态信号之间的损失与容量权衡。

## 数据与评测读法

- 报告视觉理解、视觉生成和语音多项 benchmark；“28× 压缩”是视觉表示的压缩配置，不意味着固定的图像重建质量保证。
- 必须对照同数据版本、评测协议和 model size 读分数；表中的结果是作者报告，尚未独立复测。
- LongCat-Next 采用 A3B 级 MoE（每 token 激活约 3B）；**不能**将其当成 LongCat-2.5 的 1.6T/48B 同一个模型。


## 主文实验与设计复核（§2–§6）

| 段落 | 阅读结论 | 关键边界 |
|---|---|---|
| §2.2 SAE / RVQ | Qwen2.5-ViT 作为语义对齐预编码器；RVQ 级联量化、EMA 码本更新和无效码字重置；pixel decoder + flow-matching refiner 负责解码 | “语义完整”是近似充分性目标，不代表逐像素无损 |
| §2.3 音频 | Whisper 初始化，8 层 RVQ；分层视觉代码表为 8×16384；音频分词与文本引导语音生成可采用串行/并行及随机延迟训练 | 各模态 tokenizer/detokenizer 并非无需专用模块 |
| §3.2.1 离散 vs 连续 | Qwen-7B 缩小规模实验引入 **Pre-Buffer 单层 FFN**，缓解多级 embedding 求和后的表征瓶颈；扩展训练规模缩小理解差距 | 小规模消融数值不可直接替代完整 LongCat-Next 主榜 |
| §3.2.2 信息恢复 | Table 6：随机 ViT-B/16 重建 PSNR **30.52±3.42**；预训练 ViT-B/16 **21.86±3.14** | 作者提示结构残差通路可能解释较强重建，但不能据此断言随机 ViT 的语义理解更好 |
| §3.2.3 联合训练 | Pure-Und/Pure-Gen 分别使用 100B task tokens；Unified 为 50B+50B；调平后 Unified 与 Pure-Und loss 差约 **0.006**，低于 Pure-Gen 约 **0.02** | 对齐的是论文归一化 loss 曲线，不是直接证明所有任务都互惠 |
| §3.2.4 语音 | 并行 vs 串行内部语言指导：LlamaQuestions **79.33 vs 81.67**、ReasoningQA **74.95 vs 80.30** | 并行更低延迟但语义准确率仍略低 |
| §3.2.5–3.2.6 MoE | 作者观察到模态偏好专家形成；平均专家路由 token 数从 **507.1→584.6**；t-SNE 可视化视觉/文本表征交织 | 模态专门化和共享语义空间是经验观察，不是因果定理 |
| §4 训练/数据 | 视觉 tokenizer 约 **5000 万图像**，最大 1736×1736；图像生成训练数据约 **3 亿图文对**；统一模型经历 Pre-Align→Pre-train→Mid-train→SFT | 数据含内部来源；开放 SFT 代码不足以完整重建全数据流水线 |
| §5 基础设施 | **VHalf/V-shaped pipeline** 使嵌入与模态 loss 在同一设备，减少跨阶段通信及不均匀计算造成的 bubble | 属作者分布式训练架构说明，不能当作本地实测收益 |
| §6 GRPO | 视觉生成按 RVQ 多层加权 GRPO，生成图片经冻结解码器和多维奖励评估；视觉理解采用序列熵及 actor/sampler 概率差过滤 | 这些是**视觉任务的 RL**，不是机器人动作控制 RL |

## 附录逐节复核（§9）

1. **§9.1 Training-Inference Mismatch Analysis in RL（Fig.18）**：一次异常 token 可能在采样策略下概率超过 **0.4**，但 actor 重算仅 **1e-2**；其余 token 的概率偏差约 **1e-3**。作者比较 TIS/MIS 后认为不能根治 entropy explosion，采用**序列级熵过滤 + 任一 token 概率差超阈值的整序列过滤**。不能将这种误差与平均 token KL 简单等同。
2. **§9.2 Experimental Analysis for RL（Table 9）**：使用**Qwen-7B 缩小规模实验**，视觉理解约 **30K** 样本、图像生成约 **40K** 样本。MMMU **64.22→66.45**；ChartQA **88.56→92.08**；GenEval Overall **83.94→87.33**，其中 counting **71.25→78.75**、position **74.75→81.50**。这里的增益**不能**当成 68.5B 主模型统一评测的增益。
3. **§9.3 Discrete Quantization Strategy（Fig.19）**：特征重建损失评测显示二级 RVQ 略优于普通 VQ，**八级 RVQ**进一步降低量化残差；论文据此选八级作为默认，而非无限叠级。
4. **§9.4 Vision Understanding and Generation under DiNA（Fig.20）**：理解端先逐级**相加**视觉 embedding 进 LLM；生成端隐藏状态送入 DepthTransformer 解出多级 token，在**一次 AR 步**处理多层索引；并不代表整张图只需一次 AR 步。
5. **§9.5 Qualitative Examples（Fig.21–24）**：展示数学公式/排版 OCR、数字比较、数阵逻辑与几何推理的示例；属**定性案例**，不能据此推出总体评测成功率。
6. **§9.6 Analysis of Visual De-tokenizer（Fig.25）**：ViT pixel decoder 先恢复大体结构，refiner 补高频细节；论文明确本版以**生成质量和语义一致性**为主，并非追求严格像素恢复。

## 结果核对与限制

- 主论文 Table 1 的 MMMU(val) **70.6**、MathVista(mini) **83.1**；Table 2 的 GenEval **84.44**、DPG **84.66**；Table 3 把 GenEval 写为整数 **84**。同名指标须注明表别/舍入口径。
- Table 1 中正文 OCRBench 写“865”，表格以 **86.5** 计量；归档优先采用表格口径，并显式记录文本和表中单位差异。
- 原论文 v1 的附录有丰富消融，但 **PDF/HTML 公开结果≠本地独立验证**，也不包含可从 SFT 入口复现全量 2T token 训练的前提。

## 开源结论

- 已有 [官方 GitHub](https://github.com/meituan-longcat/LongCat-Next)、[HF 模型卡](https://huggingface.co/meituan-longcat/LongCat-Next)，并有 `sft/` 训练样例，属于**代码及模型公开**。
- 本次已核对 v1 全部正文及附录章节。论文报告的数据、量化曲线、图像生成样例均是**作者结果**，尚未在本地独立执行 GPU 复现；这不妨碍原文通读完成，但不能据此声称复现实验通过。

**对 wiki 的映射：** [LongCat-Next](../../wiki/entities/paper-longcat-next.md)
