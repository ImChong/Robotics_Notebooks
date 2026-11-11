# 《大模型基础》 / Foundations of Large Language Models（浙大 DAILY Lab）

> 来源归档（book · 官方教材）

- **标题：** 大模型基础（Foundations of Large Language Models）
- **类型：** book / open textbook
- **作者：** 毛玉仁、高云君等（教材扉页）；高云君为编撰总指导
- **机构：** 浙江大学数据库与大数据分析实验室（DAILY Lab）
- **官方仓库：** https://github.com/ZJU-LLMs/Foundations-of-LLMs
- **原文 PDF：** https://github.com/ZJU-LLMs/Foundations-of-LLMs/blob/main/%E3%80%8A%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80%E3%80%8B%E6%95%99%E6%9D%90/%E5%A4%A7%E6%A8%A1%E5%9E%8B%E5%9F%BA%E7%A1%80%20%E5%AE%8C%E6%95%B4%E7%89%88.pdf
- **英文版 PDF：** https://github.com/ZJU-LLMs/Foundations-of-LLMs/blob/main/Foundations_of_LLMs%28English_version%29/Foundation_of_LLMs.pdf
- **核对日期：** 2026-11-11
- **核对范围：** 对照官方仓库 README 与中文教材全文可检索镜像（约 283 页），按六章及各小节的目录顺序复核正文、关键公式、图题、样例、章节结论和参考文献入口：第 1 章 §1.1–1.5、第 2 章 §2.1–2.6、第 3 章 §3.1–3.5、第 4 章 §4.1–4.5、第 5 章 §5.1–5.5、第 6 章 §6.1–6.5。镜像地址为 https://library.qiangtu.com/download/1108/pdf/1108.pdf；已与官方 PDF 的书名、署名、六章结构及主要内容相互核对，但**官方文件二进制版本与镜像的逐字同一性未确认**。
- **通读：** 已完成
- **一句话说明：** 面向大语言模型读者的六章开放教材，将语言模型基础、架构、提示、参数高效微调、模型编辑与 RAG 串成学习路径。

## 教材身份与出版说明

- 正式英文书名以教材扉页 **Foundations of Large Language Models** 为准；GitHub 仓库 slug 是 `Foundations-of-LLMs`，不是另一部独立教材。
- 首版六章；README 期望后续按月修订并为每章维护 Paper List，但“计划月度更新”不等于已证实每月实际更新。
- GitHub 仓库创建于 2024-06-30（可通过官方仓库创建时间核实）；因此不将“2024 年 8 月首次开源”记为已核实事实。
- 资料以 PDF、分章阅读和文献策展为主；并非一个有训练脚本、模型权重和机器人部署示例的项目。

## 六章与核心脉络（按官方 README 目录）

| 章 | 主题 | 关键小节 |
|---|---|---|
| 1 | 语言模型基础 | n-grams、RNN、Transformer、采样方法、评测 |
| 2 | 大语言模型架构 | Encoder-only、Encoder–Decoder、Decoder-only、非 Transformer（含 SSM 等） |
| 3 | Prompt 工程 | In-Context Learning、Chain-of-Thought、Prompt 技巧及应用 |
| 4 | 参数高效微调 | 参数附加、参数选择、低秩适配（LoRA）与实践 |
| 5 | 模型编辑 | 编辑范式、T-Patcher、ROME、应用 |
| 6 | 检索增强生成 | RAG 架构、知识检索、生成增强、实践 |

## 六章原文阅读记录（章节与要点）

### 第 1 章：语言模型基础（§1.1–1.5）

- **统计模型：** n-grams 以 Markov 假设缩减上下文，用极大似然估计得到条件频率；第 1.1 节通过“长颈鹿脖子长”演示 bigram/trigram 的零频率与泛化折中。
- **RNN：** 递归隐状态累积历史信息，训练梯度为多步链式相乘，介绍梯度消失/爆炸与 LSTM/GRU 门控改善。
- **Transformer：** 展开自注意力、前馈层、LayerNorm、残差与 Encoder–Decoder；介绍利用 next-token prediction 构造交叉熵训练目标。
- **采样：** 区分 greedy/beam search 与 Top-K、Top-P、Temperature，并讨论重复、创造性和随机性权衡。
- **评测：** 内在指标 perplexity (PPL) 与外在 BLEU/ROUGE；展示 BERTScore、G-EVAL 等语义/生成式评估方向。书中指标与例子是教材解释，不代表作者新增 benchmark。

### 第 2 章：大语言模型架构（§2.1–2.6）

- **主干与注意力：** Encoder-only 的双向注意力偏理解；Encoder–Decoder 结合双向编码、掩码生成与交叉注意力；Decoder-only 用因果掩码自回归生成，图 2.6 对比三类注意力矩阵。
- **代表模型：** BERT、RoBERTa、ALBERT、ELECTRA；T5、BART；GPT-1/2/3、InstructGPT、LLaMA 1/2/3 及部分多模态衍生模型。表格中的参数量和发布时间仅反映教材编写时上下文。
- **非 Transformer：** SSM 连续、离散递归及卷积三种形式，训练/推理运算权衡；RWKV、Mamba 与 TTT 作为其他序列建模方向。不能将 TTT 与传统 SSM 定义等同。

### 第 3 章：Prompt 工程（§3.1–3.5）

- **基本概念：** Prompt 结构、Token 分词与词嵌入、上下文学习的 zero-/one-/few-shot。
- **ICL 变量：** 示例检索和顺序、输入—输出映射、格式、任务说明与模型规模均会影响结果。
- **推理提示：** CoT、Zero-Shot CoT、Auto-CoT、ToT/GoT、self-consistency 等，对比逐步推理、搜索回溯与多路采样；§3.4 提示避免任务模糊和上下文噪声。
- **应用：** Agent 的 Profile/Memory/Planning/Action，数据合成与 Text-to-SQL。这里的 Agent 流程是应用案例，并非教材附带可运行智能体项目。

### 第 4 章：参数高效微调（§4.1–4.5）

- **分类：** 参数附加（Prompt/Prefix/Adapter/Proxy-tuning）、参数选择（BitFit、Child-tuning）、低秩适配（LoRA）。
- **机制：** Adapter 的瓶颈映射与残差；Child-tuning 使用梯度掩码；LoRA 用低秩矩阵近似参数增量，介绍 ReLoRA、AdaLoRA、DoRA 等调优与组合变体。
- **实践：** 第 4.5 节展示 HF-PEFT 配置、LoRA 参数 `r`、`lora_alpha`、`target_modules`，以及 FinSQL 等表格领域应用；代码片段属于当时版本，运行前须对照工具当前官方文档。

### 第 5 章：模型编辑（§5.1–5.5）

- **任务定义：** 在目标知识上更新响应而尽量保留不相关输出，性质涵盖准确性、泛化性、可迁移性、局部性和效率（§5.1）。
- **方法类别：** 外部拓展法（知识缓存与参数补丁）和内部修改法（元学习、知识神经元、定位编辑）；表 5.2 比较 SERAC、T-Patcher、MEND、ROME、MEMIT 等方法。
- **例子：** T-Patcher 通过外加神经元和激活门控控制修改范围；ROME 以因果追踪定位中层 FFN 中的知识关联，再做 Rank-One 修改。
- **应用：** 知识即时更新、隐私遗忘与安全纠错。作者用 Gemini Pro 对话举例仅为**推测性说明**，不可当作 Google 已证实采用 ROME/模型编辑。

### 第 6 章：检索增强生成（§6.1–6.5）

- **动机与架构：** 针对参数知识时效、长尾缺失和幻觉，引入外部知识；§6.2 区分黑盒增强（无微调、仅检索器微调）与白盒增强（仅模型微调、联合微调），讨论 In-Context RALM、REPLUG、RETRO、SELF-RAG、Atlas。
- **知识检索：** 清洗/分块、查询改写与分解、BM25 稀疏检索、DPR/ColBERT 稠密检索、生成式 DocID 检索、HNSW 等近邻索引、结果重排。
- **生成增强：** 何时/何处触发检索、输入/中间层/输出端增强、多次迭代增强及降本；图 6.14 描述从知识库与用户查询并流到检索器再到重排的过程。
- **实践与应用：** §6.5 使用 LangChain、Chroma、embedding、prompt 与 Runnable 链展示小型 RAG；继续讨论 Agent 和记忆以及多模态垂直领域 RAG。示例中的 API、依赖与密钥配置**不是**已验证可直接运行的 2026 年版本。

## 阅读与版本边界

1. **原文版本：** 采用与作者 GitHub 官方 PDF 标题、扉页、目录和六章主体一致的可检索中文全文镜像核对。尚未取得官方 PDF 二进制副本，因此不声明镜像 SHA、图像清晰度或每个字体/页码与当前 `main` 官方文件完全一致。
2. **图示核对方式：** 依据 PDF 解析得到的图注及结构化图表描述核对相关机制；没有声称执行 PDF 内代码或独立对图形像素精度作验证。
3. **英文版：** 只确认英文文件入口，不声称逐段对照或与中文完全同版。
4. **更新时效：** 教材反映其编写时的模型与库版本；案例数字、ChatGPT/GPT 系列描述、LangChain/HF-PEFT 脚本使用时须核对目前一手版本。
5. **授权：** 教材是否可再分发与商用须以官方仓库许可证明确条款为准，本仓仅引用原 PDF 链接和自己的解读。

## 对 wiki 的映射

- [《大模型基础》教材实体](../../wiki/entities/zju-foundations-of-llms.md)
