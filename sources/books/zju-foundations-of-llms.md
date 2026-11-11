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
- **核对范围：** 已完整核对官方仓库 readme.md 的出版说明、六章目录、PDF/分章与 Paper List 链接；查阅教材 PDF 可检索文本中的扉页/目录，以及第 1、2、3、4、5、6 章的部分正文（语言模型评测、SSM、CoT、PEFT 文献、模型编辑参考、RAG 总览）。由于未逐页检查六章及所有公式与插图，不声称通读教材全文。PDF 文本的辅助参考来自与官方书名、署名和目录匹配的可检索镜像 https://library.qiangtu.com/download/1108/pdf/1108.pdf；以官方仓库 PDF 为规范来源。
- **通读：** 未完成
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

## 已抽查的教材正文与边界

- **第 1 章 §1.5：** 对比字符串/重叠指标与语言模型评估方式，讨论 BERTScore、G-EVAL 及自动评测的限制。
- **第 2 章 §2.6：** 介绍状态空间模型（SSM）的连续、递归与卷积表示，并讨论 RWKV、Mamba 等非 Transformer 方向。
- **第 3 章 §3.3–3.4：** 思维链提示与任务适配条件；教材没有把 CoT 描述为所有任务的必选项。
- **第 4 章：** 目录将 PEFT 归为参数附加、参数选择、低秩适配，正文文献列举 LoRA / Adapter 等；尚未逐条核对每项实验细节。
- **第 5 章：** 以 T-Patcher 和 ROME 作为编辑方法的具体实例，区分附加与定位编辑思路。
- **第 6 章 §6.1：** 从模型知识时效、覆盖与幻觉问题引入 RAG；后续小节讲知识库构建、查询增强、检索器/重排和生成阶段增强。

## 尚待核验

1. 六章全部正文、推导、图表、实践代码及参考文献逐条复核；完成之前不要将“通读”改为已完成。
2. 英文 PDF 与中文版具体版本差异；仅确认英文文件存在，不宣称逐段等价。
3. 源码许可证、PDF 分发与二次转载条件按官方 LICENSE.md 单独核实。

## 对 wiki 的映射

- [《大模型基础》教材实体](../../wiki/entities/zju-foundations-of-llms.md)
