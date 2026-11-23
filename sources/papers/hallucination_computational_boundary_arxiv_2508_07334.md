# Hallucination as a Computational Boundary: A Hierarchy of Inevitability and the Oracle Escape

> 来源归档（paper）｜中文：幻觉作为计算边界：必然性层级与预言机逃逸

- **论文：** https://arxiv.org/abs/2508.07334
- **HTML 全文：** https://arxiv.org/html/2508.07334v2
- **PDF 全文：** https://arxiv.org/pdf/2508.07334v2
- **正式出版：** AAAI 2026，40(40):33675–33682；DOI https://doi.org/10.1609/aaai.v40i40.40657；[会议论文记录](https://ojs.aaai.org/index.php/AAAI/article/view/40657)
- **版本：** arXiv:2508.07334v2，2025-12-08（v1：2025-08-10）；会议发表于 2026-03-14
- **作者：** Xi Wang、Quan Shi（共同第一作者）、Zenghui Ding、Jianqing Gao、Xianjun Yang
- **机构：** 中国科学院合肥物质科学研究院、中国科学技术大学、常州大学、科大讯飞研究院
- **项目页 / 官方代码：** arXiv HTML 和 AAAI 论文记录未提供独立项目主页或可运行仓库链接；截至 2026-11-23 未确认公开实现，不能将文中仿真实验当作已复现。
- **入库日期：** 2026-11-23
- **核对范围：** 已通读 arXiv v2 HTML 全文：§1–7（定义 2.1–2.3、定理 2.4–2.5、引理 2.6、定理 3.1/3.3、持续学习层级博弈、§4 方法比较、§5 实验设置/结果/注意力分析、§6 CCA 与 §7 结论）、表 1–2、图 1–6 的正文说明和参考文献；核对 AAAI 正式发表元数据。未进行原论文实验复现，未验证官方代码。
- **通读：** 已完成

## 摘录 1：三层不可避免性（§2）

将 PLM 定义为从字符串到输出分布的可计算概率图灵机。定义相对关系型真值的越界幻觉 H_Stray（错误答案的总概率质量），和相对概率型真值的分布扭曲 H_Distort（真值分布到模型分布的 KL 散度）。作者依次论证：

1. **对角化边界（Theorem 2.4）：** 针对模型可枚举序列构造与其首选输出相反的目标；
2. **不可计算边界（Theorem 2.5）：** 用停机问题证明一般算法无法在所有输入上判定真值；
3. **信息边界（Lemma 2.6）：** 以 Kolmogorov 不可压缩字符串构造有限模型容量不足以精确记忆的目标。

**核对注意：** 定理 2.4 证明中使用最大输出概率至少为 1/|Y|，对于此前允许的无限输出串空间，此下界不足以给出统一正的 ε；这是原文证明需进一步澄清的地方，不能直接把结论推广成“所有真实任务必然幻觉”。

## 摘录 2：外部 Oracle 与持续学习（§3–4）

- **Theorem 3.1：** 作者构造真值本身由 Oracle O(s) 定义的任务，增强调用同一个 Oracle 并原样输出，因而在该构造上误差为零。这是理想 Oracle 模型的存在性论证，不是“实际 RAG 检索和生成保证零幻觉”的经验结论。
- **持续学习：** 提出皮层式慢速泛化子博弈 G_C（Generalizer / Exception Handler）与海马式快速记忆子博弈 G_H（Plasticity Agent / Memory Consolidator），通过整合协议 C 联系为层级纳什均衡 HNE；从而构思外部检索与参数知识内化的混合体系。
- **Theorem 3.3：** 在每次外部查询成本 C_query>0、一次内化更新成本 C_update 有限且新知识可被反复使用的简化模型下，高重复查询次数 N 会使持续学习的摊销成本低于每次检索。未计算遗忘、检索未命中、更新失败和验证成本。

## 摘录 3：实验与指标（§5，Table 2，Figures 5–6）

作者使用 Mistral-7B、FAISS、LoRA，在虚构新科学事实的语料上对比 Pure RAG / Pure CLM / RAG-CL，报告 1000 次查询：

| 方法 | 准确率 | TriviaQA 遗忘率 | 15% 检索噪声下鲁棒性 |
|---|---:|---:|---:|
| Pure RAG | 约 98.6% | 0% | 76.5% |
| Pure CLM | 约 81.0% | 12.4% | N/A |
| RAG-CL | 约 96.5% | 1.1% | 92.3% |

论文报告混合方案的成本交叉点约在 287 次查询，并用注意力分析解释对检索上下文依赖下降；这些是**作者报告**，未能找到官方可运行代码、数据生成配置与重复试验统计来独立复现或量化置信区间。

## 摘录 4：工程含义与风险（§6–7）

CCA（Computational Class Alignment）要求将任务复杂度与系统实际可用计算能力对齐；面向部署可翻译为：高风险任务做能力边界评估、证据/工具调用、验证和必要时拒答。真实检索器不是停机问题 Oracle，持续学习也不会自动突破可计算性限制；在机器人中尤其不能将语言层辅助证据当作物理安全保证。

## 对 wiki 的映射

- [独立论文实体：Hallucination as a Computational Boundary](../../wiki/entities/paper-hallucination-computational-boundary.md)
- [RAG 概念：实际检索与理想 Oracle 的区别](../../wiki/concepts/retrieval-augmented-generation.md)
