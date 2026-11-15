# OptiMer：持续预训练的最优分布向量合并优于直接混合数据 / Optimal Distribution Vector Merging Is Better than Data Mixing for Continual Pre-Training

> 原始论文资料归档（论文正文与附录研读，不全文转载）

- **类型：** paper / continual pre-training / model merging / data mixture optimization
- **原题：** OptiMer: Optimal Distribution Vector Merging Is Better than Data Mixing for Continual Pre-Training
- **中文标题：** OptiMer：持续预训练的最优分布向量合并优于直接混合数据
- **作者：** Haiyue Song, Huy Hien Vu, Masao Utiyama
- **机构：** 日本国立信息通信研究机构（NICT）
- **发表：** EMNLP 2026 Main，Oral，Outstanding Paper Award；ACL Anthology 2026.emnlp-main.1090，DOI 10.18653/2026.emnlp-main.1090
- **论文（正式版）：** https://aclanthology.org/2026.emnlp-main.1090/
- **PDF（正式版，21 页含附录）：** https://aclanthology.org/2026.emnlp-main.1090.pdf
- **arXiv：** https://arxiv.org/abs/2603.28858
- **项目主页：** https://shyyhs.github.io/optimer/
- **官方代码：** https://github.com/nict-astrec-att/optimer
- **入库日期：** 2026-11-15
- **核对范围：** 对照 EMNLP 2026 论文 PDF §1–6、Limitations 以及附录 A–J（超参数、指标、对比扩展、向量相似度、搜索可视化、迁移与案例）；核对作者项目主页、官方 README 与 optuna_merge_search.py 的实际入口。附录参考文献与实例作为背景使用；未下载模型 checkpoint，未执行 GPU 搜索。
- **通读：** 已完成

## 研究问题

多来源持续预训练（CPT）前须确定数据配比；不同语言/领域组合的最优配比不明显，选择失误则需重新训练。OptiMer 将“训练时数据比例优化”转化为“训练后参数差分向量的权重搜索”，令一组单域 CPT checkpoint 可重复用于不同目标。

## 方法（论文 §3，Algorithm 1）

1. 从同一预训练基座 `θ_pt` 分别对不同数据分布 `D_i` 执行 CPT，定义分布向量 `τ_i=θ_CPT_i−θ_pt`。
2. 另抽取指令微调向量 `τ_it=θ_it−θ_pt`，补回 CPT 模型不足的指令跟随能力。
3. 组合模型：`θ_merge = θ_pt + α_it τ_it + Σ_i α_i τ_i`；实现默认经 mergekit 的 DARE-Linear 合并。
4. 以验证集上的任务分数为黑盒目标，通过 TPE（Optuna）优化 `α`；初始随机试验后持续提出新组合，对 top-K 候选做更全面评测。
5. 已有向量池可针对英语、数学、日语、中文等不同目标重新优化，**无需重新做 CPT**。

## 实验与结论（论文 §4–5、Table 1–3）

- 基础模型：Gemma 3 27B；数据包含日语、中文、数学、代码等领域（各 10 亿 token）；比较 1-shot 的英语/数学/代码/日语/中文相关基准，论文提到总计 16 个 benchmark、5 组任务轴。
- 论文报告 OptiMer 的组合平均分相对均匀配比 DataMix **提升 2.1–6.7 分**；例如 Ja+Zh+Math：DataMix 63.71、按 OptiMer 比例重训 DataMix 68.66、OptiMer 70.37。
- **15–35 倍更快是“寻找配比时的搜索成本”，不含一次性 CPT 成本**（论文 Figure 2：2 数据源 128.9h vs 8.5h；3 数据源 194.0h vs 5.6h；4 数据源 258.6h vs 10.1h）。不是全流程训练加速 15–35 倍。
- 分布向量与指令微调向量余弦相似度约 0.03；不同 CPT 向量约 0.29–0.31。论文认为这解释了较低合并冲突，但不保证所有架构都可合并。
- Gemma-SEA-LION-v4-27B 迁移实验：日本排行榜 66.34→74.40，综合平均 54.37→70.19（论文 Table 3；作者指标、非本次复现）。
- 少量搜索 trial（论文常用 100 次）可找到较高分权重；转为数据混合比例再重训也能改善均匀配比结果，但合并本身仍更优。

## 对比及限制（论文 Limitations）

- 对比方法：均匀 DataMix、按 OptiMer 比例重训、Task Arithmetic、TIES、DARE-Linear、DARE-TIES；**未在 27B 规模直接比较 DoReMi、RegMix 等更强的自动配比方法**。
- 只验证 Gemma 3 / Gemma-SEA-LION 两种相关 27B 模型，未证明对 Qwen、Llama、VLA 或机器人策略通用。
- CPT 加长时偏离基座可能损害线性合并；迭代训练—合并尚待研究。
- 所有主实验采用 1-shot，无法与其他 few-shot 协议榜单直接对比。
- GitHub 已开源 Optuna+mergekit+lm-eval 合并与评测入口，**不包含从原始数据生成全部 CPT checkpoint 的完整训练管线/预制 checkpoint**；“公开脚本”不等于原实验一键复现。

## 对 wiki 的映射

- [OptiMer 论文实体](../../wiki/entities/paper-optimer-optimal-distribution-vector-merging.md)
- [CapVector：机器人 VLA 能力向量](../../wiki/entities/paper-capvector-capability-vectors-vla.md)——概念关联，不是 OptiMer 的机器人评测证据。
- [官方项目页归档](../sites/optimer_shyyhs_github_io.md)
- [官方代码归档](../repos/optimer_nict_astrec_att.md)
