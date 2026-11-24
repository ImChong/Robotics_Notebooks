# TipDex-VTLA: Learning to Refine Actions with Visual Reliability and Tactile Feedback for Dexterous Manipulation

> 来源归档（ingest）

- **类型：** paper
- **arXiv：** 2611.25683v2
- **原文：** https://arxiv.org/abs/2611.25683v2
- **HTML：** https://arxiv.org/html/2611.25683v2
- **PDF：** https://arxiv.org/pdf/2611.25683v2
- **作者：** Xudi Ge, Yepeng Liu, Haifeng Liu, Enze Shi, Liangqi Zhang, Yongchao Xu
- **机构：** 武汉大学；武汉华威科智能技术有限公司
- **日期：** 2026-11-19（v1）；2026-11-22（v2，更新标题）
- **入库日期：** 2026-11-24
- **项目页：** https://tiplab.github.io/TipDex-VTLA/
- **代码：** https://github.com/tiplab/TipDex-VTLA
- **权重：** https://huggingface.co/TipDex-VTLA/TipDex-VTLA
- **数据：** https://huggingface.co/datasets/TipDex-VTLA/TipDex-Data
- **核对范围：** v2 正文摘要、§1–5、公式 1–8、表 1–3；另核对项目页与官方推理代码、部署协议及 HF 卡片。正文引用的独立 supplementary material 未找到；训练数据构建与光照设置的额外细节未核验。
- **通读：** 未完成

## 核心论文摘录

### 1. VASE（§3.2）

固定语言、状态、触觉、采样噪声，仅改变光照；冻结策略的两组动作差构成可靠性监督 exp(-MSE/λ)。这衡量对光照的动作一致性，不是成功概率。推理不需要配对参考图。
- **对 wiki 的映射：** [TipDex-VTLA](../../wiki/entities/paper-tipdex-vtla.md)

### 2. TACE（§3.3–3.4）

以时间和已执行命令一致性匹配历史触觉预测，计算归一化 MAE；无合格预测时 e=0 仅代表回退。视觉可靠性与触觉误差共同调节有界特征残差，只替换尚未执行且可替换的动作。训练两阶段分别使用 200、100 条独立示教；后者配对光照增强。
- **对 wiki 的映射：** [TipDex-VTLA](../../wiki/entities/paper-tipdex-vtla.md)

### 3. 评测与边界（§4–5）

真机四任务正常光 87.5%、未见光照 73.8%，UniVTAC 八任务正常仿真光照 84.9%；真机未见光照去 VASE、去触觉核验、去门控分别 42.5%、62.5%、40.0%。可靠性仍是特定参考策略与光照扰动的代理指标。
- **对 wiki 的映射：** [TipDex-VTLA](../../wiki/entities/paper-tipdex-vtla.md)

## 关联原始资料

- [项目页归档](../sites/tipdex-vtla.md)
- [源码归档](../repos/tipdex-vtla.md)
