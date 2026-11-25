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
- **核对范围：** arXiv v2 已公开的完整 10 页正文（摘要、§1–5、式 1–8、图 1–5、表 1–3、参考文献定位）；v2 仅更新标题。§3.1、§3.3 所引用的慢快流与核验实现细节已通过作者 [METHOD](https://github.com/tiplab/TipDex-VTLA/blob/main/docs/METHOD.md)、[DEPLOYMENT](https://github.com/tiplab/TipDex-VTLA/blob/main/docs/DEPLOYMENT.md) 及实际核心源码补核。独立 supplementary material 未在 arXiv 公布的全文链接中找到；额外数据构建、照明配置及评测流程细节仍未公开核实，不声称已阅读独立附件。
- **通读：** 已完成

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

### 4. 训练及扰动细节（§3.2、§3.4、§4.1、§4.4）

配对采样包含低照、强照、光照方向改变、不均匀照明、原图五种等概率条件；冻结参考策略构建可靠性目标和有符号动作差监督。修正损失只更新 TACE，执行分数停止梯度。论文以 20 种未见照明配置测试每项真机任务；仿真仅标准照明。图 5 中 −3/+3 EV 相对关闭特征修正的轨迹 RMSE 分别降低 24.2%/13.0%。作者在 §5 将视觉可靠性限定为光照变化下的策略相关代理量。
- **对 wiki 的映射：** [TipDex-VTLA](../../wiki/entities/paper-tipdex-vtla.md)

## 关联原始资料

- [项目页归档](../sites/tipdex-vtla.md)
- [源码归档](../repos/tipdex-vtla.md)
