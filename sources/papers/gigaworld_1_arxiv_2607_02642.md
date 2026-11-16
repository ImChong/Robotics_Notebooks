# GigaWorld-1 技术报告

- **标题：** GigaWorld-1: A Roadmap to Build World Models for Robot Policy Evaluation
- **原始链接：** <https://arxiv.org/abs/2607.02642>
- **版本：** arXiv:2607.02642v1（2026-07-02），PDF封面2026-07-07。
- **核对范围：** 正文§1–§7全文、全部方法与结果表；该版本无附录。另核官方页、README与固定源码推理入口。
- **通读：** 已完成
- **沉淀到 wiki：** [GigaWorld-1](../../wiki/entities/paper-gigaworld-1-policy-evaluation.md)

## 主要核对

- WMBench2989配对轨迹，训练82470秒/测试7200秒；324k为社区生成rollout段，20–30段拼完整长程episode，不能称324k次独立完整真机对照。
- WMES是0–3成败正确性×保真度的序数标签，不是Pearson排序相关本身；论文图6/网页有1–3示例口径，与正文定义不同须保留差异。
- 主表0.6834是六个归一化诊断的均值，不是WMES、成功率或策略排名相关；相对Wan2.2 .5948提高14.9%，相对最强Cosmos2.5 .6123只提高11.6%。网页图注称七指标/Nano最佳，但表为六指标/Plus第一，以表为准。
- VLM在5000+视频与人类精确一致87.80%，相邻99.16%，Spearman .7574，不能写近乎完美精确一致。物理接触VLM诊断负相关也提醒人审边界。
- Table2含被认为不可靠的Photometric指标，且AVG不能直接由可见列等权复算；不把其均分增益当成严格WMES提升，JEPA/Trajectory下降要同时报告。
- 模型是动作条件视频评估器：EE pose与ray map拼图后编码，控制latent按channel拼noisy latent；首帧锚点+三级history+局部相对RoPE防漂移，不是联合输出策略动作的WAM。
- 论文32 H20训练、35.93倍叠加多种加速；网页27.23倍/20FPS/<24GB混合设定，与公开Stage-1慢采样不同。DMD2必需最终少步但其权重当前未发，RL脚本亦待发；12980h不等于全部语料可下载。
