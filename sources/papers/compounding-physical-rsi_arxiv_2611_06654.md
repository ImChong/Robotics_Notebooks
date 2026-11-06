# Compounding Physical RSI: Turning Execution Failure into Inherited Capability（arXiv:2611.06654）

> 来源归档（paper）

- **类型：** paper（技术报告）
- **arXiv：** <https://arxiv.org/abs/2611.06654>（最新版本 v1，摘要页仅列 [v1]）
- **PDF：** <https://arxiv.org/pdf/2611.06654>（10 页，无附录）
- **提交日期：** 2026-11-04，cs.RO
- **作者：** Yixuan Yang, Qiang Chen, Run Shao, Shuwei Dong, Wenhao Li, Yunpeng Wang, Wenhan Fang, Haomiao Wu, Xingyu Wang, Lichen Man, Rong Zhang, Xuezhi Cui, Yuan Liu, Bingyan Lu, Jiawei Ma, Xiaoyang Ma, Zhemin Sun, Yichong Mu 等共 34 位（arXiv 元数据）；通讯作者 Xiu Su
- **署名：** CSU MinosLab；题头标注「ECCV 2026 RoboDojo Safe World Models Challenge | Champion | CSU-AI-07」。作者上标 1–6 在 PDF 中为贡献角色而非机构编号；CSU 全称论文未展开
- **项目页 / 代码 / 权重 / 数据：** 论文与 arXiv 摘要页均未给出；未发现可核实的官方仓库（因此无 sources/sites、sources/repos 归档）
- **入库日期：** 2026-11-06

## 核心摘录（人工转述，数字取自原文表格 / 正文）

- 论点（§1）：操作为阶段串联，任务完成率近似各阶段通过率之积；最早无法可靠通过的阶段称「能力前沿」；三个瓶颈：终端反馈不能定位失败阶段、失败轨迹无纠正动作、共享权重更新损伤旧技能。
- 框架（§2.2，式 1）：A（能力缺口评估）→ g_k；F（验证）→ B_{k+1}；C（筛选）→ D_{k+1}；Uoff（离线精调）→ π'_k；S（权重候选晋升）→ π_{k+1}。
- 修补（§4.5，式 2）：π'_k = argmax E[Q(s,a|g_k)] − λ·D(π, π_k)，沿用行为正则化 / 保守离线 RL，未给 Q、D、λ 的具体取值或形式。
- 保留（§5.2，式 3）：θ_k^(i) = (1−α_i)θ_k + α_i θ'_k；§5.3 用 Overall Score、SR、任务完成有效性在 RoboDojo-Sim 全量评测中择优；§6.2 称 Seed 0 用于迭代选择。
- 结果（Table 1）：Seed 0/1/2 Overall = 26.53/25.46/26.45，SR = 21.05/20.31/20.75；三 seed 均值 26.15 / 20.70%。
- 榜单（Table 2）：13 个公开基线，次优 GalaxeaVLA (G0.5) 20.23 / 14.88%；π0.5 11.44 / 6.93%；π0 3.48 / 1.53%。
- 自述局限（§7）：缺消融；改进算子固定；真机为后续工作。

## 与草稿登记的出入

- 草稿 `sources/papers/2026-11-06-seven-robotics-intake-draft.md` 仅 v1 摘要口径并标注「是否支持真机需核对全文」。全文核对：仅 RoboDojo-Sim 仿真，无真机；无消融；未披露基础策略与关键超参。

## 对 wiki 的映射

- [paper-compounding-physical-rsi](../../wiki/entities/paper-compounding-physical-rsi.md)
- 相关：[physical-rsi](../../wiki/entities/physical-rsi.md)、[physical-rsi-research-map](../../wiki/entities/physical-rsi-research-map.md)、[robodojo](../../wiki/entities/robodojo.md)
