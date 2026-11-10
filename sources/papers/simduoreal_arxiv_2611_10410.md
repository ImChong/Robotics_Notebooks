# SimDuoReal（arXiv:2611.10410）

> 来源归档（paper）

- **标题：** SimDuoReal: Object-Centric Bimanual Policy for Emergent Dexterity
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.10410>
- **PDF：** <https://arxiv.org/pdf/2611.10410v1>（v1，2026-11-07，cs.RO）
- **项目页：** <https://simduoreal.github.io/>
- **代码：** 未发布（入库日 2026-11-10 项目页标注 "Code Coming soon"，未列 GitHub 链接）
- **入库日期：** 2026-11-10
- **一句话说明：** 用统一的物体位姿目标与按工作空间划分的目标采样，训练左/右单臂专家与交接专家，经转移态精修和 DAgger+RL 蒸馏为单一双手策略；真机四项任务 61/80 成功。
- **核对范围：** 通读 arXiv v1 PDF 全部 9 页：摘要、引言、相关工作、方法（III-A 至 III-D）、实验（IV-A 至 IV-C）、结论与局限、参考文献列表、附录 VI-A（奖励各项的公式细节，式 8 至式 14）。图（Fig. 1 至 Fig. 6）只读了图注与正文引用，未逐图检视像素级内容；Fig. 6 的曲线数值未读出。附录里奖励系数、训练超参的具体数值论文未给出（正文称 "Further details appear in Appendix VI-A"，该附录只有公式）。项目页只读了文字部分，未看视频。引用论文（SimToolReal 等）未读。
- **通读：** 已完成

## 核心摘录

1. **机构（首页）：** Amazon FAR；UCSD；University of Chicago；CMU；Stanford University。作者 Qiyang Yan, Yulin Liu, Akash Sharma, Haozhi Qi†, Guanya Shi†, C. Karen Liu†（† 为 Amazon FAR team co-lead）。
2. **方法（Sec. III）：** 三个专家共享物体中心表示与奖励（式 3），只在目标采样、初始化、终止上不同：左/右单臂专家 πL、πR（动作 29 维），交接专家 πH（动作 58 维）；交接目标要求接收手接触指定表面区域且给出手释放（Sec. III-B）。训练算法为 EPO，非对称 critic（Sec. III-B.3）。
3. **蒸馏（Sec. III-C）：** 三类训练场景按概率 0.4、0.4、0.2 抽取；专家在转移态池上精修；学生用 DAgger + RL 损失（式 7）。
4. **真机（Sec. IV-B，Table I）：** 两台 Franka Research 3 加 22 自由度 Sharpa Wave 手；条码扫描 15/20、插销 15/20、清扫 16/20、烹饪 15/20，合计 61/80（76.25%），物体均未见过，无任务微调。
5. **仿真消融（Sec. IV-C，Table II）：** 条码扫描基准 3072 个环境，每个 7 个路标；完整策略成功率 85.0%、进度 90.8%、用时 8.6±3.9 s；教师去掉交接专家（靠放到共享工作空间再抓取来转移）60.4%、去掉转移态精修 79.4%、完整教师 84.3%；学生去掉专家场景 68.3%、去掉 RL 损失 81.6%。摘要把 60.4% 表述为「两个单臂策略、无交接」，Table II 中对应行名为 teacher w/o cross-workspace expert（表格为 PDF 文本抽取，行列对应按数值顺序与正文 Sec. IV-C 交叉核对）。
6. **局限（结论段）：** 需要准确物体网格与多视角位姿跟踪；6-DoF 目标接口不能直接表达力或时序相关行为；目标轨迹由外部给出而非自主生成。

## 对 wiki 的映射

- [paper-simduoreal-object-centric-bimanual](../../wiki/entities/paper-simduoreal-object-centric-bimanual.md)
