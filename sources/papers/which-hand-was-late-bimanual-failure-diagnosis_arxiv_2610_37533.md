# Which Hand Was Late（arXiv:2610.37533）

> 来源归档（paper）

- **标题：** Which Hand Was Late? Diagnosis for Robot Failure Recovery in Bimanual Tool Use
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.37533>
- **PDF：** <https://arxiv.org/pdf/2610.37533>
- **作者：** Bonin, Baptiste, Strickland, Caro, Durand, Audrey
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** 哪只手延迟了？双臂工具使用中的失败诊断与恢复。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** 双臂工具操作的失败可能来自左右臂不同步、工具滑移或接触力不足。单一相机或单臂传感器未必能够区分这些原因。
3. **周报所述创新点：** 系统分析不同传感器配置能够识别哪些失败原因，并建立基于诊断结果的恢复策略。实验发现，部分失败原因在被动观测下无法唯一确定，因此恢复策略需要根据可观测性决定直接修正还是进一步获取信息。
4. **arXiv 摘要（英文节选）：** Failure diagnosers for robot manipulation are increasingly vision--language models (VLMs), trained and evaluated on single-arm failures produced by perturbing a successful demonstration. Bimanual tool use adds failures of other kinds: a tool that slips out of its socket while still held, two correct gestures made at different times, and attempts that fail for two reasons at once. On a simulated bimanual tool-use task whose failure causes are read from the recorded trace, we measure what each sensing channel lets a diagnoser tell.

**对 wiki 的映射**

- [paper-which-hand-was-late-bimanual-failure-diagnosis](../../wiki/entities/paper-which-hand-was-late-bimanual-failure-diagnosis.md)
