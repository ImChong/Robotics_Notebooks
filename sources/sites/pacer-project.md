# PACER 项目页（pacer-corl.github.io）

> 来源归档

- **标题：** PACER: Process-Aware VLA Policy Learning from Human Corrections for Robotic Disassembly
- **类型：** site（项目页）
- **URL：** <https://pacer-corl.github.io/>
- **论文：** <https://arxiv.org/abs/2610.19248> — 归档见 [`sources/papers/pacer_arxiv_2610_19248.md`](../papers/pacer_arxiv_2610_19248.md)
- **代码：** <https://github.com/ChangChrisLiu/PACER>（MIT）— 归档见 [`sources/repos/pacer.md`](../repos/pacer.md)
- **入库日期：** 2026-10-17
- **一句话说明：** "Judge practice by its process"：按过程而非仅最终结果评判 rollout 与纠正，只改逐行损失权重。

## 页面要点（2026-10-17 核查）

| 区块 | 内容 |
|------|------|
| 数据来源 | 策略自身 rollout、从出错状态起的人工纠正、在参考策略失败处补录的 clean 演示 |
| 资格门 | 仅有效且安全执行、角色可模仿的行入选；错位置与失败 rollout 模仿权重为 0，仅留作审计 |
| 加权 | 进展、接近度、终端进入、停止/交接、操作员标签；近失误给部分分；纠正与 clean 有权重下限 |
| 训练与选择 | 只改原生动作损失的逐行权重；每个预声明设定训一个策略，按留出验证分数选择 |
| 对比叙事 | outcome-only 丢弃近失误；基于奖励的适应需额外奖励/价值模型与反复 rollout 采集；PACER 两者皆无 |
| 结果数字 | 页面文本中未给出具体成功率（以 arXiv 为准） |
| 可用性 | 页面标注论文 "soon"，代码 MIT 开源 |
