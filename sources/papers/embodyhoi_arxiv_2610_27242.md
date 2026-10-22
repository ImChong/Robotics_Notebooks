# EmbodyHOI 论文来源归档

## 元数据

- **题目：** EmbodyHOI: Embodiment-Aware Generation of Hand-Object Interactions for Humanoid Dexterous Manipulation
- **作者：** Chaitanya Chawla, Kartik Sharma, Ananya Bal, Laszlo A. Jeni, Guanya Shi
- **单位：** 抓取页面未列出
- **预印本：** https://arxiv.org/abs/2610.27242（cs.RO，v1，2026-10-19，CC BY 4.0）
- **HTML：** https://arxiv.org/html/2610.27242v1
- **项目页：** https://embodyhoi.github.io/（含仿真结果）
- **代码：** 抓取页面未列出代码仓库
- **入库日期：** 2026-10-22

## 论文要解决的问题

文本条件的手-物交互生成器在人手空间中生成示范，机器人适配通常靠事后 retargeting，容易破坏物体接触。

## 方法要点（据 arXiv 摘要）

- 在冻结的 flow-matching 轨迹生成器采样过程中，嵌入针对特定人形机器人的可微逆运动学（IK）求解器。
- IK 的梯度引导采样，并联合调整人形机器人相对物体的位置，因此同一生成器可跨机器人使用，无需重训。

## 报告的结果（据 arXiv 摘要）

- 数据：ARCTIC 轨迹，机器人为 Unitree G1 与 H1-2。
- 手腕误差：8.9 cm → 0.6 cm。
- 任务成功率：G1 68% → 79%；H1-2 62% → 76%。
- 作者报告：事后手腕修正与计算量匹配的 best-of-N 选择虽降低 IK 能量，但训练出的策略更差。

## Robotics_Notebooks 映射

- 实体页：[EmbodyHOI](../../wiki/entities/paper-embodyhoi.md)

## 一手来源

- arXiv 摘要页（经 WebFetch 摘要，未逐段核对 PDF）
