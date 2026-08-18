# DF-ExpEnse: Diffusion Filtered Exploration for Sample Efficient Finetuning

> 来源归档（ingest · REALab 14 篇盘点）

- **标题：** DF-ExpEnse: Diffusion Filtered Exploration for Sample Efficient Finetuning
- **类型：** paper
- **状态：** ICML 2026
- **原始链接：**
  - arXiv：<https://arxiv.org/abs/2606.19656>
  - 项目页：https://df-expense.github.io/
- **代码：** https://github.com/real-stanford/dfexpense
- **作者：** Calvin Luo, Chen Sun, Shuran Song
- **机构：** Stanford University; Brown University
- **入库日期：** 2026-08-18
- **一句话说明：** DF-ExpEnse（ICML 2026）：扩散策略多模态采样+critic ensemble 滤波探索，提升生成式策略 RL 微调样本效率；机群协同探索；代码已开源。

## 核心论文摘录（MVP）

### 问题与贡献

- **摘录要点：** 在扩散/flow 预训练策略上做 RL 微调时，用策略自身多模态采样构造候选动作集，再用 critic ensemble 在「执行质量」与「探索兴趣」间选动作，显著提升在线样本效率。
- **对 wiki 的映射：**
  - [wiki/entities/paper-df-expense.md](../../wiki/entities/paper-df-expense.md)

### 方法与结果（归纳）

- **方法：** 每步：(1) 扩散策略采样多条动作候选；(2) critic ensemble 估计价值与不确定性；(3) 选探索兴趣最高的动作执行；fleet 间可同步归一化兴趣分数。
- **评测：** 操作与运动任务上相对默认微调与替代动作选择方案持续更高样本效率。

## 当前提炼状态

- [x] 公众号盘点 + arXiv/项目页交叉核对
- [x] wiki 实体页：`wiki/entities/paper-df-expense.md`
