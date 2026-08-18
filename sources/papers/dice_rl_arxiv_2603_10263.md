# From Prior to Pro: Efficient Skill Mastery via Distribution Contractive RL Finetuning

> 来源归档（ingest · REALab 14 篇盘点）

- **标题：** From Prior to Pro: Efficient Skill Mastery via Distribution Contractive RL Finetuning
- **类型：** paper
- **状态：** ICML 2026
- **原始链接：**
  - arXiv：<https://arxiv.org/abs/2603.10263>
  - 项目页：https://zhanyisun.github.io/dice.rl.2026/
- **代码：** https://github.com/real-stanford/dice-rl
- **作者：** Zhanyi Sun, Shuran Song
- **机构：** Stanford University
- **入库日期：** 2026-08-18
- **一句话说明：** DICE-RL（ICML 2026）：把 RL 视为分布收缩算子，选择性行为正则+价值引导精炼扩散/flow BC 先验；Robomimic 与真机长周期操作；代码+HF 数据已开源。

## 核心论文摘录（MVP）

### 问题与贡献

- **摘录要点：** 把 RL 微调看成在预训练生成式策略周围「收缩」动作分布：高价值状态保留先验，低价值区放大已观测高回报模式，稳定地把 Prior 练成 Pro。
- **对 wiki 的映射：**
  - [wiki/entities/paper-dice-rl.md](../../wiki/entities/paper-dice-rl.md)

### 方法与结果（归纳）

- **方法：** 扩散/flow BC 先验 + 残差 off-policy RL；选择性行为正则（高价值状态贴先验）；价值引导动作选择抑制低价值采样。
- **评测：** Robomimic Can/Square/Tool Hang 等；像素输入仿真与真机长周期操作均显著提升成功率与收敛速度。

## 当前提炼状态

- [x] 公众号盘点 + arXiv/项目页交叉核对
- [x] wiki 实体页：`wiki/entities/paper-dice-rl.md`
