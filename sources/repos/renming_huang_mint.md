# MINT（RenMing-Huang/MINT）

> 来源归档

- **标题：** MINT — Mimic Intent, Not Just Trajectories
- **类型：** repo
- **组织：** Renming Huang 等（上海交通大学 / 上海创智学院）
- **代码：** <https://github.com/RenMing-Huang/MINT>
- **项目页：** <https://renming-huang.github.io/MINT/>
- **论文：** <https://arxiv.org/abs/2602.08602>（RSS 2026）
- **入库日期：** 2026-07-03
- **一句话说明：** MINT 官方实现：SDAT 频域多尺度动作分词、MINT 跨尺度自回归策略、MINT-Zero 单样本意图注入与 LIBERO 等基准训练/评测脚本。
- **沉淀到 wiki：** [MINT（论文实体）](../../wiki/entities/paper-mint-vla.md)

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [VLA](../../wiki/methods/vla.md) | 针对 VLA/IL **泛化与迁移** 瓶颈，在 **动作表示层** 解耦意图与执行 |
| [Action Tokenization](../../wiki/formalizations/vla-tokenization.md) | **SDAT** 把分词从「压缩」推进到 **频域谱监督的语义层次** |
| [Manipulation](../../wiki/tasks/manipulation.md) | LIBERO / MetaWorld / CALVIN / 真机操作评测语境 |

## 为何值得保留

- 论文强调 **组合泛化 + 小样本迁移**；开源仓是复现 **SDAT 训练、意图注入、LIBERO-Plus 鲁棒性** 的直接入口。
- 与 [Humanoid_Robot_Learning_Paper_Notebooks](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks) 单篇深读互补：本库负责 **跨主题 VLA 泛化方法** 索引。
