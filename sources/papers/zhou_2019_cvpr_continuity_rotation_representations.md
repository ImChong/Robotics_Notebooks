# On the Continuity of Rotation Representations in Neural Networks（Zhou et al., CVPR 2019）

> 来源归档（ingest）

- **标题**: On the Continuity of Rotation Representations in Neural Networks
- **作者**: Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, Hao Li
- **Venue**: CVPR 2019（Oral，Best Paper Award Candidate）
- **arXiv**: https://arxiv.org/abs/1812.07035
- **PDF**: https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhou_On_the_Continuity_of_Rotation_Representations_in_Neural_Networks_CVPR_2019_paper.pdf
- **类型**: paper / rotation-representation / deep-learning
- **入库日期**: 2026-09-09
- **一句话说明**: 证明四元数、欧拉角等常见旋转参数化在 $\mathbb{R}^n \to \mathrm{SO}(3)$ 映射上存在不连续点；提出用旋转矩阵前两列（6D）经 Gram–Schmidt 重建第三列的**连续**表示，成为后续机器人学习里「6D rotation」的理论母题。

## 开源状态（步骤 2.5）

- **无官方代码仓库**；论文以理论分析与姿态估计实验为主。工程侧常见复现见各框架自实现（PyTorch3D `rotation_6d`、ProtoMotions / MimicKit 的变体等）。
- **结论**: 理论一手资料为 PDF；实现对照以 MimicKit `tan_norm` 等工程变体为准（见 [`sources/repos/mimickit_tan_norm.md`](../repos/mimickit_tan_norm.md)）。

## 摘录 1：问题陈述

神经网络在 $\mathbb{R}^n$ 上回归旋转时，若参数化映射 $f:\mathbb{R}^n \to \mathrm{SO}(3)$ 在欧氏空间不连续，则微小参数变化可能导致输出旋转突变，损失与梯度行为恶化。

**对 wiki 的映射**: [`wiki/formalizations/se3-representation.md`](../../wiki/formalizations/se3-representation.md) 已有 6D 连续表示概述；本页 [`tan-norm-rotation.md`](../../wiki/formalizations/tan-norm-rotation.md) 补充运动模仿栈里的 **tan_norm 工程变体**。

## 摘录 2：6D 连续表示（论文原版）

取旋转矩阵 $R=[\mathbf{a}_1\ \mathbf{a}_2\ \mathbf{a}_3]$ 的前两列作为网络输出 $(\mathbf{a}_1, \mathbf{a}_2) \in \mathbb{R}^6$，解码：

1. $\mathbf{b}_1 = \mathrm{normalize}(\mathbf{a}_1)$
2. $\mathbf{b}_2 = \mathrm{normalize}\big(\mathbf{a}_2 - (\mathbf{b}_1 \cdot \mathbf{a}_2)\mathbf{b}_1\big)$
3. $\mathbf{b}_3 = \mathbf{b}_1 \times \mathbf{b}_2$

拼成 $R=[\mathbf{b}_1\ \mathbf{b}_2\ \mathbf{b}_3]$。该构造在 $\mathrm{SO}(3)$ 上给出连续 surjective 映射（除测度零奇异集外）。

**对 wiki 的映射**: 与 MimicKit **tan_norm** 的差异——论文 6D 是**任意**前两列 + 正交化；tan_norm 是**固定**参考切向/法向经 $R(q)$ 旋转后的 6 维，语义更贴近「体轴方向观测」。

## 摘录 3：与其他表示的对比（论文 Table / 实验结论）

| 表示 | 维度 | 连续性 | 备注 |
|------|------|--------|------|
| 欧拉角 | 3 | 否（万向锁） | 直观但不适合 DL 回归 |
| 四元数 | 4 | 否（$q \sim -q$） | 紧凑，观测/损失需处理双覆盖 |
| 旋转矩阵 | 9 | 是（需正交约束） | 冗余 |
| **6D（前两列）** | 6 | **是** | CVPR'19 主推 |

**对 wiki 的映射**: 运动模仿 RL 选 tan_norm 的动机：连续 + 无四元数符号歧义 + 与 DeepMimic 栈维度习惯一致（每关节 6D 块）。

## 建议 wiki 动作

- 新建 **`wiki/formalizations/tan-norm-rotation.md`**，理论节链到本 paper source
- 更新 **`wiki/formalizations/se3-representation.md`** 的参考来源为可点击 `sources/papers/...` 路径
