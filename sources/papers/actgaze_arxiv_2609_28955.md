# ActGaze: Learning Action-Grounded Gaze through Counterfactual Visual Interventions for High-Precision Manipulation（arXiv:2609.28955）

> 一手论文归档；本文以作者原文与项目站点为依据，非二手公众号转述。

- **原文：** https://arxiv.org/html/2609.28955v1
- **arXiv：** https://arxiv.org/abs/2609.28955
- **PDF：** https://arxiv.org/pdf/2609.28955
- **项目页：** https://anonymous.4open.science/w/ActGaze/
- **独立附录：** https://anonymous.4open.science/api/repo/ActGaze/file/assets/appendix.pdf
- **作者：** Jinxuan Zhu, Jiaheng Wang, Chao Tang, Mengfan Wang, Hao Wei, Shengbao Li, Hong Yin, Yiwen Gao, Chenrui Tie, Tingguang Li
- **机构：** Samsung Robotics eXperience（RX）、National University of Singapore（NUS）
- **版本：** arXiv:2609.28955v1（2026-09-24）
- **代码：** 截至 2026-11-09，项目站点未列可运行 GitHub 仓库、训练脚本、权重或数据下载链接；**不能视为已开源**。
- **开源状态：** 研究论文、项目展示、附录公开；实现未公开
- **首次入库：** 2026-09-27；**全文复核：** 2026-11-09
- **核对范围：** arXiv HTML 原文完整阅读（Abstract、§I–VI、方法 Eq.1–5、Algorithm 1、表 I–IV、讨论/局限、参考文献）；论文所附 Appendix A–D（理论/证明、标注协议、四任务公差与随机范围、视觉干扰表 V）；对照作者项目主页及独立 appendix.pdf 全部 3 页（交叉核对）。主 PDF 31.8 MB 在浏览工具中下载超限，故以官方同版本 arXiv HTML 全文及作者附录 PDF 逐页核对。
- **通读：** 已完成

## 研究问题与方法

标准 VLA 的动作损失并不监督视觉注意力的位置。ActGaze 先训练一个能够预测动作的 π0.5，再冻结它，在多相机输入上做两级视觉反事实干预：**整相机遮挡**和**候选 patch 区域模糊**。如果某个区域被干预后，原动作的预测损失明显增大，就把它视作对当前动作重要的候选区域。

- 输入：多视角 RGB `I_t`、机器人状态 `s_t`、语言指令 `l`；策略输出 H 步 action chunk。
- Eq. (2)：`r_t(S) = L_act(I(I_t; S)) - L_act(I_t)`，同一次 clean/intervened 比较中保持示范动作、状态、语言、flow time、噪声、noisy action 和目标速度一致。
- 相机级干预得到权重 `w_{t,v}`，patch 级干预得到候选区域的归一化重要性 `q_{t,v,k}`。
- Eq. (3)：`L_gaze = -Σ_v w_{t,v} Σ_k q_{t,v,k} log(q_hat_{t,v,k})`。由原始未干预图像训练多视角 MLP gaze head。
- Eq. (4)：冻结 gaze head，将其预测的空间分布 `g_hat` 蒸馏回 VLA 的 action-to-vision attention：`L_refine = L_act + λ_align ||α_{θ,t} - g_hat_t||²`；优化时用 **PCGrad** 处理动作/对齐目标梯度冲突。
- **三阶段不可省**：初始化策略 → 反事实动作到注视蒸馏（冻结策略）→ 注视到动作精炼（冻结 gaze head）。本文最佳工程取舍采用一次 refinement。
- **干预并非在真实物理环境施加因果操作**；得到的是固定策略依赖的相对预测重要性，不应称为无偏物理因果效应（§VI / Appendix A）。

## 实验（作者报告；未独立复现）

基座 π0.5；Rainbow RB-Y1；两路腕部 Intel RealSense D405 与一路头部 ZED X Mini；视觉 patch grid 16×16，候选区域 2×2 patches，每视角每步 K=2 个候选；初始化验证阈值 ε_act=0.002。每任务 200 例专家演示；训练在 8×NVIDIA RTX 5880 上进行，比较策略使用相同示范数与同场次评估环境。

### 表 I：任务相关区域 Ground-Truth Alignment Score（%）

| 空间图 | Straw | Hose | LEGO | Connector | 均值 |
|---|---:|---:|---:|---:|---:|
| π0.5 action attention | 14.0 | 12.0 | 21.2 | 17.9 | 16.3 |
| ActGaze gaze head | 42.5 | 33.4 | 46.7 | 38.7 | 40.3 |
| ActGaze refined action attention | 37.3 | 30.7 | 38.0 | 36.9 | 35.7 |

每个任务、每个视角从跨多个 episode 的稀疏采样帧中人工标注 200 帧，仅用于 **评估**，不进入训练（Appendix B）；不要将“无需空间标注训练”误读成“论文评测完全没有人工标注”。

### 表 II：四项真机任务成功数（每方法每任务 50 次）

| 方法 | Straw | Hose | LEGO | Connector | 合计 |
|---|---:|---:|---:|---:|---:|
| π0.5 | 37/50（74%） | 34/50（68%） | 44/50（88%） | 39/50（78%） | **154/200（77.0%）** |
| ActGaze | 44/50（88%） | 46/50（92%） | 48/50（96%） | 45/50（90%） | **183/200（91.5%）** |

提升 **14.5 个百分点**，不是相对提升 14.5%；增益最大为 Hose +24 个百分点。

### 表 III：空间监督基线（仅 Hose + Connector，20 次/项）

| 方法 | Hose | Connector | 平均 |
|---|---:|---:|---:|
| CoT Grounding | 10/20 | 12/20 | 55.0% |
| Implicit Grounding | 13/20 | 12/20 | 62.5% |
| ActGaze | 18/20 | 17/20 | **87.5%** |

两项外部 grounding 基线的空间框由 Grounding-DINO 产生；与四任务 50 次/项的 91.5% **不是同一实验口径**。

### 表 IV：关键消融（仅 Straw + LEGO，20 次/项）

| 变体 | Straw | LEGO | 平均 |
|---|---:|---:|---:|
| w/o counterfactual supervision | 8/20 | 7/20 | 37.5% |
| w/o staged training | 13/20 | 12/20 | 62.5% |
| ActGaze | 19/20 | 18/20 | **92.5%** |

LEGO iterative refinement：Round 0=78%、1=90%、2=90%、3–4=92%。作者默认采用一轮，以平衡训练成本与边际收益。

### 附录 D：视觉干扰鲁棒性（LEGO，20 次/设置）

| 方法 | Clean | Clutter | Strong lighting |
|---|---:|---:|---:|
| π0.5 | 18/20 | 13/20 | 11/20 |
| ActGaze | 19/20 | 15/20 | 11/20 |

对**局部杂物**有小幅提升，但对**全局强光变化**没有可见优势；不能泛化为所有视觉扰动均改善。

## “亚毫米”表述边界（附录 C）

论文称任务需要毫米至亚毫米**几何精度**，并非公布末端定位误差测量的“亚毫米精度控制器”。
- Straw：11.8 mm 吸管、13.8 mm 杯孔，杯位置 10×10 cm 随机。
- Hose：4.6 mm 软管、5.0 mm 孔（名义直径差 0.4 mm），底座平移至 10 cm、旋转至 15°。
- LEGO：8 孔积木孔宽 9.3 mm；黄色块 5×5 cm、±30° 随机，目标橙色块 15×15 cm 随机。
- Connector：24-pin ATX 插接件与插座**间隙在 0.4 mm 内**，主板约 3 cm 位置随机，插头 10 cm 水平范围随机。

## 理论与边界（Appendix A / §VI）

理想信息移除且最优平方损失预测器下，`V(S) = E ||Y_hat - Y_hat_minus_S||² ≥ 0`（Proposition 1）。实际 ActGaze 只用视觉模糊/遮挡近似移除信息，并在固定初始策略上计算单策略损失差；视觉分布偏移、模型既有偏差会传入 gaze target。未来是否能迁移到其他 VLA、不同任务与光照，本文未充分验证。

## 对 wiki 的映射

- [ActGaze 唯一实体（全文复核）](../../wiki/entities/paper-actgaze.md)
- [项目站点来源归档](../sites/actgaze-4open-science.md)
