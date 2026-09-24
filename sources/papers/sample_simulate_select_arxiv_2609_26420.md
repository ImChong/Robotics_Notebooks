# Sample, Simulate, Select（arXiv:2609.26420）

> 来源归档（ingest）

- **标题：** Sample, Simulate, Select: Physics-in-the-Loop Text-to-Motion for Humanoids Without Training
- **类型：** paper / humanoid / text-to-motion / physics-in-the-loop
- **arXiv abs：** <https://arxiv.org/abs/2609.26420>
- **PDF：** <https://arxiv.org/pdf/2609.26420>
- **项目页：** <https://raphaelmemmesheimer.github.io/sample-simulate-select/> — 归档见 [`sources/sites/sample-simulate-select-memmesheimer.md`](../sites/sample-simulate-select-memmesheimer.md)
- **代码：** **未开源** — 项目页 Code 按钮 disabled（2026-09-24）
- **机构：** 波恩大学（University of Bonn, AIS）
- **入库日期：** 2026-09-24
- **一句话说明：** 无训练 S³：MoMask 采样 N 条文本动作 → G1 IK 重定向 → SONIC 物理 rollout 选最优；HumanML3D 直立执行 83.5→89.5%（N=8）；177/177 真机 gate 片段全部站立完成。

## 核心摘录

### 1) 流程

1. **Sample：** 冻结 text-to-motion（MoMask）每 prompt 采 N 候选。
2. **Simulate：** direction-matching IK 重定向到 Unitree G1；SONIC tracking policy 全刚体动力学 rollout。
3. **Select：** 保留 policy 执行最好（tracking error 最低）的候选 — verifier 即确定性仿真器本身。

### 2) headline 数字

| 设定 | 无选择 | S³ best-of-N |
|------|--------|--------------|
| 200 stratified prompts, N=8 直立执行 | 83.5% | **89.5%** |
| 全测试集 4184 prompts | 80.5% | **89.5%** |
| hardware-gate passes（200） | 33 | **85** |
| 真机 gate 片段站立完成 | — | **177/177** |
| sim vs 真机 tracking error | 0.115 rad | **0.114 rad**（r=0.94） |

### 3) 关键读法

- 运动学 verifier（AUROC 0.90）只恢复约 **1/4** 增益 — **排序同 prompt 候选** 比 **分类总体** 更难。
- **不可恢复类：** 降低骨盆的 sit/kneel/deep bend — 冻结生成器不产生可执行样本。
- 双 retargeter（direction IK vs GMR）互补；any-of-8 合并上限 **95.0%**。

## 对 wiki 的映射

- 新建：[paper-sample-simulate-select](../../wiki/entities/paper-sample-simulate-select.md)
- 交叉：[sonic-motion-tracking](../../wiki/methods/sonic-motion-tracking.md)、[locomotion](../../wiki/tasks/locomotion.md)、[sim2real](../../wiki/concepts/sim2real.md)
