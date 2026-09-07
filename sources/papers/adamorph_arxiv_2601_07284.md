# adamorph_arxiv_2601_07284

> 来源归档（ingest）

- **标题：** AdaMorph: Unified Motion Retargeting via Embodiment-Aware Adaptive Transformers
- **类型：** paper
- **来源：** arXiv
- **原始链接：** <https://arxiv.org/abs/2601.07284>
- **入库日期：** 2026-09-07
- **一句话说明：** 单一 Transformer 统一重定向 12 种人形：SMPL 动作 → morphology-agnostic intent latent → AdaLN + robot soft prompt 解码为目标 embodiment 的 base-frame 速度/关节轨迹；课程式物理一致性损失。

## 核心摘录

### 1) 条件生成式统一框架
- **Intent encoder：** 拼接 Dynamic Human Prompt（来自 SMPL shape β）与 canonical base-frame 特征（v, ω, projected gravity, 6D joint rotations）。
- **Embodiment decoder：** Learnable Static Robot Prompts + **AdaLN** 全局调制 decoder 归一化统计；Cross-Attention 检索 prompt token。
- **Output adapters：** 轻量 MLP ψ_k 投影到各机器人 9+N_k 维输出（base velocities + DoF）。

### 2) 物理一致性
- 预测 base-frame 速度而非绝对位姿；训练时可微积分 + SO(3) Gram-Schmidt 投影；orientation / trajectory consistency 课程损失。

### 3) 实验
- **12 种人形** 联合训练；t-SNE 显示 robot prompt 按拓扑聚类（如 Unitree 家族）。
- **零样本：** 未见过的 stylized dance 仍保留节奏与 root velocity 相关性（Pearson）。

### 4) 开源核查（步骤 2.5）
- **arXiv 页（2026-09-07）：** 无 Code / Project 链接 → 截至入库日 **代码未开源**。

## 对 wiki 的映射

- 新建 [AdaMorph 论文实体](../../wiki/entities/paper-adamorph-unified-motion-retargeting.md)
- **消歧：** 勿与 [UMR（表面点云对应）](../../wiki/entities/paper-umr-unified-motion-retargeting.md) 或 PALUM（2601.07272）混页
- 交叉 [Motion Retargeting](../../wiki/concepts/motion-retargeting.md)、[MoReFlow](../../wiki/entities/paper-moreflow-motion-retargeting-flow.md)、[GMR](../../wiki/methods/motion-retargeting-gmr.md)

## 当前提炼状态

- [x] arXiv 全文要点
- [x] 开源状态写入 wiki
