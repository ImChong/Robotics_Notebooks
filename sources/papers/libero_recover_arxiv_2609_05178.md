# LIBERO-Recover（arXiv:2609.05178）

> 来源归档（ingest）

- **标题：** LIBERO-Recover: Beyond Task Success Towards Failure Recovery in Robotic Manipulation Models
- **简称：** LIBERO-Recover
- **类型：** paper / benchmark / failure-recovery / libero / vla
- **arXiv：** <https://arxiv.org/abs/2609.05178>
- **PDF：** <https://arxiv.org/pdf/2609.05178>
- **项目页：** <https://liulin815.github.io/LIBERO-Recovery/> — 归档见 [`sources/sites/libero-recovery-github-io.md`](../sites/libero-recovery-github-io.md)
- **代码：** <https://github.com/liulin815/LIBERO-Recovery> — 归档见 [`sources/repos/liulin815-libero-recovery.md`](../repos/liulin815-libero-recovery.md)
- **数据：** ModelScope `ataier/LIBERO_Recovery_Expert`（3184 人类恢复 demo）、`LIBERO_Recovery_Assets`（评测场景）、`LIBERO_10_LL`（失败前历史）
- **机构：** 大连理工大学（DUT，Huchuan Lu 组等）
- **入库日期：** 2026-09-08
- **一句话说明：** 从 SOTA 具身模型真实执行失败构造 2178 恢复场景（L1–L4），把评测从「能否成功」转向「失败后能否恢复」；六模型 RSR 普遍比标准 LIBERO 跌 50%+。

## 开源状态（步骤 2.5，2026-09-08）

| 组件 | 状态 |
|------|------|
| 项目页 | 已上线；链 GitHub + ModelScope 三数据集 |
| GitHub | **已开源** — starVLA 评测管线（policy server + MuJoCo client + 聚合） |
| ModelScope | **已发布** Expert / Assets / LIBERO_10_LL |

**结论：已开源** — 评测栈与训练数据可获取；论文仍为 double-blind 匿名页。

## 核心摘录

### 摘录 1：四级恢复难度

- L1 Action Retry：计划仍成立，重试动作即可。
- L2 Action Adaptation：微调下一步动作。
- L3 Object State Recovery：需推理物体状态。
- L4 Environmental Recovery：需恢复环境/拓扑阻塞状态。
- **2178** 场景来自 **真实模型执行失败**，非人工扰动初态。

**对 wiki 的映射：** [paper-libero-recover](../../wiki/entities/paper-libero-recover.md)

### 摘录 2：关键实验发现

- 六模型（OpenVLA-OFT、π₀、π₀-FAST、GR00T-N1.5、Wan2-Policy、Cosmos-Predict2-Policy）在真实失败态上 **普遍跌超 50%**；标准榜排名 **不能** 预测恢复能力。
- L1/L2 远高于 L3/L4；L2→L3 是从动作修正到状态恢复的未解鸿沟。
- chunk 4→32 时 RSR 单调下降；WAM（Cosmos/Wan）RC 高于传统 VLA。
- 联合训练 recovery 数据可升 RSR，但对标准 LIBERO 成功率 **几乎无增益甚至略降**。

**对 wiki 的映射：** [paper-libero-recover](../../wiki/entities/paper-libero-recover.md)

## 当前提炼状态

- [x] 项目页、GitHub、ModelScope 核查（2026-09-08）
- [x] wiki 映射：`wiki/entities/paper-libero-recover.md`
