# Humanoid Safe Stop via Learned Stoppability Value（arXiv:2609.02358）

> 来源归档（ingest）

- **标题：** Humanoid Safe Stop via Learned Stoppability Value
- **简称：** Safe-Stop
- **类型：** paper / humanoid / safe-stop / reach-avoid
- **arXiv：** <https://arxiv.org/abs/2609.02358>
- **PDF：** <https://arxiv.org/pdf/2609.02358>
- **项目页：** <https://junfeng-long.github.io/safestop/> — 归档见 [`sources/sites/safestop.md`](../sites/safestop.md)
- **代码：** 截至 2026-09-04 **无** GitHub 链接（项目页仅 Paper / arXiv / Video）
- **机构：** 加州大学伯克利（UC Berkeley）、卡内基梅隆大学（CMU）、斯坦福大学（Stanford）
- **入库日期：** 2026-09-03
- **最后更新：** 2026-09-04
- **索引来源：** [具身智能小站 8 篇盘点](../blogs/wechat_embodied_station_8_papers_open_source_2026-09-03.md)；2026-09-04 与 [WM-LOCO](wm_loco_arxiv_2609_02542.md)、[FOCUS](focus_foot_observation_confidence_arxiv_2609_02222.md) 并行升格，**不另造第二详情节点**
- **一句话说明：** 将急停建模为 reach-avoid；learned stop policy + 双互补 stoppability 估计器；仅当两者均认为可停才提交停止，否则 damping fallback。

## 开源状态（步骤 2.5，2026-09-04 再核）

| 组件 | 状态 |
|------|------|
| 项目页 | 已上线（<https://junfeng-long.github.io/safestop/>） |
| GitHub / 权重 | **未见** |

**结论：待发布**

## 核心摘录

### 摘录 1：双估计器

- stop-probability：固定停止策略实际结果监督。
- reach-avoidance：物理状态 Hamilton-Jacobi backup 监督。

**对 wiki 的映射：** [paper-safe-stop-humanoid](../../wiki/entities/paper-safe-stop-humanoid.md)

### 摘录 2：Unitree G1

- OOD 停止成功 **96.4%**；unsafe-approval **3.89%**。

**对 wiki 的映射：** [paper-safe-stop-humanoid](../../wiki/entities/paper-safe-stop-humanoid.md)

## 当前提炼状态

- [x] 项目页核查
- [x] wiki 映射
