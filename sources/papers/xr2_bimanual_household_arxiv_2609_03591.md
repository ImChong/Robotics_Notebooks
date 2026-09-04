# Scaling Bimanual Household Manipulation / XR-2（arXiv:2609.03591）

> 来源归档（ingest）

- **标题：** Scaling Bimanual Household Manipulation from 1,500 hours of Demonstrations to On-Policy Corrections
- **简称：** XR-2
- **类型：** paper / vla / bimanual / dataset / household
- **arXiv：** <https://arxiv.org/abs/2609.03591>
- **PDF：** <https://arxiv.org/pdf/2609.03591>
- **数据集：** <https://huggingface.co/datasets/challenge-2026/challenge_data> — 归档见 [`sources/sites/hf-challenge-2026.md`](../sites/hf-challenge-2026.md)
- **代码：** 截至 2026-09-04 **未见** XR-2 训练/推理官方仓
- **机构：** PrimeBot Research Institute / 上纬新材、北京大学
- **入库日期：** 2026-09-04
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_9_papers_open_source_2026-09-04.md)
- **一句话说明：** 释放 1500 小时多样化双臂家务示范并训练 XR-2 VLA；示范规模与实时 DAgger 修正两条轴均呈稳定成功率提升。

## 开源状态（步骤 2.5，2026-09-04）

| 组件 | 状态 |
|------|------|
| HF 数据集 | **已开源** `challenge-2026/challenge_data`，CC-BY-SA-4.0，LeRobot 格式 |
| XR-2 模型代码/权重 | **未见** |

**结论：部分开源** — 可下载示范数据做自己的训练；官方策略栈待发布。

## 核心摘录

### 摘录 1：两条 scaling 轴

- 专家示范数据量。
- 来自实时人类干预的 DAgger correction。
- 高吞吐采集管线 + 多阶段训练。

**对 wiki 的映射：** [paper-xr2-bimanual-household](../../wiki/entities/paper-xr2-bimanual-household.md)

## 当前提炼状态

- [x] HF 数据集核查（2026-09-04）
- [x] wiki 映射：`wiki/entities/paper-xr2-bimanual-household.md`
