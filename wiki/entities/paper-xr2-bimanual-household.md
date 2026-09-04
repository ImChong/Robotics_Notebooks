---
type: entity
tags: [paper, vla, bimanual, household, dataset, pku]
status: complete
updated: 2026-09-04
arxiv: "2609.03591"
related:
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../concepts/embodied-scaling-laws.md
  - ./paper-minerva-libero.md
  - ../overview/open-source-reproducibility-9-papers-technology-map.md
sources:
  - ../../sources/papers/xr2_bimanual_household_arxiv_2609_03591.md
  - ../../sources/sites/hf-challenge-2026.md
  - ../../sources/blogs/wechat_embodied_station_9_papers_open_source_2026-09-04.md
summary: "XR-2 / 1500 小时双臂家务（arXiv:2609.03591，PrimeBot × 北大）：开放多样化双臂示范并训练 XR-2 VLA；示范规模与实时 DAgger 修正均呈稳定成功率提升；HF 数据集已开源，策略代码未见。"
---

# XR-2：1500 小时双臂家务到在线修正

**Scaling Bimanual Household Manipulation from 1,500 hours of Demonstrations to On-Policy Corrections**（[arXiv:2609.03591](https://arxiv.org/abs/2609.03591)，[数据集](https://huggingface.co/datasets/challenge-2026/challenge_data)）由 **PrimeBot Research Institute / 上纬新材** 与 **北京大学** 等提出：通用双臂操作的主要瓶颈是高质量大规模人类示范不足。作者释放 **1500 小时** 多样化双臂家务示范，并用该语料训练 **XR-2** 视觉语言动作模型；同时探测两条 scaling 轴——专家示范量，以及来自实时人类干预的 **DAgger correction**。

## 一句话定义

**双臂家务机器人开始进入「数据规模化 + 在线纠错」的训练阶段，而不只是再发一个单任务 demo。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| XR-2 | XR-2 VLA | 本文在 1500 小时数据上训练的双臂策略 |
| VLA | Vision-Language-Action | 视觉-语言-动作模型 |
| DAgger | Dataset Aggregation | 用在线人类干预纠错并回灌训练 |
| HF | Hugging Face | 示范数据发布平台 |

### 数据集速查

| 维度 | 内容 |
|------|------|
| 规模 | 1500 小时多样化双臂家务示范 |
| 模态 | 多相机 RGB + 双臂本体/动作（LeRobot 格式；含头部 RealSense 视频 chunk） |
| 许可证 | CC-BY-SA-4.0 |
| 重定向就绪度 | 绑定采集本体；迁到其他双臂需重定向或按目标本体再采 |

## 为什么重要

- 纳入 [九篇盘点](../../sources/blogs/wechat_embodied_station_9_papers_open_source_2026-09-04.md) 的「开放数据」支线。
- 把家务双臂从「小数据集拟合」推到可画 scaling 曲线。
- 数据集以 LeRobot 格式公开，便于第三方复训。

## 方法

| 项 | 内容 |
|----|------|
| **机构** | PrimeBot Research Institute、北京大学 |
| **数据** | 1500 小时双臂家务；HF CC-BY-SA-4.0 |
| **训练** | 高吞吐采集管线 + 多阶段范式 |
| **开源** | **部分开源**：数据已放；XR-2 代码/权重未见 |

### 流程总览

```mermaid
flowchart LR
  tele[双臂遥操作采集] --> demo[1500h 专家示范]
  demo --> pre[多阶段训练 XR-2]
  pre --> deploy[策略 rollout]
  deploy --> dagger[实时人类干预]
  dagger --> corr[DAgger correction]
  corr --> pre
```

## 评测

论文称 XR-2 在系统实验中展现较强操作性能，并保持较高训练效率与数据利用率；在探测范围内，**示范量** 与 **DAgger 修正量** 都带来稳定成功率提升。入库日公开材料以 HF 数据集卡片与论文摘要为主，**未见** 与 LIBERO/RoboCasa 对齐的单一公开总分——引用时写「双轴 scaling 趋势」，不要编造百分比。

## 结论

**家务双臂的下一步是把示范小时数和在线纠错一起做成可复现的开放资产。**

1. **1500 小时是资产** — 先问数据是否可下，再问模型是否 SOTA。
2. **两条轴都有效** — 只堆示范或只做 DAgger 都不完整。
3. **LeRobot 格式** — 降低第三方接数据的成本。
4. **策略栈未开源** — 不能把 HF 数据集写成「XR-2 已复现」。
5. **对照小容量闭集** — [MINERVA](./paper-minerva-libero.md) 说明 LIBERO 容量低；本页走的是开放家务规模。

## 源码运行时序图

**不适用（策略）** — 截至 **2026-09-04** 未见 XR-2 官方训练/推理仓。数据集可经 Hugging Face + LeRobot 加载（卡片含 `dataloader/custom_lerobot_dataset.py`）。

## 工程实践

| 项 | 建议 |
|----|------|
| 数据入口 | `challenge-2026/challenge_data` |
| 许可证 | CC-BY-SA-4.0（衍生需保持相同许可） |
| 训练 | 自建栈；勿等待未发布的官方 XR-2 脚本 |

## 局限与风险

- **模型未开源** — 论文数字无法按官方配方复核。
- **采集偏差** — 高吞吐管线可能偏向易采集任务。
- **DAgger 成本** — 实时人类干预不便宜，scaling 曲线含人力。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| 桌面单臂 LIBERO | 闭集短任务；本页是双臂家务小时库 |
| [MINERVA](./paper-minerva-libero.md) | 问最小容量；XR-2 问数据与纠错怎么 scale |
| 纯离线 BC | 缺 on-policy 修正轴 |

## 关联页面

- [VLA](../methods/vla.md)
- [Manipulation](../tasks/manipulation.md)
- [具身规模法则](../concepts/embodied-scaling-laws.md)
- [开源可复现性 9 篇地图](../overview/open-source-reproducibility-9-papers-technology-map.md)

## 参考来源

- [xr2_bimanual_household_arxiv_2609_03591](../../sources/papers/xr2_bimanual_household_arxiv_2609_03591.md)
- [HF challenge-2026](../../sources/sites/hf-challenge-2026.md)
- [具身智能小站 2026-09-04 九篇盘点](../../sources/blogs/wechat_embodied_station_9_papers_open_source_2026-09-04.md)

## 推荐继续阅读

- [arXiv:2609.03591](https://arxiv.org/abs/2609.03591)
- [Hugging Face 数据集](https://huggingface.co/datasets/challenge-2026/challenge_data)
