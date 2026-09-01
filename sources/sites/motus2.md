# Motus2 项目页（motus-robotics.github.io/motus2）

> 来源归档（ingest 配套站点）

- **URL：** <https://motus-robotics.github.io/motus2/>
- **论文：** <https://arxiv.org/abs/2608.30237>
- **机构：** GensPI（生数科技）；清华大学；北京航空航天大学；北京理工大学
- **前作：** Motus（arXiv:2512.13030）— <https://motus-robotics.github.io/motus>
- **入库日期：** 2026-09-01
- **一句话说明：** Motus2 灵巧操作自进化通用世界模型项目页：统一 policy / simulator / evaluator 三接口、人数据金字塔、MBRL + Best-of-N、触觉专家与多本体真机 demo。

## 步骤 2.5 开源核查（截至 2026-09-01）

| 项 | 状态 |
|----|------|
| **项目页 Code / GitHub 区** | **未列** 训练/推理/权重仓库链接 |
| **motus-robotics GitHub 组织** | 仅 [`motus-robotics/motus-robotics.github.io`](https://github.com/motus-robotics/motus-robotics.github.io)（静态站源码） |
| **前作 Motus** | 项目页未在 Motus2 页重复挂链；生数产品线其他页称 Motus 曾开源，**不可**等同 Motus2 可复现 |
| **结论** | **未开源** — 适合读方法与 demo；复现须等官方发布代码/权重 |

## 公开信息要点

| 模块 | 要点 |
|------|------|
| **General World Model** | 共享参数一套 backbone，暴露 **Policy（WAM）**、**Simulator（动作条件世界模型）**、**Evaluator（价值模型）** 三控制接口 |
| **Chunk masks** | 联合预训练用 **joint mask**（块内 video–action 双向）；中后期用 **action-first mask**（动作不可读未来 video / value query） |
| **数据金字塔** | ~**130K h** 原始录制（单目 → 高清单目 → 立体 ego）；机端 mid-training **>100 h** 机器人轨迹 + 人对齐数据 |
| **自进化** | **DiffusionNFT** MBRL + **Best-of-N** 测试时规划；失败/次优轨迹供动力学与价值学习 |
| **记忆** | 默认 **sliding window**；另评 **global autoregression** 与 **Hybrid Memory**（MemoryWAM 式） |
| **触觉** | 轻量 **tactile expert**：backbone 去噪到中间态 → 子块执行前用触觉窗口精修；Sharpa 形变图仅作条件 |
| **硬件** | WuJi-1/2 + Tianji Marvin/Gento Luna；Sharpa Wave + Gento Luna；人对齐采集（Wuji Human Gloves） |

## Demo 能力标签（项目页筛选器）

- **Embodiment：** WuJi-1、WuJi-2、Sharpa
- **Capability：** Normal、Dexterous、Memory、Tactile、Generalize
- **示例任务：** Screw Bulb、Put Phone、Multi-Finger 等真机推理视频

## 关联资料

- 论文摘录：[`sources/papers/motus2_arxiv_2608_30237.md`](../papers/motus2_arxiv_2608_30237.md)
- Wiki：[`wiki/entities/paper-motus2.md`](../../wiki/entities/paper-motus2.md)
- 前作索引：[`wiki/entities/paper-sa-2512-13030-motus-a-unified-latent-action-world-model.md`](../../wiki/entities/paper-sa-2512-13030-motus-a-unified-latent-action-world-model.md)
- 同族产品：[`wiki/entities/paper-motubrain.md`](../../wiki/entities/paper-motubrain.md)
