# ashish-kmr.github.io/rma-legged-robots（RMA 项目页）

- **标题：** RMA: Rapid Motor Adaptation for Legged Robots — 官方项目页
- **类型：** site / project-page
- **URL：** <https://ashish-kmr.github.io/rma-legged-robots/>
- **入库日期：** 2026-06-11
- **配套论文：** [RMA（arXiv:2107.04034）](https://arxiv.org/abs/2107.04034) — 归档见 [`sources/papers/rma_arxiv_2107_04034.md`](../papers/rma_arxiv_2107_04034.md)
- **方法视频：** <https://youtu.be/qKF6dr_S-wQ>

## 一句话摘要

Berkeley / CMU / FAIR 的 **RMA（RSS 2021）** 官方站点：展示 A1 在 **户外岩石、油滑、植被、楼梯、泡沫床垫** 等多样地形上的 **零微调 sim2real** 行走；含与 **A1 原厂控制器** 对照、各场景精选视频与 BibTeX。

## 公开信息要点（截至入库日）

- **核心主张：** base policy + adaptation module 在 **不到一秒** 内在线适应未见场景；**完全仿真训练**，无参考轨迹 / 足端轨迹生成器；A1 **无 fine-tuning** 部署。
- **Selected demos（页面分区）：** Outdoor / Indoor runs；Rocky & Pebble；Unstable & Loose ground；Vegetation & Grass & Hiking stairs；**Oil slip test**；Plank test；Foam & Mattress；Step-up / Step-down；**vs A1 controller** 并排对比。
- **同组后续工作（页脚 Incremental Growth）：**
  - *Minimizing Energy Consumption Leads to the Emergence of Gaits*（Fu et al., CoRL 2021）
  - *Coupling Vision and Proprioception for Navigation of Legged Robots*（Fu et al.）— 视觉 + 本体导航扩展
- **BibTeX：** `@inproceedings{kumar2021rma, ... Robotics: Science and Systems, year={2021}}`

## 为何值得保留

- **非 PDF 证据：** 油滑、床垫、植被等 **动态失败模式** 比论文静图更直观，且与 adaptation 消融叙事互证。
- **与 arXiv 三角互证：** 摘要、异步 10/100 Hz 部署、生物力学奖励等细节与 PDF 一致。
- **谱系锚点：** 后续 Extreme Parkour、ROA、CMS 视觉 locomotion 等均引用 RMA 式 **特权→历史估计** 蒸馏。

## 关联资料

- 论文归档：[`sources/papers/rma_arxiv_2107_04034.md`](../papers/rma_arxiv_2107_04034.md)
- 训练代码（Raisim / CMS 基座）：[`sources/repos/antonilo_rl_locomotion.md`](../repos/antonilo_rl_locomotion.md)
- Wiki 实体：[`wiki/entities/paper-rma-rapid-motor-adaptation.md`](../../wiki/entities/paper-rma-rapid-motor-adaptation.md)
