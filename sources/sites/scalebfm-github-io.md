# scalebfm.github.io（ScaleBFM 项目页）

- **标题：** Scaling Behavior Foundation Model for Humanoid Robots — 官方项目页
- **类型：** site / project-page
- **URL：** <https://scalebfm.github.io/>
- **入库日期：** 2026-07-18
- **配套论文：** [ScaleBFM（arXiv:2607.15163）](https://arxiv.org/abs/2607.15163) — 归档见 [`sources/papers/scaling_bfm_arxiv_2607_15163.md`](../papers/scaling_bfm_arxiv_2607_15163.md)

## 一句话摘要

ScaleBFM 技术报告官方项目页：展示 **on-policy 数据 / 参考运动 / Humanoid Transformer** 三维 scaling 实验曲线、**八模式稀疏控制接口** 真机演示，以及 loco-manipulation、高动态 locomotion 等行为画廊；页头链到论文 PDF 与 GitHub 代码仓。

## 公开信息要点（截至 2026-07-18 核查）

- **作者与机构**：Weishuai Zeng、Kangning Yin、Xiaojie Niu 等；港中大、上交大、浙大、北大、清华、Galbot、**上海人工智能实验室**。
- **代码：** 页头 **Code** 按钮指向 <https://github.com/zengweishuai/ScaleBFM>。仓库已创建（2026-07-16），当前 **仅 README**；作者公告 **2026-07-26 前** 逐步释出重定向、训练、部署等组件 → 归类 **部分开源 / 待发布**。
- **演示分组**（页面 Gallery / Videos）：
  - **灵巧操作**：插花、捡水果、开电饭煲、插拔充电器、倒水等。
  - **敏捷 locomotion**：拳击、侧手翻、太极、随机行走、后空翻等。
  - **全身 loco-manipulation**：开柜取物、洗衣、捡垃圾、购物、铺床、羽毛球、宠物互动等。
  - **稀疏约束八模式**：Root / Bimanual / Root-and-Hand / End-Effector / Root-and-End-Effector / Upper-Body / Whole-Body 及 local/global 对照视频。
- **Scaling 可视化**：on-policy width×depth、reference motion 五档分区（XXS–L）、MLP vs Humanoid Transformer 规模曲线、潜空间 3D 投影与跨模式收敛对比。

## 为何值得保留

- **BFM scaling 一手证据**：比 arXiv PDF 更易浏览的交互曲线与真机视频，便于与 [SONIC](../../wiki/methods/sonic-motion-tracking.md) 等 scaling 叙事对照。
- **代码开放跟进锚点**：README 给出明确释出时间表，后续 lint 可据此更新 `sources/repos/scalebfm.md` 开放状态。
- **与 CVAE-BFM 项目页区分**：[bfm4humanoid.github.io](https://bfm4humanoid.github.io/) 对应 arXiv:2509.13780；本页对应 **2607.15163** scaling 技术报告，勿混用。

## 关联资料

- 论文归档：[`sources/papers/scaling_bfm_arxiv_2607_15163.md`](../papers/scaling_bfm_arxiv_2607_15163.md)
- 代码归档：[`sources/repos/scalebfm.md`](../repos/scalebfm.md)
- 同团队 CVAE-BFM：[`sources/sites/bfm4humanoid-github-io.md`](bfm4humanoid-github-io.md)
- wiki 实体：[`wiki/entities/paper-scaling-bfm-humanoid.md`](../../wiki/entities/paper-scaling-bfm-humanoid.md)
