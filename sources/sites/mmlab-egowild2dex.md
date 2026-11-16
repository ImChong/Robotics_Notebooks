# EgoWild2Dex 项目页（mmlab.hk）

- **标题：** EgoWild2Dex: Learning Dexterous Robotic Manipulation from In-the-Wild Human Experience
- **类型：** site / project-page
- **URL：** <https://mmlab.hk/egowild2dex/>
- **论文：** [arXiv:2609.23755](https://arxiv.org/abs/2609.23755) — 归档见 [`sources/papers/egowild2dex_arxiv_2609_23755.md`](../papers/egowild2dex_arxiv_2609_23755.md)
- **PDF：** <https://arxiv.org/pdf/2609.23755>
- **机构：** 香港大学（The University of Hong Kong，Ping Luo* 等）；Kinetix AI；项目 lead：Kunyang Lin†
- **入库日期：** 2026-09-27
- **复核日期：** 2026-11-10
- **核对范围：** 官方主页 Learning from in-the-wild ego-human data、GeoFormer、Progressive Human–Robot Training、EgoWild、Results、Cross-Embodiment 与 Citation 全部公开文字板块及资源链接
- **通读：** 已完成

## 一句话摘要

野外第一视角人类经验 → **GeoFormer** 轻量视角对齐 + **三阶段渐进人–机训练** + **EgoWild**（538.9 h / 179k episodes）；真机三项长时程双手灵巧任务平均 **96.7%** 成功率（每任务 <1 h 机器人示范）。

## 页面公开板块（截至入库日）

1. Learning from in-the-wild ego-human data — 动机与三贡献
2. **GeoFormer** — 可微单应 warp，训练时用 \(T^{-1}\) 把 ego 帧对齐到 robot 视角；相对 Project+Inpaint **21.9×** 加速
3. **Progressive Human–Robot Training** — Human-to-Robot → Human–Robot co-training → Robot refinement；ego / glove / robot 三路数据统一到 robot-native action
4. **EgoWild** — 家庭、工厂、药店、驿站等未脚本化采集；2048×1536 视频 + 双手轨迹 + 多粒度语言
5. **Results** — Open-Box / Glue-Figure / Ice-Water；跨本体（Tianji Marvin Pro）与抗扰/自恢复视频
6. Citation — BibTeX

## 开源核查（步骤 2.5，2026-11-10 复核）

| 资产 | 状态 |
|------|------|
| 项目页 Code / GitHub 链 | **未见** EgoWild2Dex 官方可运行仓库链接；项目页技术报告指向 arXiv |
| 论文 Abstract | 「The data, models, and code **will be released**」 |
| EgoWild 数据集 | **尚无页面可核验的公开下载地址**，论文承诺未来发布 |
| GeoFormer / 训练代码 | **尚无页面可核验的训练 / 推理入口**，论文承诺未来发布 |

## 为何值得保留

- **数据形态：** 比 staged ego 数据集更 clutter、更高头动（15.93°/s 累计旋转、34.78 前景实例/帧）。
- **方法闭环：** 视角对齐（GeoFormer）与 embodiment 对齐（IK + 手指 retarget + 渐进训练）在同一 VLA + flow-matching 目标下。
- **工程读点：** 每任务 **<1 h** 机器人数据即可长时程灵巧任务高成功率，对标 [EgoDex](../../wiki/entities/paper-notebook-egodex-learning-dexterous-manipulation-from-larg.md) 等「只预训练、少真机」路线。

## 对 wiki 的映射

- [paper-egowild2dex](../../wiki/entities/paper-egowild2dex.md)

## 与 arXiv v1 交叉核验（2026-11-10）

- 项目页讲述 robot→ego 学 T、对 ego 使用 T⁻¹，和正文 §III-A 一致；数值 **21.9×** 来自原文 Table III 图像预处理延迟 **162.58→7.43 ms**，不是整机控制加速。
- 项目页展示 Open-Box、Glue-Figure、Ice-Water 和 Tianji Marvin Pro 迁移视频；量化成功率、消融、数据时长应以 [论文 v1 原文归档](../papers/egowild2dex_arxiv_2609_23755.md) 中的 Table I–IV/IX 为准。
- 页面报告 EgoWild 538.9 h / 179,049 episodes / 125,961 unique descriptions / 1,282 object categories；头部手部追踪降低指尖误差 4.21→0.66 cm，和论文附录 C 一致。
- 页面 **没有**单独的下载/安装/运行指南；XRoboToolkit 仅是论文引用的遥操作依赖，不能充作作者已公开的 GeoFormer 实现。
