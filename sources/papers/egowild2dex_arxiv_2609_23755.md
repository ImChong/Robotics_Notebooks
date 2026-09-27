# EgoWild2Dex（arXiv:2609.23755）

> 来源归档（ingest）

- **标题：** EgoWild2Dex: Learning Dexterous Robotic Manipulation from In-the-Wild Human Experience
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.23755>
- **PDF：** <https://arxiv.org/pdf/2609.23755>
- **项目页：** <https://mmlab.hk/egowild2dex/> — 归档见 [`sources/sites/mmlab-egowild2dex.md`](../sites/mmlab-egowild2dex.md)
- **机构：** 香港大学；Kinetix AI（Ping Luo* 通讯；Kunyang Lin† project lead）
- **入库日期：** 2026-09-27
- **一句话说明：** 野外 ego 数据 + GeoFormer 视角对齐 + 三阶段人–机渐进训练 → 双手灵巧 VLA；发布 EgoWild 538.9 h。

## 开源状态（步骤 2.5，2026-09-27）

- **待发布：** 论文与项目页均写明 data / models / code **将发布**；截至入库日 **无** 官方 GitHub / Hugging Face 链接。

## 核心摘录（面向 wiki 编译）

- **GeoFormer（Geometric Transformer）：** 从无配对 ego/robot 图像学习 **8 维单应** 残差级联；部署时对 ego 应用 **\(T^{-1}\)** 对齐 robot 固定相机；对抗训练 + 几何正则；比 3D project+inpaint **快 21.9×**。
- **渐进训练：** (1) **538.9 h EgoWild** 预训练 VLM 先验；(2) **任务相关** GeoFormer 对齐后的 ego + **glove**（robot 视角、精确腕指）+ **少量 robot** 共训；(3) **robot 示范与 recovery** 微调。各阶段共享 **robot-native action** 与 **flow-matching** 目标。
- **动作对齐：** 共享参考系 + 臂 **IK** + 手指 **retarget** → 统一高 DoF 手–臂命令。
- **EgoWild：** 179,049 episodes；125,961 唯一任务描述；1,282 物体类；校准手追踪指尖中位误差 **4.21 cm → 0.66 cm**。
- **真机评测：** 三项长时程（Open-Box / Glue-Figure / Ice-Water）；平均成功率 **96.7%**（每任务 **<1 h** robot 数据）；未见物体 **33.3%** 平均零样本物体级成功率；跨本体 **Tianji Marvin Pro**。
- **对 wiki 的映射：** [paper-egowild2dex](../../wiki/entities/paper-egowild2dex.md)；交叉 [EgoDex](../../wiki/entities/paper-notebook-egodex-learning-dexterous-manipulation-from-larg.md)、[motion-retargeting](../../wiki/concepts/motion-retargeting.md)、[VLA](../../wiki/methods/vla.md)

## 当前提炼状态

- [x] 项目页与 arXiv 摘要交叉核查
- [x] wiki 映射：`wiki/entities/paper-egowild2dex.md` 新建
