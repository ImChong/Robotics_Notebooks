# EgoExoMoCap 项目页（ECCV 2026 Spotlight）

> 来源归档（ingest）

- **标题：** EgoExoMoCap: Distributed Ego-Exo Human Motion Capture
- **类型：** site（SIPLAB / Meta Reality Labs 官方项目页）
- **发布方：** Jiaxi Jiang（Meta RLA internship + ETH SIPLAB）等
- **原始链接：** <https://siplab.org/projects/EgoExoMoCap>
- **论文：** <https://arxiv.org/abs/2607.15868>
- **PDF：** <https://static.siplab.org/papers/eccv2026-egoexomocap.pdf>
- **代码：** <https://github.com/eth-siplab/EgoExoMoCap>
- **入库日期：** 2026-09-05
- **一句话说明：** ECCV 2026 Spotlight 落地页：分布式 HMD ego-exo 动捕示意、方法管线（EgoNet / ViTPose / DINOv3 / Transformer）、Nymeria 与 EgoHumans 定性对比、BibTeX；相关项目链到 AvatarPoser、EgoPoser、MANIKIN、EgoSim、Group Inertial Poser、Ultra Inertial Poser。

## 项目页 / 源码开放核查（步骤 2.5 · 2026-09-05）

| 核查项 | 结论 |
|--------|------|
| **项目页 Code / Resources** | 页内未直接嵌 GitHub 按钮，但论文与 SIPLAB 生态惯例指向 `eth-siplab/EgoExoMoCap` |
| **GitHub 仓库** | <https://github.com/eth-siplab/EgoExoMoCap> 已存在，MIT 许可 |
| **开放程度** | **待发布**：README 仅写「Code — Coming soon」，无 `train.py` / 数据下载 / 权重链接 |
| **数据** | 实验基于 Nymeria、EgoHumans 等已有公开数据集，非本仓新发数据包 |

## 摘录要点（与论文分工）

- **对外叙事：** 两人各戴智能眼镜即可分布式动捕；ego 连续轨迹 + exo 间歇图像互补，处理 out-of-view 与严重遮挡。
- **方法卡片：** EgoNet 粗定位 → ViTPose 2D → DINOv3 置信射线 → Spatial/Temporal Transformer。
- **结果展示：** Nymeria 多活动 + EgoHumans 网球等定性对比；强调多 observer 融合收益。

## 对 wiki 的映射

- [EgoExoMoCap（论文）](../../wiki/entities/paper-egoexomocap.md)
- 姊妹归档：[论文摘录](../papers/egoexomocap_arxiv_2607_15868.md)、[代码仓](../repos/egoexomocap.md)
