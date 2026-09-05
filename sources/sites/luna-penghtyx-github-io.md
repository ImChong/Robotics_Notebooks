# LUNA 项目页（penghtyx.github.io/LUNA）

> 来源归档（ingest 配套站点）

- **URL：** <https://penghtyx.github.io/LUNA/>
- **对应论文：** [LUNA: Learning Universal 3D Human Animation Beyond Skinning](https://arxiv.org/abs/2606.31981)（arXiv:2606.31981，ECCV 2026，HKUST + Meta Codec Avatars Lab）
- **入库日期：** 2026-09-05
- **一句话说明：** 官方落地页：摘要、方法示意图、RGB / 关键点 / 草图驱动视频、高斯轨迹、松衣自驱动对照、与 LBS 基线对比、BibTeX。
- **论文：** <https://arxiv.org/abs/2606.31981>
- **代码：** 截至 **2026-09-05** 未发布（见下方步骤 2.5）

## 页面要点（2026-09-05 快照）

### 核心主张

**LBS-free universal neural animation：少数身份图 + 隐式 2D 驱动（RGB / keypoints / sketch / 未见角色）→ 3D Gaussian 形变，不做显式人体拟合。**

### 页头元数据

| 项 | 内容 |
|----|------|
| 单位 | ¹ HKUST；² Codec Avatars Lab, Meta |
| 会议 | ECCV 2026 |
| 作者链 | Peng Li（[主页](https://penghtyx.github.io/yuki-lipeng/)）、Rawal Khirodkar、Junxuan Li、Yuan Liu、Wenhan Luo、Shunsuke Saito 等 |
| 方法说明 | Identity Encoder（\(N=4\) 未标定多视角）→ Implicit Neural Animator（2D 条件） |

### 结果展示（项目页视频/图）

| 区块 | 展示内容 |
|------|----------|
| RGB-Driven | 驱动 RGB 视频 → 动画结果 |
| Keypoints-Driven | 2D 骨架视频 → 动画结果 |
| Sketch-Driven | 手绘草图视频 → 动画结果 |
| Gaussian Trajectory | 四组高斯轨迹样例 |
| Clothing Self-enactment | 松衣自驱动；对照 MV-LHM + SAM 3D Body |
| Comparison with LBS | 与 LBS 管线对照 |

### BibTeX（项目页 Citation）

以 arXiv:2606.31981 为准；项目页提供 Citation 按钮。

## 步骤 2.5：源码 / 数据开放核查（2026-09-05）

1. **页头资源栏：** 可见 `📄 arXiv`，目标仅为 `https://arxiv.org/`（未绑 `abs/2606.31981`）。
2. **GitHub 按钮：** HTML 中为注释占位 `<!-- <a href="#" target="_blank">🐙 GitHub</a> -->`，**无真实仓库 URL**。
3. **Hugging Face / 权重 / 数据集：** 页上未列。
4. **论文正文：** 未给出可运行实现或数据下载入口；Video35K / iPhone1K / Dome 为专有或同源受限集合。
5. **开放程度：宣称将开源 / 截至入库日未列可用链接。** 不建 `sources/repos/`。后续若按钮上线，再补仓库页并改实体页开源表。

## 对 wiki 的映射

- [LUNA 论文实体](../../wiki/entities/paper-luna-universal-3d-human-animation.md)
- [LUNA 论文摘录](../papers/luna_arxiv_2606_31981.md)
