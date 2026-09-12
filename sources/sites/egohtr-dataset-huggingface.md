# EgoHTR 数据集（Hugging Face）

- **标题:** EgoHTR — Egocentric Human-Terrain Reconstruction Dataset
- **类型:** dataset / huggingface
- **链接:** <https://huggingface.co/datasets/leggedrobotics/egohtr>
- **论文:** [EgoHTR（arXiv:2607.13472）](https://arxiv.org/abs/2607.13472)
- **项目页:** <https://egohtr.github.io>
- **机构:** 苏黎世联邦理工（ETH Zürich）× 斯坦福大学（Stanford）× 加州大学伯克利分校（UC Berkeley）× 慕尼黑工业大学（TU Munich）
- **收录日期:** 2026-09-12

## 一句话摘要

ETH RSL 在 Hugging Face 发布的 **rough-terrain 人–场景 4D** 多模态数据集（**55** 序列 / ~**1.37 h** / ~**150k** 帧 @ 30 fps）；含 Aria ego/exo、IMU MoCap、3D 场景扫描、SMPL-X 与可选机器人 retarget；总体量约 **719 GB**；**重建/训练代码仍待发布**。

## 开放状态（截至 2026-09-12）

| 项 | 状态 |
|----|------|
| **数据集** | **已发布** — <https://huggingface.co/datasets/leggedrobotics/egohtr>（项目页 Dataset 按钮已指向 HF；访问可能需 HF 登录/授权） |
| **重建管线代码** | **待发布** — 项目页 **Code (coming soon)**；HF README 写「Generation pipeline (code): Github」但尚无公开复现仓 |
| **GitHub org** | <https://github.com/egohtr> 仍仅 [`egohtr/egohtr.github.io`](https://github.com/egohtr/egohtr.github.io) 站点仓 |

## 模态与目录结构（HF README 摘要）

每条序列文件夹含压缩处理后输出：

| 文件 | 说明 |
|------|------|
| `full_sequence.npz` | 完整同步数据 |
| `full_opt_sequence.npz` | 优化后序列 |
| `full_retarget_sequence.npz` | 可选：重定向到机器人 embodiment |

原始传感与管线输出子目录：

| 目录 | 内容 |
|------|------|
| `aria/` | Project Aria egocentric RGB + SLAM 轨迹 + 手部跟踪 |
| `egomc/` | Rokoko IMU 服 MoCap（BVH + raw） |
| `exomc/` | 可选 marker MoCap（评测 GT） |
| `scene/` | BLK2GO 点云与纹理 mesh |
| `retarget/` | 重定向到机器人（URDF/XML） |

字段级 schema 以未来发布的 **EgoHTR pipeline 仓库** README 为准。

## 直接用途（官方）

- 4D 人–场景重建 / HMR 基准（相对 MoCap GT）
- 感知、地形感知全身控制 / 人形 locomotion 策略训练与评测（论文演示 Unitree G1）
- 经配套管线将人体运动 retarget 到机器人 embodiment

## 对 Wiki 的映射

- [`wiki/entities/paper-egohtr.md`](../../wiki/entities/paper-egohtr.md) — 论文与数据集实体页
- [`sources/sites/egohtr-github-io.md`](egohtr-github-io.md) — 项目页开源核查主入口
- [`sources/papers/egohtr_arxiv_2607_13472.md`](../papers/egohtr_arxiv_2607_13472.md) — 论文摘录
