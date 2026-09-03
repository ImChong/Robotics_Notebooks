# ParcelStow Demonstrations, Checkpoints, and Videos（Hugging Face）

> 来源归档（dataset）

- **标题：** ParcelStow Demonstrations, Checkpoints, and Videos
- **类型：** dataset / imitation-learning / temporal-robustness / unitree-g1
- **Hugging Face：** <https://huggingface.co/datasets/cenwerem/parcelstow>
- **组织：** cenwerem（Clinton Enwerem）
- **论文：** <https://arxiv.org/abs/2609.01453>
- **代码：** <https://github.com/coenwerem/parcelstow>
- **许可：** Apache-2.0
- **入库日期：** 2026-09-03
- **一句话说明：** Isaac Lab ParcelStow 的 Parquet / `.pt` 示范、ACT checkpoint 与示意视频；数值复现以 GitHub `data/records/` 为准，不靠视频。

## 访问要点（截至 2026-09-03）

| 项 | 内容 |
|----|------|
| 体积 | 约 **3.94 GB**；**970,565** 行（50 Hz 逐步） |
| 配置 | `parcel_insertion` / `upright_placement` / `peg_insertion` |
| 示范回合 | 297 / 315 / 325（成功专家） |
| 逐步 schema | `episode, step, task, rate, observation[147], action[16]` |
| 权重 | `checkpoints/act_stow.pt` = 论文 ACT-A；直立 / 插销 ACT **不是**标称匹配对照 |
| 下载入口 | 仓库 `python scripts/download_artifacts.py --task parcel\|upright\|peg` |
| 真机 | **无** |

卡片写明：`v1.0.0` 对应包裹插入论文；`main` 含三任务开发结果，**不是**已发布 v2。

## 关联资料

- 论文：[`sources/papers/parcelstow_arxiv_2609_01453.md`](../papers/parcelstow_arxiv_2609_01453.md)
- 代码：[`sources/repos/coenwerem-parcelstow.md`](../repos/coenwerem-parcelstow.md)
- Wiki：[`wiki/entities/paper-parcelstow.md`](../../wiki/entities/paper-parcelstow.md)
