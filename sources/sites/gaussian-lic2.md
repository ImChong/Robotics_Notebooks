# Gaussian-LIC2 项目页（xingxingzuo.github.io/gaussian_lic2）

> 来源归档（ingest 配套站点）

- **URL：** <https://xingxingzuo.github.io/gaussian_lic2/>
- **标题：** Gaussian-LIC2: LiDAR-Inertial-Camera Gaussian Splatting SLAM
- **机构：** 浙江大学控制科学与工程学院 · 穆罕默德·本·扎耶德人工智能大学（MBZUAI）机器人系
- **论文：** <https://arxiv.org/abs/2507.04004> — 归档见 [`sources/papers/gaussian_lic2_arxiv_2507_04004.md`](../papers/gaussian_lic2_arxiv_2507_04004.md)
- **代码：** <https://github.com/APRIL-ZJU/Gaussian-LIC> — [`sources/repos/gaussian-lic.md`](../repos/gaussian-lic.md)
- **视频：** <https://www.youtube.com/watch?v=SkPnpuCfh88>
- **入库日期：** 2026-08-20
- **一句话说明：** 官方落地页：实时 LIC 3DGS-SLAM；连续时间里程计 + 深度补全初始化 + LiDAR 深度监督建图；展示 in-seq / out-of-seq RGB·深度渲染与帧插值、网格提取应用。

## 公开信息要点（截至入库日）

| 项 | 状态 |
|----|------|
| **Paper / BibTeX** | 已链 arXiv:2507.04004；前作 Gaussian-LIC（ICRA 2025） |
| **Demo 视频** | YouTube / bilibili 已发布 |
| **方法叙事** | 两模块：连续时间紧耦合 LIC 里程计 + 增量 3DGS 建图；零样本深度补全填 LiDAR 盲区 |
| **代码** | **已开源** — GitHub `APRIL-ZJU/Gaussian-LIC`（IJRR 2026 / ICRA 2025） |
| **数据集** | 论文宣称自采 LIC 数据集 — **待发布**（仓库 checklist） |
| **结论** | 项目页 + 代码可用于复现与定性对比；自采评测数据待跟进 |

## 页面结构速记

1. **Pipeline** — 前端 Coco-LIC 式连续时间因子图；后端深度补全 + CUDA 加速 3DGS。
2. **In-Sequence NVS** — 训练轨迹上的 RGB / 深度 novel view。
3. **Out-of-Sequence NVS** — 绿线训练、红线外推评测；自采数据集支撑。
4. **Applications** — 视频帧插值（连续时间轨迹 + 3DGS）；高斯地图快速网格提取。

## 关联资料

- 论文摘录：[`sources/papers/gaussian_lic2_arxiv_2507_04004.md`](../papers/gaussian_lic2_arxiv_2507_04004.md)
- 代码仓：[`sources/repos/gaussian-lic.md`](../repos/gaussian-lic.md)
- Wiki 实体：[`wiki/entities/paper-gaussian-lic2.md`](../../wiki/entities/paper-gaussian-lic2.md)
