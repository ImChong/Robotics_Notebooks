# 立体匹配基础模型与基准 — 生态书目（NBS 对照轴）

> 来源归档（ingest · 2026-09-09）  
> 本文件汇总 [NBS](nbs_arxiv_2608_28933.md) 论文与项目页对照的 **基线方法、骨干与 benchmark**；各条可升格 wiki 方法页/实体页交叉引用。

## 基线方法

### S²M² — Scalable Stereo Matching Model（ICCV 2025）

- **论文：** <https://arxiv.org/abs/2507.13229>
- **项目页：** <https://junhong-3dv.github.io/s2m2-project/>
- **代码：** <https://github.com/junhong-3dv/s2m2>（**已开源**）
- **一句话：** 可扩展双目匹配基础模型；Middlebury / ETH3D 等可靠深度；NBS 项目页列为强对照。
- **wiki：** [`wiki/methods/stereo-matching-foundation-models.md`](../../wiki/methods/stereo-matching-foundation-models.md)

### FoundationStereo — Zero-Shot Stereo Matching（CVPR 2025）

- **论文：** <https://arxiv.org/abs/2501.09898>
- **项目页：** <https://nvlabs.github.io/FoundationStereo/>
- **代码：** <https://github.com/NVlabs/FoundationStereo>（**已开源**）
- **一句话：** NVIDIA 零样本立体基础模型；机器人/Real2Sim 栈常见（NuRec、OSMO、LadderMan Fast-FoundationStereo 等）。
- **repos：** [`sources/repos/nvlabs_foundation_stereo.md`](../repos/nvlabs_foundation_stereo.md)

### Selective-Stereo / Selective-IGEV

- **论文：** <https://arxiv.org/abs/2403.00486>
- **代码：** <https://github.com/Windsrain/Selective-Stereo>（Selective-IGEV 在 `Selective-IGEV/` 子目录）
- **一句话：** 自适应频域信息选择；NBS 交互对比之一。

### IGEV-Stereo（CVPR 2023）

- **论文：** <https://arxiv.org/abs/2303.06615>
- **代码：** <https://github.com/gangweix/IGEV>（**已开源**）
- **一句话：** Iterative Geometry Encoding Volume；Selective-IGEV 的上游。

### CroCo / CroCo v2

- **CroCo 论文：** <https://arxiv.org/abs/2210.10716>（NeurIPS 2022）
- **CroCo v2：** <https://arxiv.org/abs/2303.12017>（ICCV 2023，立体匹配与光流改进预训练）
- **代码：** <https://github.com/naver/croco>（**已开源**）

### CREStereo（CVPR 2022 Oral）

- **论文：** <https://arxiv.org/abs/2203.11483>
- **代码：** <https://github.com/megvii-research/CREStereo>（**已开源**）

### RAFT-Stereo（3DV 2021）

- **论文：** <https://arxiv.org/abs/2109.07547>
- **代码：** <https://github.com/princeton-vl/RAFT-Stereo>（**已开源**）

## NBS 使用的骨干与头

### DINOv2

- **论文：** <https://arxiv.org/abs/2304.07193>
- **代码：** <https://github.com/facebookresearch/dinov2>
- **wiki：** [`wiki/entities/paper-dinov2.md`](../../wiki/entities/paper-dinov2.md)（NBS 用 **ViT-Large 初始化**）

### DPT — Vision Transformers for Dense Prediction

- **论文：** <https://arxiv.org/abs/2103.13413>
- **代码：** <https://github.com/isl-org/DPT>
- **wiki：** [`wiki/entities/paper-dpt.md`](../../wiki/entities/paper-dpt.md)（NBS 视差解码头）

## 评测基准

| 基准 | 链接 | wiki |
|------|------|------|
| **ETH3D Two-View** | <https://www.eth3d.net/low_res_two_view.php> | [`wiki/entities/eth3d-stereo-benchmark.md`](../../wiki/entities/eth3d-stereo-benchmark.md) |
| **Middlebury V3** | <https://vision.middlebury.edu/stereo/eval3/> | [`wiki/entities/middlebury-stereo-benchmark.md`](../../wiki/entities/middlebury-stereo-benchmark.md) |
| **KITTI 2012 / 2015** | [2012](https://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo) / [2015](https://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo) | [`wiki/entities/kitti-stereo-benchmark.md`](../../wiki/entities/kitti-stereo-benchmark.md) |
