# DriveSTA：模型权重与备选轨迹库 / Released Checkpoints and Trajectory Banks

- **官方模型卡：** https://huggingface.co/Rtwotwo/DriveSTA
- **资产版本：** [Hugging Face 792e1a4](https://huggingface.co/Rtwotwo/DriveSTA/tree/792e1a4c6f4be3beb1cdf40bd34342207b16782a)
- **源码版本：** [Rtwotwo/DriveSTA f89bcf9](https://github.com/Rtwotwo/DriveSTA/tree/f89bcf98233e417bb44e515302b712fec80d3c3d)
- **核对日期：** 2026-11-26
- **核对范围：** 公开模型卡、Hub tree 和 LFS 指针元数据、资产索引及发布验证摘要；对照源码清单、兼容路径映射和下载器。未下载或加载模型权重、未解压轨迹分片、未重跑基准。
- **通读：** 已完成

## 四个 checkpoint：区分 profile、本地路径与 Hub 文件

| CLI profile | 阶段 | 下载器的本地名称 | 固定 Hub 版本中的文件 |
|---|---|---|---|
| `base_proposal` | Base Stage 1，候选生成与评分 | `base_proposal.ckpt` | `base_s1.ckpt` |
| `base` | Base Stage 2，含时序精修 | `base.ckpt` | `base_s2.ckpt` |
| `scale_proposal` | Scale Stage 1，候选生成与评分 | `scale_proposal.ckpt` | `scale_s1.ckpt` |
| `scale` | Scale Stage 2，含时序精修 | `scale.ckpt` | `scale_s2.ckpt` |

以上路径均在 `weights/Checkpoints/`。完整 checkpoint 含推理所需 backbone 权重，不需要单独下载训练初始化权重。不要将 README 的本地新名当成固定 Hub 版本里的直接下载文件名。

## 轨迹库与训练集不是同一个口径

| 发布集合 | 场景数 | 分片数 |
|---|---:|---:|
| Navtrainval | 103,288 | 16 |
| SimScale | 204,831 | 32 |
| 合计 | 308,119 | 48 |

另有 1 个元数据归档，包含固定训练索引与定位分片所需信息。每场景有 3,072 条带评分候选，每条为自车系 `8 × (x, y, heading)`，0.5 秒间隔，另有可用高分子集。原始挖掘的高分子集不足标记不等于文件损坏。**固定 Base 配方只用 103,263 条真实场景；Scale 用 103,263 个真实 + 129,409 个合成场景，共 232,672。** 发布库不含相机原图、原始日志、地图、特征或评分缓存，也不含另行构建的 EPDMS 轨迹库。

## 如何分层核验

1. 用 `python scripts/download_assets.py --profiles base --plan` 预览本地清单，不发起下载；实际复现可加 `--revision 792e1a4c6f4be3beb1cdf40bd34342207b16782a` 固定 Hub 版本。下载器解析兼容名称，验证字节数和 SHA-256，再把文件写到本地规范路径。
2. 需要特定场景时先查 [scene-to-shard 索引](https://huggingface.co/Rtwotwo/DriveSTA/blob/792e1a4c6f4be3beb1cdf40bd34342207b16782a/manifests/scene_to_shard.csv)，再按 [官方下载/修复指南](https://github.com/Rtwotwo/DriveSTA/blob/f89bcf98233e417bb44e515302b712fec80d3c3d/docs/DOWNLOAD_VERIFY.md) 定位分片，避免整包下载。
3. 本次核查：4 个 checkpoint、48 个轨迹分片和 1 个元数据包的 **Hub LFS 声明大小及 SHA-256** 均与代码发布清单一致。这是指针元数据比对，不是对实际下载字节的校验。源码清单另列 DINOv2 和 Drive-JEPA 两个训练初始化文件，但固定 Hub 目录中没有这两项，应按 [训练数据准备文档](https://github.com/Rtwotwo/DriveSTA/blob/f89bcf98233e417bb44e515302b712fec80d3c3d/docs/DATA_PREPARATION.md#4-prepare-caches-and-initialization-backbones) 分别取自上游。
4. [作者发布验证摘要](https://huggingface.co/Rtwotwo/DriveSTA/blob/792e1a4c6f4be3beb1cdf40bd34342207b16782a/docs/release/validation_summary.json) 记录软件测试与真实分片损坏检测/修复；它明确不是重跑完整训练或 benchmark。模型卡标注 Apache-2.0，第三方原始数据及上游模型仍各有许可。

- **Wiki：** [DriveSTA 论文与工程复现](../../wiki/entities/paper-drivesta.md)
