# SatNav 项目页（eku127.github.io/SatNav）

> 来源归档

- **标题：** SatNav: A Scalable Benchmark for Long-Horizon UAV Vision-Language Navigation from Satellite Imagery
- **类型：** site（论文项目页、交互演示、结果与实机视频）
- **URL：** <https://eku127.github.io/SatNav/>
- **论文：** <https://arxiv.org/abs/2609.31507>
- **代码：** <https://github.com/Eku127/SatNav>（公开、MIT）
- **数据：** [Episodes](https://huggingface.co/datasets/Eku127/SatNav-Episodes-v0.1)；[Scenes](https://huggingface.co/datasets/Eku127/SatNav-Scenes-v0.1)（页面指向申请获取）；[Models](https://huggingface.co/collections/Eku127/satnav-baseline-model-zoo)
- **机构：** 香港科技大学广州校区；LASER / IDEA；香港科技大学
- **入库日期：** 2026-10-14
- **一句话说明：** 项目页展示 Boundary / Landmark / Route 交互导航演示、benchmark 结果、SwiftVLN 可切换记忆模块，以及卫星图训练策略迁移到真实 UAV 的飞行视频。

## 页面结构（维护索引）

| 区块 | 内容要点 |
|------|----------|
| Overview | 118K episode、59 个场景、18 座城市，平均轨迹 379 m |
| Interactive demo | 选择 Boundary / Landmark / Route 指令，通过转向与前进操作更新卫星视图并观察 SR、OS、SPL、NE |
| Benchmark results | Train/fine-tune 于 SatNav 后，在 seen / unseen 场景评测；展示 StreamVLN* 等基线与多项导航指标 |
| SwiftVLN | 滑动对话上下文 + 可切换历史采样、空间增强与记忆压缩 |
| Memory ablation | 对照不同记忆设计对 seen / unseen 结果的影响 |
| Real-world UAV flights | 展示三类任务的真实航拍迁移；页面称使用轻量视觉适配器 |
| Resources | 论文、MIT 代码、Hugging Face episode 数据与模型；高分辨率场景需申请 |

## 开源状态

- **代码：已开源**；官方仓库有安装步骤、合成小场景示例、训练和评测文档。
- **Episodes：公开提供**，包含 train、val_seen、val_unseen 分割。
- **模型：公开集合入口**；具体基线检查点见 Model Zoo。
- **高分辨率 GeoTIFF 场景：部分开放**；SatNav-Scenes 页面采用申请访问，亦可使用自有 API 凭据生成场景。
- 代码 MIT 不代表数据、地图内容和第三方依赖使用相同许可；应一并核对仓库 [DATA_LICENSE.md](https://github.com/Eku127/SatNav/blob/master/DATA_LICENSE.md) 与 [NOTICE](https://github.com/Eku127/SatNav/blob/master/NOTICE)。

## 对 wiki 的映射

- 主实体：[SatNav（论文实体）](../../wiki/entities/paper-satnav.md)
- 论文摘录：[satnav_arxiv_2609_31507.md](../papers/satnav_arxiv_2609_31507.md)
- 代码仓库：[eku127_satnav.md](../repos/eku127_satnav.md)
