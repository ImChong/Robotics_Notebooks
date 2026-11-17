# Ego-OSCAR / Stereo Cap 项目页（FPV Labs）

> 来源归档（ingest 配套站点）

- **URL：** <https://www.fpvlabs.ai/ego-oscar/cap>
- **机构站：** <https://www.fpvlabs.ai/>
- **对应论文：** [Ego-OSCAR](https://arxiv.org/abs/2608.08285)（arXiv:2608.08285）
- **数据集：** <https://huggingface.co/datasets/fpvlabs/stereo-550>
- **硬件规格：** <https://drive.google.com/file/d/1ZMgKqFdM65cAtcaI2s7SHr3Z-DljlHbe/view>
- **入库日期：** 2026-08-12
- **一句话说明：** FPV Labs 可穿戴 **Stereo Cap** 3D 落地页：帽檐立体相机 + 惯性传感叙事；HF card 由此链出论文与 Hardware Spec。
- **代码（2026-11-17 复核）：** [fpv-labs/ego-oscar](https://github.com/fpv-labs/ego-oscar) 已发布 Apache-2.0 的 ESP32 固件、Radxa 录制器和诊断工具，官方[完整发布页](https://www.fpvlabs.ai/ego-oscar)直接列出源码与 DIY 指南。此前未列 GitHub 的判断仅适用于 2026-08-12 历史时点。

## 开源核查（步骤 2.5）

**2026-11-17 更新：** 完整采集栈现有[公开源码仓](https://github.com/fpv-labs/ego-oscar)，设备装配/CAD 见官方 DIY 指南；[Stereo-550](https://huggingface.co/datasets/fpvlabs/stereo-550) 仍为 gated + 定制数据许可。下表为原 2026-08-12 的历史快照，并非当前结论。


| 项 | 状态（2026-08-12） |
|----|-------------------|
| 3D / 产品叙事 | **已上线** Stereo Cap 可视化页 |
| 数据 | **Stereo-550** HF gated 发布 |
| 硬件文档 | Drive **Hardware Spec** PDF |
| 采集软件仓 | **未列** 可辨识 Ego-OSCAR 训练/录制入口 |
| 结论 | **部分开源（数据 + 硬件规格文档）**；完整软件/CAD 仓待跟进 |

## 对 wiki 的映射

- [论文归档](../papers/ego_oscar_arxiv_2608_08285.md)
- [数据集归档](../datasets/stereo-550.md)
- [公开源码归档](../repos/fpv-labs-ego-oscar.md)
- [wiki/entities/paper-ego-oscar.md](../../wiki/entities/paper-ego-oscar.md)
