# ARROW: Arbitrary Reconstruction and Tracking of 4D Observations in the Wild

> 来源归档（ingest · arXiv 论文）

- **标题：** ARROW: Arbitrary Reconstruction and Tracking of 4D Observations in the Wild
- **类型：** paper / 4d-reconstruction / point-tracking / multi-view / feedforward
- **arXiv：** <https://arxiv.org/abs/2610.01314>（2026-10-01）
- **作者：** Ilya Fradlin、Christian Schmidt、Jens Piekenbrinck、Karim Knaebel、Gonzalo Martin Garcia、Bastian Leibe
- **机构：** 亚琛工业大学（RWTH Aachen University）
- **项目页：** <https://ifradlin.github.io/arrow/>
- **官方代码：** <https://github.com/ifradlin/arrow>
- **权重：** <https://huggingface.co/ifradlin/arrow>
- **开源状态：** 已发布推理代码、预训练权重与交互式查看器；README 更新记为 2026-10-12
- **入库日期：** 2026-10-26
- **一句话说明：** ARROW 把输入顺序不固定的图像集、视频和多视角流编码为共享场景上下文，再以内容派生的观测身份进行独立查询，统一 3D 重建与点跟踪。

## 摘录 1：问题与核心贡献

- **问题：** 动态场景可能来自移动相机、多路视频或不同时刻拍摄的照片。它们没有统一的时间顺序，且视角、拍摄时间与可见性变化使跨图像对应困难。
- **核心方法：** 采用 order-invariant querying。多视图编码器生成共享图像上下文和每个观测的内容派生身份 token；查询按源像素、目标观测与参考观测指定几何对应。
- **与 D4RT 的关系：** ARROW 借鉴 D4RT 的查询接口，但为任意图像集合改变观测身份编码，移除依赖固定序列位置的时间编码。

**对 wiki 的映射：** [ARROW 实体页](../../wiki/entities/paper-arrow-4d-reconstruction.md)的方法与定位。

## 摘录 2：推理接口与实现

- **输入：** 无序图像集合、单段或多段视频、多视角流，或上述观测的组合。
- **查询：** 指定源图像上的像素、目标观测和参考坐标系；独立查询共享编码后的上下文。
- **输出：** 3D 点轨迹、稠密重建、深度与相机位姿等几何结果。
- **运行入口：** `arrow infer` 命令行 / Python API；`arrow-viewer` 可在浏览器中选择图像或视频直接运行推理，也可加载保存的预测结果。

**对 wiki 的映射：** 实体页「流程总览」与「工程实践」。

## 摘录 3：评测摘要

- 论文报告 ARROW 在 **WorldTrack** 与 **TAPVid-3D** 的 3D tracking 上达到新的最佳结果。
- 在适配为 RGB-only 输入的 **MVTracker** 多视角跟踪基准上，优于专用多视角跟踪器。
- 在 3D reconstruction 与 camera pose estimation 上保持竞争力。
- 以上为论文报告；与其他工作的结果比较时应核对数据划分、尺度对齐与输入设置。

**对 wiki 的映射：** 实体页「评测要点」与「结论」。

## 摘录 4：代码与权重边界

| 资产 | 入口 / 许可 |
|------|-------------|
| 推理代码与交互查看器 | [GitHub](https://github.com/ifradlin/arrow)；ARROW 代码和随附查看器按 MIT 条款，部分改编模型源码保留 Apache-2.0 |
| 预训练权重 | [Hugging Face](https://huggingface.co/ifradlin/arrow)；CC BY-NC 4.0，并受 Waymo Dataset License Agreement 非商业条款约束 |
| 训练数据 | 多个公开数据集；数据集本身不随仓库重新许可 |

**对 wiki 的映射：** [官方仓库归档](../repos/arrow.md)与实体页「局限与风险」。

## 对 wiki 的映射（汇总）

- [ARROW 实体页](../../wiki/entities/paper-arrow-4d-reconstruction.md)
- [D4RT](../../wiki/entities/paper-d4rt.md) — 查询式动态 4D 重建的相关工作
- [Track4World](../../wiki/entities/paper-track4world.md) — 前馈稠密 3D 跟踪对照
- [State Estimation](../../wiki/concepts/state-estimation.md) — 视觉几何作为状态估计上游

## 当前提炼状态

- [x] 核对 arXiv 摘要、项目页与官方仓库
- [x] 核对代码、权重与查看器发布状态
- [x] 区分代码与权重的许可范围
- [x] 升格为统一项目实体页
