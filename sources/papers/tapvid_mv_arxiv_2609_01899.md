# TAPVid-MV: A Benchmark for Tracking Any Point in 3D Across Multiple Views（arXiv:2609.01899）

> 来源归档（ingest）

- **标题：** TAPVid-MV: A Benchmark for Tracking Any Point in 3D Across Multiple Views
- **简称：** TAPVid-MV
- **类型：** paper / benchmark / multi-view-tracking / 3d-point-tracking
- **arXiv：** <https://arxiv.org/abs/2609.01899>
- **PDF：** <https://arxiv.org/pdf/2609.01899>
- **项目页：** <https://tapvidmv.github.io/> — 归档见 [`sources/sites/tapvid-mv.md`](../sites/tapvid-mv.md)
- **数据查看器：** <https://tapvidmv.github.io/dataviewer>
- **代码：** 基准数据、评测协议、Perpetua 生成器经项目页发布；**无** 独立 GitHub 仓库（截至 2026-09-03）
- **机构：** Google DeepMind、ETH 等（多机构）
- **入库日期：** 2026-09-03
- **索引来源：** [具身智能小站 8 篇盘点](../blogs/wechat_embodied_station_8_papers_open_source_2026-09-03.md)
- **一句话说明：** 首个多同步移动相机长时 3D 任意点跟踪基准；284 序列、1,142 相机流、109,769 轨迹；30+ baseline 均未接近解决。

## 开源状态（步骤 2.5，2026-09-03）

| 组件 | 状态 |
|------|------|
| 基准标注 / Rerun 可视化 | **已发布**（项目页 dataviewer） |
| Perpetua 生成器 + DROID 训练轨迹 | **已发布**（论文声明 release artifacts） |
| 统一 GitHub 训练代码 | **未见** |

**结论：部分开源** — 基准与评测资源可获取；完整训练栈无单一官方仓库。

## 核心摘录

### 摘录 1：规模

- 7 子集：机器人、人类活动、驾驶、合成等。
- 辅助模态：深度、LiDAR、SLAM/SfM、人体/物体 mesh、仿真。

**对 wiki 的映射：** [paper-tapvid-mv](../../wiki/entities/paper-tapvid-mv.md)

### 摘录 2：诊断结论

- 多视角点跟踪器未稳定优于单目。
- geometry recovery 是准确 3D 点跟踪主要瓶颈。

**对 wiki 的映射：** [paper-tapvid-mv](../../wiki/entities/paper-tapvid-mv.md)

## 当前提炼状态

- [x] 项目页核查
- [x] wiki 映射
