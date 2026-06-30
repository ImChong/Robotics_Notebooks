# OpenCap Monocular 项目页（utahmobl.github.io）

> 来源归档（ingest 配套站点）

- **URL：** <https://utahmobl.github.io/OpenCap-monocular-project-page/>
- **对应论文：** [OpenCap Monocular](https://arxiv.org/abs/2603.24733)（arXiv:2603.24733，2026 preprint）
- **机构：** 犹他大学运动生物工程实验室（Movement Bioengineering Lab, MoBL）
- **入库日期：** 2026-06-30
- **一句话说明：** 官方落地页：单手机 demo 视频、与 Vicon 金标准对比可视化、采集最佳实践与 OpenCap Visualizer 交互入口。

## 页面要点（2026-06 快照）

### TL;DR

1. **单台静态 iPhone** 视频 → **3D 运动学 + 肌肉骨骼动力学**。
2. 相对 marker mocap 验证：行走、深蹲、坐站；**免费开源**，经 <https://opencap.ai> 部署。
3. 交互式 3D 可视化：原视频 vs Motion Capture vs OpenCap Monocular 并排对比。

### 采集最佳实践（项目页）

| 指南 | 说明 |
|------|------|
| 相机位姿 | 被试前方 **45°**；身体长时间不可见则跟踪失败 |
| 着装 | 避免宽松/多层衣物 |
| 光照 | 正常室内光；避免强阴影或过暗 |
| 场景 | 前景避免多人；背景远处可有人 |
| 距离 | 全程全身入画；被试距相机 **<5 m** |
| 已验证活动 | 坐站、行走、深蹲；**跳跃当前不支持** |

### 产品入口

- 数据采集与云端处理：<https://opencap.ai>
- 交互可视化器：<https://visualizer.opencap.ai>（`mono.json` 格式）
- 源码：<https://github.com/utahmobl/opencap-monocular>

## 对 wiki 的映射

- 与 [sources/papers/opencap_monocular_arxiv_2603_24733.md](../papers/opencap_monocular_arxiv_2603_24733.md)、[sources/repos/opencap-monocular.md](../repos/opencap-monocular.md) 配对
- 实体页：[wiki/entities/paper-opencap-monocular.md](../../wiki/entities/paper-opencap-monocular.md)
