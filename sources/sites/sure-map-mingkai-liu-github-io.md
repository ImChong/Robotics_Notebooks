# SURE-Map（mingkai-liu.github.io）

> 来源归档（ingest）

- **标题：** SURE-Map: Self-Correcting Streaming Geometric Foundation Models
- **类型：** project site
- **URL：** <https://mingkai-liu.github.io/projects/sure-map/>
- **论文：** <https://arxiv.org/abs/2609.15795>
- **代码：** <https://github.com/RCL-Robotics/SURE-map>
- **权重：** <https://huggingface.co/milchstrasse/SURE-Map>
- **入库日期：** 2026-09-24
- **一句话说明：** MBZUAI / 北大 / 清华联合工作的官方项目页：摘要、方法两模块（跨视几何不确定性、多时间尺度自校正）与长程轨迹/点云过滤可视化。

## 开源核查（2026-09-24）

| 资源 | URL | 状态 |
|------|-----|------|
| 代码 | [RCL-Robotics/SURE-map](https://github.com/RCL-Robotics/SURE-map) | **已开源**（Apache-2.0；`training/dust3r/` 等保留 CC BY-NC-SA 4.0 片段） |
| Uncertainty 权重 | [milchstrasse/SURE-Map](https://huggingface.co/milchstrasse/SURE-Map) | **已发布**（`uncertainty.pt`） |
| 骨干（论文实验） | [robbyant/lingbot-map](https://huggingface.co/robbyant/lingbot-map) | **第三方已发布**（README 要求 `lingbot.pt`） |
| 训练 | `training/tests/train_flow_sigma_tartanair.py` | **已发布**（TartanAir v1；8 GPU） |
| 评测 | `benchmark/`、`online/` | **已发布**（KITTI / Oxford / VBR / 7-Scenes / NRGBD / DTU 等） |

## 对 wiki 的映射

- [paper-sure-map](../../wiki/entities/paper-sure-map.md) — 论文实体页
