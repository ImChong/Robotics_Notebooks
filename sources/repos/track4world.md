# Track4World — 原始资料归档

- **来源**：https://github.com/TencentARC/Track4World
- **类型**：repo
- **机构**：腾讯 ARC Lab（TencentARC）；香港科技大学（HKUST）
- **归档日期**：2026-09-15
- **项目页**：https://jiah-cloud.github.io/Track4World.github.io/
- **论文**：arXiv:2603.02573
- **权重**：https://huggingface.co/TencentARC/Track4World
- **许可**：代码仓库 MIT 类开源栈；HF 模型 **Tencent Green License**
- **开源结论（步骤 2.5）**：**已开源** — `demo.py` 推理、`evaluation/` 基准脚本、可视化工具与三份 checkpoint；需按 README 安装 Pi3 / Grounded-SAM-2 等子模块
- **快照**：约 317 stars；默认分支 `master`；CUDA 12.1 + Python 3.11

## 一句话说明

**Track4World** 官方实现：VGGT 风格 ViT 编码全局 3D 场景表示，经 **sparse-to-dense scene flow decoder** 与 **2D-to-3D correlation** 估计任意帧对的联合 2D/3D 稠密流，融合为世界系 **全像素 3D 跟踪**。

## 为什么值得保留

- **前馈稠密 3D 跟踪** 可直接服务 **4D 重建、动态场景理解、操作视频几何** 等机器人上游模块
- README 提供 **WorldTrack × OpenD4RT** 公平对比脚本（同图像信息量、仅换 predictor）
- 三骨干（DA3 / Pi3 / MoGe）便于按 **度量尺度 / 速度 / 精度** 选型

## 工程入口

| 入口 | 命令 / 路径 |
|------|-------------|
| 环境 | `conda create -n track4world python=3.11`；`pip install -r requirements.txt` |
| 权重 | `checkpoints/track4world_da3.pth` 等（HF 或 wget） |
| 稠密 3D 跟踪 | `python demo.py --mode 3d_efep --coordinate world_depthanythingv3 --ckpt_init checkpoints/track4world_da3.pth` |
| 2D 跟踪 | `python demo.py --mode 2d` |
| 评测 | `evaluation/eval.md`；`evaluation/opend4rt_comparison/` |
| 可视化 | `visualization/vis_3d_efep.py` 等 |

`--metric_scale` 目前仅 **DA3** 骨干支持米制输出；Pi3/MoGe 为相对尺度。

## 对 wiki 的映射

1. **[paper-track4world](../../wiki/entities/paper-track4world.md)** — 论文实体主入口
2. **[D4RT](../../wiki/entities/paper-d4rt.md)** — 动态 4D 查询范式对照（未开源）
3. **[state-estimation](../../wiki/concepts/state-estimation.md)** — 状态估计/几何 hub

## 关联原始资料

- [Track4World 项目页](../sites/track4world-project.md)
- [Track4World 论文摘录](../papers/track4world_arxiv_2603_02573.md)
