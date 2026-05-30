# SAM 3D Body（全身单图人体网格恢复）

> 来源归档（ingest）

- **标题：** SAM 3D Body: Robust Full-Body Human Mesh Recovery
- **类型：** paper
- **来源：** arXiv preprint
- **原始链接：**
  - <https://arxiv.org/abs/2602.15989>
  - <https://arxiv.org/pdf/2602.15989.pdf>
- **官方代码：** <https://github.com/facebookresearch/sam-3d-body>
- **作者：** Xitong Yang, Devansh Kukreja, Don Pinkus, Anushka Sagar, Taosha Fan, Jinhyung Park, Soyong Shin, Jinkun Cao, Jiawei Liu, Nicolas Ugrinovic, Matt Feiszli, Jitendra Malik, Piotr Dollar, Kris Kitani（Meta Superintelligence Labs）
- **入库日期：** 2026-05-30
- **一句话说明：** 可提示（2D 关键点 / mask）的单图全身 HMR 基础模型，基于新参数化 **MHR（Momentum Human Rig）** 解耦骨架与表面，在野外多样视角与遮挡下报告 SOTA 级 MPJPE/PVE，并发布 checkpoint、数据集与 Hugging Face 权重。

## 核心论文摘录（MVP）

### 1) 问题定位：SAM 3D 人体支路与 MHR 表示

- **链接：** <https://github.com/facebookresearch/sam-3d-body>（README / 论文摘要级）
- **摘录要点：** **SAM 3D Body（3DB）** 是 **SAM 3D** 双模型之一（另一支为 [SAM 3D Objects](https://github.com/facebookresearch/sam-3d-objects) 物体重建）。面向 **单张 RGB 图像** 的 **full-body human mesh recovery（HMR）**，估计 **身体 + 脚 + 手** 姿态；网格参数化采用 **[Momentum Human Rig（MHR）](https://github.com/facebookresearch/MHR)**——将 **骨骼结构** 与 **表面形状** 解耦，相对传统 SMPL 族在精度与可解释性上更易对齐下游 rig。
- **对 wiki 的映射：**
  - [SAM 3D Body](../../wiki/entities/sam-3d-body.md) — 定义、MHR 与 SAM 可提示范式。
  - [Motion Retargeting Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md) — 单目视频估计上游节点。

### 2) 架构与可提示推理

- **摘录要点：** **Encoder–decoder**；支持 **2D keypoints、masks** 等辅助提示，交互风格延续 **SAM 家族** 的 user-guided inference。训练数据经 **可微优化 + 多视角几何 + 稠密关键点 + data engine** 的多阶段标注管线覆盖常见与罕见姿态、宽视角。
- **对 wiki 的映射：**
  - [SAM 3D Body](../../wiki/entities/sam-3d-body.md) — 流程图与提示模式表。
  - `wiki/methods/wilor.md` — 手部专用估计与全身 MHR 的互补关系（交叉引用见实体页）。

### 3) 公开 checkpoint 与基准（README 表，2025-11-19 发布）

- **摘录要点：**
  - **DINOv3-H+（840M）**：3DPW MPJPE **54.8**、EMDB **61.7**、RICH PVE **60.3**、COCO PCK@.05 **86.5**、LSPET **68.0**、Freihand PA-MPJPE **5.5**。
  - **ViT-H（631M）**：同量级指标，部分野外集略优。
  - 权重与配置托管 **Hugging Face**（`facebook/sam-3d-body-dinov3` 等）；配套 **SAM 3D Body Dataset** 与 demo notebook。
- **对 wiki 的映射：**
  - [SAM 3D Body](../../wiki/entities/sam-3d-body.md) — 选型与评测锚点。
  - [SAM3DBody-cpp](../../wiki/entities/sam3dbody-cpp.md) — 工程侧 ONNX 导出与实时管线对照。

### 4) 与 SAM 3D Objects 及机器人下游

- **摘录要点：** 官方提供 **人体 mesh 与物体 mesh 对齐到同一坐标系** 的示例 notebook（`sam-3d-objects` 仓内 `demo_3db_mesh_alignment.ipynb`），便于 **人–物–场景** 联合理解。对机器人栈，单图/视频 **MHR 参数 + 70 关键点** 可作为 **重定向、遥操作标注、模仿学习参考** 的上游（需再经 Motion Retargeting 映射到机器人关节）。
- **对 wiki 的映射：**
  - [SAM 3D Body](../../wiki/entities/sam-3d-body.md) — 与物体支路、BVH/视频导出衔接。
  - [Whole-Body Tracking Pipeline](../../wiki/concepts/whole-body-tracking-pipeline.md) — 视频估计分支候选。

## 当前提炼状态

- [x] README / 项目页级架构、MHR、基准表与 Hugging Face 发布信息已摘录
- [x] 与 `sources/repos/sam-3d-body.md`、`sources/repos/sam3dbody-cpp.md` 分工明确
- [x] wiki 映射：`wiki/entities/sam-3d-body.md`、`wiki/entities/sam3dbody-cpp.md`
