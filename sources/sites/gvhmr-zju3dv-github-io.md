# GVHMR 项目页（zju3dv.github.io/gvhmr）

> 来源归档（ingest 配套站点）

- **URL：** <https://zju3dv.github.io/gvhmr/>
- **对应论文：** [World-Grounded Human Motion Recovery via Gravity-View Coordinates](https://arxiv.org/abs/2409.06662)（SIGGRAPH Asia 2024；GitHub README 标注 **TPAMI 2026**）
- **机构：** 浙江大学 CAD&CG 等
- **入库日期：** 2026-06-21
- **一句话说明：** 官方落地页：Gravity-View 坐标动机、预处理→Transformer→世界轨迹管线图、训练/评测指标表、应用案例与 BibTeX。

## 页面要点（2026-06 快照）

### TL;DR

1. 聚焦 **单目视频 → world-grounded 全局人体运动**。
2. 核心思想：在 **Gravity-View（GV）坐标系** 中逐帧预测人体姿态；GV 由 **世界重力方向 + 相机视线方向** 唯一确定，天然重力对齐，减轻长序列全局运动学习的坐标歧义。
3. 相对 **自回归逐帧相对运动** 方法，**逐帧估计** 避免误差沿重力方向累积。
4. 在 AMASS、BEDLAM、H36M、3DPW 混合训练；**代码与权重公开**。

### 方法管线（项目页 Fig.）

**预处理：** 人体 bbox 跟踪 → 2D 关键点 → 图像特征 → **相对相机旋转**（视觉里程计 VO 或陀螺仪）。

**主干：** 多模态特征融合为 per-frame token → **相对 Transformer** + 多任务 MLP。

**中间表征：**
- GV 系人体朝向；
- SMPL 系根速度；
- 预定义关节的 **静止概率**（stationary probability）。

**输出：**
- 相机系 SMPL 参数；
- 经相机运动将中间表征变换到 **世界坐标** 得到 **全局轨迹**。

### 训练与速度（项目页）

| 项 | 数值 |
|----|------|
| 训练数据 | AMASS + BEDLAM + H36M + 3DPW 混合 |
| 训练配置 | 从零训练，420 epoch，batch 256，2×RTX 4090，约 **13 h** |
| 推理速度 | 排除预处理，RTX 4090 上 **1430 帧（~45 s）视频约 280 ms** |

### 评测

- **World-grounded metrics** 与 **Camera-space metrics** 分列展示（项目页表格）。
- 在野外基准上相对 SOTA 同时改善 **精度与速度**（摘要叙述）。

### GV 坐标动机（三点）

1. 天然考虑重力；
2. 每帧图像 **唯一** 定义 GV 系；
3. 避免连续帧沿重力方向的 **误差累积**。

## 对 wiki 的映射

- 与 [sources/papers/gvhmr_arxiv_2409_06662.md](../papers/gvhmr_arxiv_2409_06662.md)、[sources/repos/gvhmr.md](../repos/gvhmr.md) 配对
- 实体页：[wiki/entities/gvhmr.md](../../wiki/entities/gvhmr.md)
