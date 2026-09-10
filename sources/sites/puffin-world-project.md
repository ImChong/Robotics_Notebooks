# Puffin-World 项目页

> 来源归档

- **标题：** Puffin-World — Scaling with Native 3D World States
- **类型：** 项目页（`*.github.io`）
- **链接：** https://kangliao929.github.io/projects/puffin-world/
- **论文：** arXiv:2609.04196
- **机构：** NTU S-Lab、University of Michigan、BJTU、ACE Robotics
- **入库日期：** 2026-09-10
- **一句话说明：** 官方展示页：三类原生 3D 世界状态（physics / geometry / appearance）、Omni-Camera、交互式 3D 重建 demo、相机理解基准表与 Puffin-16M 数据发布入口。
- **代码：** https://github.com/KangLiao929/Puffin（**已开源**）
- **模型：** https://huggingface.co/KangLiao/Puffin-World
- **数据：** https://kangliao929.github.io/projects/puffin-16m/ · https://huggingface.co/datasets/KangLiao/Puffin-16M
- **沉淀到 wiki：** [paper-puffin-world](../../wiki/entities/paper-puffin-world.md)
- **交叉归档：** [puffin_world_arxiv_2609_04196.md](../papers/puffin_world_arxiv_2609_04196.md)、[puffin.md](../repos/puffin.md)

---

## 页面摘要（2026-09）

**定位：** 单一统一多模态模型连接 **物理世界感知**、**自由视点空间仿真** 与 **3D 世界建模/重建**。

### 三类原生状态

| 状态 | 内容 | 能力 |
|------|------|------|
| Physics | 重力场、纬度 | 相机到世界理解、物理一致轨迹传播 |
| Geometry | 深度 | 稠密空间结构、原生 3D 重建 |
| Appearance | RGB 轨迹 | 高保真视觉，跨相机运动一致 |

### 架构（页面图示）

几何对齐视觉编码器 + LLM + 扩散模型 + 轻量 connector；理解、生成、重建不依赖任务专用外部几何模块。

### 代表性指标（项目页表格摘录）

- 相机理解：四基准 **12/12** 最佳 median roll/pitch/vFoV error；LaMAR roll **0.26°**。
- Puffin-Cam-Bench：median up-vector **0.84°**、latitude **1.26°**、gravity **0.79°**；FID 最低。
- RealEstate10K：PSNR **17.22**、LPIPS **0.318**（Table 5 第一）。
- Puffin-16M：**15M** VL-camera 三元组 + **1M** 轨迹；**28** 公开数据集 **~44.5M** 图像相机标注。

### 闭环应用（Fig. 7）

- **Mimic world exploration：** 固定轨迹下扩展不同 3D 世界。
- **Self-calibrated world exploration：** 从重力错位观测推理并预测校正相机动作。

---

## 对 wiki 的映射

- [paper-puffin-world](../../wiki/entities/paper-puffin-world.md)
- [puffin.md](../repos/puffin.md)
