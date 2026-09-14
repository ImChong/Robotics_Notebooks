# Axolotl3D: a Unified Framework for Faithful 3D Shape Completion

> 来源归档

- **标题：** Axolotl3D: a Unified Framework for Faithful 3D Shape Completion
- **类型：** paper
- **作者：** Anita Hu, Maria Shugrina
- **机构：** 英伟达（NVIDIA）Spatial Intelligence Lab（SIL）
- **链接：** https://arxiv.org/abs/2607.20660
- **arXiv：** 2607.20660
- **项目页：** https://research.nvidia.com/labs/sil/projects/axolotl3d/
- **会议：** ECCV 2026
- **年份：** 2026
- **入库日期：** 2026-09-14
- **一句话说明：** 多模态、遮挡感知的 3D 形状补全：联合图像、可见性 mask、相机参数与部分点云；在 Hunyuan3D-DiT 上微调多模态条件 token，ShapeVAE 解码；统一训练策略覆盖单视图、稀疏多视图与编辑场景。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-axolotl3d.md`](../../wiki/entities/paper-axolotl3d.md)

---

## 开源状态（步骤 2.5，截至 2026-09-14）

| 资源 | 状态 |
|------|------|
| 项目页 / 论文 | 公开摘要、视频、对比与编辑演示 |
| GitHub / Hugging Face 权重 | **待发布** — 项目页 HTML 含注释 `Code (Coming Soon)`，截至入库日无公开仓库链接 |
| 依赖骨干 | Hunyuan3D-DiT / ShapeVAE（腾讯混元 3D 生态，与本方法微调栈绑定） |

**结论：截至入库日代码未开源；复现须等待官方发布或自研对齐 Hunyuan3D 微调管线。**

## 核心摘录

1. **缺口：** 近期 3D 生成模型（扩散 + 大规模先验）多假设 **单视图、完全可见**；多视图、遮挡与 **局部编辑** 场景缺乏统一可控补全框架。
2. **条件：** 联合 **图像、可见性 mask、相机参数、部分点云** —— 点云作 **几何锚** 保真补全，相机参数保证多视图在共享 3D 坐标系对齐。
3. **架构：** 各模态编码后融合为 **multi-modal condition tokens**；在 **Hunyuan3D-DiT** 上微调产出 completed shape latents，**ShapeVAE** 解码为完整 mesh。
4. **训练：** **统一 on-the-fly 合成** —— 从大规模 3D mesh 数据模拟部分观测、遮挡与图文混合条件，覆盖单视图 / 稀疏多视图 / 编辑 regime。
5. **评测：** Toys4K、OmniObject3D，**干净与合成遮挡** 设定下 SOTA 级几何精度与重建保真。
6. **应用：** **形状编辑**（inpaint 单视图 + 条件点保真未编辑区）；**Image-to-3D**（结合 **Pi3X** 相机与点预测，鲁棒稀疏噪声点）；**物理仿真**（补全真实场景部分观测物体以支撑更完整仿真）。

## 对 wiki 的映射

| 摘录主题 | 目标 wiki |
|----------|-----------|
| SIL 研究组 | [`wiki/entities/nvidia-spatial-intelligence-lab.md`](../../wiki/entities/nvidia-spatial-intelligence-lab.md) |
| Hunyuan3D 资产生成对照 | [`wiki/concepts/text-to-cad.md`](../../wiki/concepts/text-to-cad.md)、[`wiki/entities/paper-milo.md`](../../wiki/entities/paper-milo.md) |
| Real2Sim 网格补全 | [`wiki/entities/paper-simfoundry-real2sim-scene-generation.md`](../../wiki/entities/paper-simfoundry-real2sim-scene-generation.md) |
| Pi3X 几何基础模型 | [`wiki/entities/paper-glob3r.md`](../../wiki/entities/paper-glob3r.md) |
| 点云补全与世界模型 | [`wiki/entities/paper-sa-2607-00148-3d-point-world-models-point-completion-enables-m.md`](../../wiki/entities/paper-sa-2607-00148-3d-point-world-models-point-completion-enables-m.md) |
