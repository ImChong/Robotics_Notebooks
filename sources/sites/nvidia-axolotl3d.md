# Axolotl3D（NVIDIA SIL 项目页）

> 来源归档（site）

- **标题：** Axolotl3D: a Unified Framework for Faithful 3D Shape Completion
- **类型：** site
- **作者：** Anita Hu, Maria Shugrina
- **机构：** NVIDIA Spatial Intelligence Lab（SIL）
- **链接：** https://research.nvidia.com/labs/sil/projects/axolotl3d/
- **论文：** https://arxiv.org/abs/2607.20660
- **会议：** ECCV 2026
- **入库日期：** 2026-09-14
- **一句话说明：** 多模态遮挡感知 3D 形状补全统一框架；演示含 Kaolin Web UI、形状编辑、splat 分割到 mesh 与真实场景物理仿真。
- **代码：** **待发布**（页面注释 `Code (Coming Soon)`，截至 2026-09-14 无 GitHub 链）
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-axolotl3d.md`](../../wiki/entities/paper-axolotl3d.md)

---

## 页面要点（2026-09-14 抓取）

- **TL;DR：** 从部分多模态观测做几何保真的 3D 形状生成、补全与编辑。
- **方法图：** posed images + visibility masks + partial points → 多模态 token → Hunyuan3D-DiT 微调 → ShapeVAE 解码。
- **对比：** Toys4K / OmniObject3D，单视图与多视图、有无合成遮挡；几何精度与重建保真优于 SOTA。
- **难例：** 高遮挡（自行车、椅子、马、机器人）利用几何相似未遮挡区域。
- **应用轴：** 形状编辑（inpaint + 条件点）；Image-to-3D（Pi3X 相机/点 → Axolotl3D）；物理仿真（补全捕获场景物体）。
