# NVIDIA DVL — VGG-T³ 项目页

> 来源归档（ingest · 步骤 2.5）

- **标题：** VGG-T³ — Offline Feed-Forward 3D Reconstruction at Scale
- **类型：** site（NVIDIA DVL 官方项目页）
- **发布方：** NVIDIA Deep Vision Learning（DVL）
- **原始链接：** <https://research.nvidia.com/labs/dvl/projects/vgg-ttt/>
- **论文：** <https://arxiv.org/abs/2602.23361>
- **代码：** <https://github.com/nv-dvl/vgg-ttt>
- **权重：** <https://huggingface.co/nvidia/vgg-ttt>
- **入库日期：** 2026-09-09
- **一句话说明：** 项目页提供方法动画、1k 图 **VGGT / TTT3R / VGG-T³** 交互对比、**冻结场景表示后的视觉定位** 演示，以及 TTT 长度泛化（1 步 vs 2 步 optimizer）说明。

## 步骤 2.5 开源核查（2026-09-09）

| 项 | 项目页 / 关联链接 |
|----|-------------------|
| **代码** | **已开源** — 页眉链 [GitHub nv-dvl/vgg-ttt](https://github.com/nv-dvl/vgg-ttt) |
| **权重** | **已发布** — [Hugging Face nvidia/vgg-ttt](https://huggingface.co/nvidia/vgg-ttt) |
| **论文** | arXiv [2602.23361](https://arxiv.org/abs/2602.23361) |
| **训练数据/脚本** | README 写明训练 harness 已放、**数据集实现缺失**（非项目页隐瞒） |
| **许可** | NVIDIA OneWay **Noncommercial**（非商业科研） |

## 主页摘录

### Abstract 要点

- 离线前馈方法对输入图像数 **二次** 扩展；根因是场景几何的 **变长 KV 表示**。
- 用 **TTT** 蒸馏为 **固定尺寸 MLP** → **线性** 扩展；1k 图 **数秒级** 重建。
- 保留全局聚合 → 点图误差 **可比 VGGT**，并优于其他线性时间方法。
- 支持用 **未见查询图** 对已有场景表示做 **视觉定位**。

### 1k 图定性对比（Wayspots 类序列）

| 方法 | 耗时（页内标注） | 观感 |
|------|------------------|------|
| VGGT | ~11 分钟 | 质量略高 |
| TTT3R | ~61 秒 | **场景重建不完整** |
| VGG-T³ | ~58 秒 | **完整重建**，显著快于 VGGT |

### 视觉定位机制

- 批处理场景后 **冻结** TTT 优化权重 $\theta$。
- 新查询图：全局 attention 层 **仅对 $q_i$ 应用冻结 MLP** 读取场景，**不更新** $\theta$。
- 演示：**~10 FPS** 对未见查询图做定位（页内 Wayspots Map 等场景）。

### 长度泛化

- 训练长度：**1** optimizer step 足够。
- **1k 图 OOD 长度：** 增至 **2 steps**（相对训练时 1 step）即可接近完美泛化。

## 对 wiki 的映射

- 论文实体：[`wiki/entities/paper-vgg-ttt.md`](../../wiki/entities/paper-vgg-ttt.md)
- 代码归档：[`sources/repos/vgg_ttt.md`](../repos/vgg_ttt.md)
- 论文摘录：[`sources/papers/vgg_ttt_arxiv_2602_23361.md`](../papers/vgg_ttt_arxiv_2602_23361.md)
