# LSRM: High-Fidelity Object-Centric Reconstruction via Scaled Context Windows

> 来源归档（ingest / 复核增强）

- **标题：** LSRM: High-Fidelity Object-Centric Reconstruction via Scaled Context Windows
- **类型：** paper / 3d-reconstruction / object-centric / sparse-attention / inverse-rendering
- **arXiv：** [2604.05182](https://arxiv.org/abs/2604.05182)
- **作者：** Zhengqin Li、Cheng Zhang、Jakob Engel、Zhao Dong
- **机构：** Meta Reality Labs Research（项目页 / arXiv / Springer ECCV 2026 条目一致）
- **出处：** ECCV 2026 Long Oral
- **项目页：** <https://lzqsd.github.io/LSRM.github.io/>
- **代码：** <https://github.com/facebookresearch/Large-Sparse-Reconstruction-Model>
- **权重：** <https://huggingface.co/facebook/Large-Sparse-Reconstruction-Model>
- **入库日期：** 2026-09-12
- **最后更新：** 2026-09-16
- **一句话说明：** 用 **原生稀疏注意力（NSA）** 把物体/图像 token 上下文扩到 prior SOTA 的 **20× / 2×+**，两阶段 coarse-to-fine 前馈重建高保真可重光照 3D 资产。

## 核心论文摘录

### 1) 问题与动机

- 物中心前馈重建已 robust，但在**细粒度纹理/外观**上仍落后于 dense-view 优化。
- **扩展 transformer 上下文窗口**（更多 active object + image tokens）可显著缩小差距，并支持高保真 **逆渲染**。

**对 wiki 的映射：** [`wiki/entities/paper-sa-2604-05182-lsrm.md`](../../wiki/entities/paper-sa-2604-05182-lsrm.md)

### 2) 方法要点（三大贡献）

1. **Coarse-to-fine pipeline：** Stage1 Dense Reconstruction Transformer 出粗低分辨率体积；Stage2 在活跃稀疏体素上预测高分辨率残差。
2. **3D-aware spatial routing：** 用显式几何距离建立 2D–3D 对应，而非纯 attention score 路由。
3. **Block-aware sequence parallelism：** All-gather-KV 协议平衡动态稀疏 workload 跨 GPU。

**稀疏注意力：** 采用 **NSA（Native Sparse Attention）**；Triton 实现针对 **NVIDIA GPU / Tensor Core** 优化（论文工程语境，非 NVIDIA 联合作者机构）。

**对 wiki 的映射：** 流程总览 + 源码运行时序图

### 3) 指标（论文 / 项目页）

| 维度 | 结果 |
|------|------|
| 上下文 | 物体 token **20×** prior SOTA；图像 token **>2×** |
| NVS | PSNR **+2.4 dB+**；LPIPS **−40%+** vs SOTA |
| 逆渲染 | ORB/DTC 等 benchmark 纹理/几何细节一致提升 |

### 4) 开源与复现（步骤 2.5）

| 类别 | 状态 | 说明 |
|------|------|------|
| 代码 | **已开源** | [facebookresearch/Large-Sparse-Reconstruction-Model](https://github.com/facebookresearch/Large-Sparse-Reconstruction-Model) |
| 权重 | **已发布** | HF `facebook/Large-Sparse-Reconstruction-Model`（rgb + brdf checkpoints） |
| 许可证 | CC BY-NC 4.0 | 非商业；见仓库 LICENSE.md |
| 依赖 | DINOv3 + Blender | README：`../dinov3` 权重 + `../blender` headless 渲染 |
| 硬件 | NVIDIA H200 验证 | 推理 **<40 GB** VRAM（README） |

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-sa-2604-05182-lsrm.md`](../../wiki/entities/paper-sa-2604-05182-lsrm.md)
- 项目页：[`sources/sites/lsrm-project.md`](../sites/lsrm-project.md)
- 仓库：[`sources/repos/facebookresearch-large-sparse-reconstruction-model.md`](../repos/facebookresearch-large-sparse-reconstruction-model.md)
