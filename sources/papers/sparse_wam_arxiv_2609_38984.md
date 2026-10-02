# Sparse-WAM：动作引导的稀疏想象加速

- **标题：** Sparse-WAM: Accelerating World Action Models via Action-Guided Sparse Imagination
- **类型：** paper
- **论文：** <https://arxiv.org/abs/2609.38984>（2026-09-30，v1）；[PDF](https://arxiv.org/pdf/2609.38984)
- **机构：** 南京大学、香港科技大学、哈尔滨工业大学、洛桑联邦理工学院。
- **入库日期 / 最后更新：** 2026-10-02
- **一句话说明：** 联合视频–动作去噪时，只计算动作相关的未来 token；Pilot 降低打分和打包开销，无需额外训练加速模块。

## 核心摘录

### 1. 动作相关性指导计算预算（§3–4）

用动作 query 对未来帧 key 的注意力筛选逐帧核心区域，再保留跨帧共享位置的锚点；观测与动作 token 全部保留。锚点共享的是位置，各帧特征仍独立。

**对 wiki 的映射：** [Sparse-WAM](../../wiki/entities/paper-sparse-wam.md)、[WAM](../../wiki/concepts/world-action-models.md)。

### 2. Pilot 执行稀疏推理（§4.2）

默认每个 action chunk 的首个去噪步骤完整计算并建立选择，后续复用位置与打包信息。被省略区域仍由缓存的预测参与采样更新，动作预测每步重算；新 chunk 重新选取。

**对 wiki 的映射：** [Sparse-WAM](../../wiki/entities/paper-sparse-wam.md)。

### 3. 加速要对齐执行后端（Table 1/2/5、Appendix B.4）

RTX 4090：FastWAM-Joint / LIBERO 为 1.98×，成功率 98.75% → 98.45%；Cosmos 3 Edge / RoboLab-120 为 1.85×，22.90% → 23.00%。主表以 dense eager 为参照，包含执行优化收益。

同后端比较：Edge eager 为 1.55×；双方均启用 CUDA Graphs 与 torch.compile 时为 **1.56×**（727.44 ms → 464.95 ms / chunk）。计时包含筛选/缓存/采样，但不含 CPU 输出拷贝、视频解码、RPC 与仿真。training-free 指加速方法不新增训练；真机基座策略仍做过微调。

**对 wiki 的映射：** [Sparse-WAM](../../wiki/entities/paper-sparse-wam.md)、[WAM](../../wiki/concepts/world-action-models.md)。

## 开放状态

截至 2026-10-02，所给论文摘要页与 v1 全文未列官方项目页、GitHub、权重或数据发布入口；检索亦未核实官方仓库。状态记为**未核实实现开放**，不猜测 URL，不把其他 WAM 仓库当成 Sparse-WAM 源码。
