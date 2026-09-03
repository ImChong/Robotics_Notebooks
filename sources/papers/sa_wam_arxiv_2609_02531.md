# SA-WAM: Spatially Aware World Action Model via Geometric Latent Diffusion（arXiv:2609.02531）

> 来源归档（ingest）

- **标题：** Spatially Aware World Action Model via Geometric Latent Diffusion
- **简称：** SA-WAM
- **类型：** paper / world-action-model / 3d-geometry / diffusion-policy
- **arXiv：** <https://arxiv.org/abs/2609.02531>
- **PDF：** <https://arxiv.org/pdf/2609.02531>
- **项目页：** <https://jlopetegui98.github.io/projects/sa_wam.html> — 归档见 [`sources/sites/sa-wam.md`](../sites/sa-wam.md)
- **代码：** 截至 2026-09-03 **无** GitHub 链接
- **机构：** Inria、École normale supérieure、CNRS、PSL Research University
- **入库日期：** 2026-09-03
- **索引来源：** [具身智能小站 8 篇盘点](../blogs/wechat_embodied_station_8_papers_open_source_2026-09-03.md)
- **一句话说明：** 将预训练视频扩散模型改造为联合预测 action、RGB、depth 的 3D-aware WAM；对数尺度 depth 编码接入冻结 VAE tokenizer；RoboCasa / LIBERO-Plus SOTA，UR5 真机随机环境超强基线。

## 开源状态（步骤 2.5，2026-09-03）

| 组件 | 状态 |
|------|------|
| 项目页 | 已上线（方法、表格、视频） |
| GitHub / 权重 | **未见** 公开链接 |

**结论：待发布**

## 核心摘录

### 摘录 1：几何 latent 注入

- 单一 diffusion backbone 联合 action + RGB + depth。
- 非线性 depth 编码映射无界深度到冻结 VAE 有界输入域。

**对 wiki 的映射：** [paper-sa-wam](../../wiki/entities/paper-sa-wam.md)

### 摘录 2：RoboCasa（50 demo/task）

- 平均 **76.6%** vs Cosmos-Policy **67.1%**（训练数据 6–20× 更少）。

**对 wiki 的映射：** [paper-sa-wam](../../wiki/entities/paper-sa-wam.md)

### 摘录 3：真机 UR5

- Clean **90.0%** vs Cosmos-Policy **75.0%**；Randomized **77.5%** vs **48.8%**。

**对 wiki 的映射：** [paper-sa-wam](../../wiki/entities/paper-sa-wam.md)

## 当前提炼状态

- [x] 项目页核查
- [x] wiki 映射
