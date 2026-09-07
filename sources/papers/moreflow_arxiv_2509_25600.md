# moreflow_arxiv_2509_25600

> 来源归档（ingest）

- **标题：** MoReFlow: Motion Retargeting Learning through Unsupervised Flow Matching
- **类型：** paper
- **来源：** arXiv / 项目页
- **原始链接：**
  - <https://arxiv.org/abs/2509.25600>
  - <https://dnjsxor999.github.io/projects/MoReFlow/MoReFlow.html>
- **作者：** Wontaek Kim, Tianyu Li*, Sehoon Ha*（Georgia Institute of Technology；* co-advised）
- **入库日期：** 2026-09-07
- **一句话说明：** 无配对数据的跨角色运动重定向：各角色先训 VQ-VAE motion tokenizer，再用 conditional flow matching 对齐 codebook 潜空间，支持可逆重定向与局部/世界系条件。

## 核心摘录

### 1) 两阶段框架
- **Stage 1 — VQ-VAE tokenizer：** 每角色独立 encoder/decoder + codebook；SMPL 人形与 Booster T1 用 512×512 codebook，Spot 四足用 256×256；32 帧窗口 temporal downsample ×4 → 8 tokens。
- **Stage 2 — Flow matching：** Multi-Sample Condition Coupling 在训练时构造伪配对；Discrete-Flow-Transformer 将源 codebook 分布映射到目标；推理时 ODE 积分后由目标 decoder 重建。

### 2) 相对基线
- 相对 handcrafted IK 约束：更灵活的任务条件（style / world-frame alignment）。
- 相对 GAN 式无监督：flow matching 对齐更稳定；实验覆盖 locomotion、arm motion、可控与 chain-reversal。

### 3) 开源核查（步骤 2.5）
- **项目页（2026-09-07）：** 提供摘要、框架图、demo 视频与 BibTeX；**未列 GitHub / Hugging Face** → 截至入库日 **代码待发布**。

## 对 wiki 的映射

- 新建 [MoReFlow 论文实体](../../wiki/entities/paper-moreflow-motion-retargeting-flow.md)
- 交叉 [Motion Retargeting](../../wiki/concepts/motion-retargeting.md)、[GMR](../../wiki/methods/motion-retargeting-gmr.md)、[NMR](../../wiki/methods/neural-motion-retargeting-nmr.md)、[AdaMorph](../../wiki/entities/paper-adamorph-unified-motion-retargeting.md)

## 当前提炼状态

- [x] arXiv + 项目页核查
- [x] 开源状态写入 wiki
