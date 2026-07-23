# RigMo: Unifying Rig and Motion Learning for Generative Animation

> 来源归档（ingest）

- **标题：** RigMo: Unifying Rig and Motion Learning for Generative Animation
- **类型：** paper
- **机构：** Snap Inc. · UIUC · UC Santa Cruz · Carnegie Mellon University · Nanyang Technological University
- **原始链接：**
  - arXiv abs：<https://arxiv.org/abs/2601.06378>（v1，2026-01-10）
  - PDF：<https://arxiv.org/pdf/2601.06378>
  - 项目页（推荐，含 Code/Data）：<https://haoz19.github.io/RigMo-page/>（见 [`sources/sites/rigmo-page.md`](../sites/rigmo-page.md)）
  - 项目页镜像：<https://rigmo-page.github.io/>（截至入库日仍显示 Code/Data Coming Soon，以作者个人页为准）
  - 代码：<https://github.com/haoz19/RigMo>（见 [`sources/repos/rigmo.md`](../repos/rigmo.md)）
  - 数据集（gated）：<https://huggingface.co/datasets/haoz19/RigMo-data>
- **入库日期：** 2026-07-23
- **一句话说明：** **无标注骨架** 的统一 **rig + motion** 生成框架：双路径编码器从原始 mesh 序列发现 **Gaussian bones + skinning**，运动支路产出局部/根部 **SE(3)**；并在 RigMo 潜空间上接 **Motion-DiT** 做可控补全/生成。

## 核心论文摘录（MVP）

### 1) 动机：rig 与 motion 被拆成两条互不兼容的管线

- **链接：** <https://arxiv.org/abs/2601.06378>
- **摘录要点：** Auto-rigging 依赖艺术家骨架与蒙皮标注，难跨类扩展；人体/动物运动生成常假定 **预定义运动学树** 已给定；顶点空间 4D 生成（AnimateAnyMesh 等）又缺少可复用、可解释的 rig 资产。缺少「从原始 mesh 序列联合发现结构与动力学」的统一范式。
- **对 wiki 的映射：**
  - [RigMo](../../wiki/entities/rigmo.md) — 问题定位与范式对照
  - [Character Animation vs Robotics](../../wiki/concepts/character-animation-vs-robotics.md) — 图形学 rig 资产 vs 机器人控制栈边界

### 2) RigMo-VAE：双路径编码器 + Gaussian LBS

- **链接：** <https://arxiv.org/abs/2601.06378>
- **摘录要点：**
  - **Rigging branch**：以首帧几何为规范形，拓扑感知自注意力 + FPS 选 \(K\) bone token，解码 **Gaussian bone** \(G_k=[\Delta c_k,s_k,q_k]\) 与测地/高斯启发的 soft skinning。
  - **Motion branch**：帧间顶点位移经时空注意力，经 **Dynamic / Root VAE** 采局部与全局潜变量，解码为 per-bone / root 的 **SE(3)**（四元数 + 平移）。
  - **GaussianSkinningLBS** 可微重建 \(\hat V\)；可选 **temporal attention** 增强时序一致。
  - 相对 per-sequence SSDR 式优化：**feed-forward** 推断，跨运动可迁移。
- **对 wiki 的映射：**
  - [RigMo](../../wiki/entities/rigmo.md) — 架构、Mermaid 流程与评测表
  - [Diffusion-based Motion Generation](../../wiki/methods/diffusion-motion-generation.md) — 结构感知潜空间作为下游扩散条件

### 3) Motion-DiT：在 RigMo 运动潜空间上生成/补全

- **链接：** <https://arxiv.org/abs/2601.06378>
- **摘录要点：** 以静态 rig 特征为条件，condition encoder 产出 anchor/global token，**Diffusion Transformer** 在 motion latent 上做时空与帧条件 cross-attention；稀疏帧 mask 下可补全缺失运动并输出可动画网格。论文主贡献仍落在 VAE 的 rig–motion 分解；DiT 用于证明潜空间可用于下游生成。
- **对 wiki 的映射：**
  - [RigMo](../../wiki/entities/rigmo.md) — Motion-DiT 与开源边界（官方仓未含 DiT）
  - [ARDY](../../wiki/entities/ardy.md) — 同属生成式动画，但 ARDY 面向人体骨架+交互约束，RigMo 面向任意变形 mesh 的无标注 rig

### 4) 数据、训练与主结果

- **链接：** <https://arxiv.org/abs/2601.06378>
- **摘录要点：** 约 **2 万** 序列：DeformingThings4D、TrueBones、Objaverse-XL（质量过滤）；顶点统一到约 **5K**（FPS + 邻域）。VAE：24×A100，batch 144，\(T{=}20\)，\(K\in\{48,128\}\)，约 50K step。DT4D 上相对 per-case opt / UniRig+opt / MagicArticulate+opt，**重建 + 跨运动迁移** 均值 CD 显著更低（表 1）。推理约 **40 ms/帧**（A100，20 帧 5K 顶点）。
- **对 wiki 的映射：**
  - [RigMo](../../wiki/entities/rigmo.md) — 工程实践与复现入口

## 开源核查（步骤 2.5）

| 项 | 状态（截至 2026-07-23） |
|----|-------------------------|
| 作者项目页 `haoz19.github.io/RigMo-page` | **Code** → GitHub；**Data** → HF gated dataset |
| 镜像 `rigmo-page.github.io` | 仍写 Code/Data Coming Soon（过时） |
| 代码 | **部分开源**：[`haoz19/RigMo`](https://github.com/haoz19/RigMo) 含 **RigMo-VAE** 训练/验证/导出；README 明确 **Motion-DiT 未发布** |
| 数据 | HF `haoz19/RigMo-data`（~28 GB 压缩，需申请访问） |
| 许可 | 代码 **CC BY-NC 4.0**；数据集另见 HF card（衍生自 DT4D / Objaverse-XL / TrueBones） |
| 结论 | **部分开源** — VAE 可复现训练；生成阶段 DiT 与预训练权重边界以 README 为准 |

## 当前提炼状态

- [x] 摘要、双路径 VAE、Motion-DiT、数据与表 1 主结果已摘录
- [x] 项目页双 URL + 仓库/数据开源边界已核查并互链

## BibTeX

```bibtex
@article{zhang2026rigmo,
  title   = {RigMo: Unifying Rig and Motion Learning for Generative Animation},
  author  = {Zhang, Hao and Luo, Jiahao and Wan, Bohui and Zhao, Yizhou and Li, Zongrui
             and Vasilkovsky, Michael and Wang, Chaoyang and Wang, Jian and Ahuja, Narendra
             and Zhou, Bing},
  journal = {arXiv preprint arXiv:2601.06378},
  year    = {2026}
}
```
