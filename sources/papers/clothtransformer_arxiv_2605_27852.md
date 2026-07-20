# ClothTransformer — Unified Latent-Space Transformers for Scalable Cloth Simulation

> 来源归档（ingest）

- **标题：** ClothTransformer: Unified Latent-Space Transformers for Scalable Cloth Simulation
- **类型：** paper
- **来源：** arXiv preprint
- **原始链接：**
  - <https://arxiv.org/abs/2605.27852>
  - 项目页：<https://yucrazing.github.io/clothtransformer/>
  - 代码：<https://github.com/YuCrazing/ClothTransformer>
  - 数据集：<https://huggingface.co/datasets/YuCrazing1/ClothTransformer-dataset>
- **机构：** S-Lab, Nanyang Technological University（南洋理工大学）；Feeling AI；University of Oxford（牛津大学）；Shanghai AI Laboratory（上海人工智能实验室）
- **作者：** Yu Zhang、Yidi Shao、Wenqi Ouyang、Yushi Lan、Zhexin Liang、Chengrui Wu、Xudong Xu、Xingang Pan
- **入库日期：** 2026-07-20
- **一句话说明：** 将布料仿真 reformulate 为 **潜空间自回归序列建模**：**统一 Transformer** 在 **人体着装 / 机器人抓取布料 / 自由落体碰撞** 三类场景下 **单模型联合训练**，以 **cross-attention 固定 latent token** 解耦网格分辨率，并用 **GIPC 生成的 ~493.4k 帧无穿透数据集** 支撑 **可微 CCD 损失 + 推理 CCD 后处理**；相对 HOOD/ContourCraft 系 GNN、MAT、LayersNet 等基线 **MVE 约低 4–9×**。

## 核心论文摘录（MVP）

### 1) 问题：神经布料仿真缺乏泛化、分辨率耦合与穿透

- **链接：** <https://arxiv.org/abs/2605.27852> §1
- **摘录要点：**
  - 现有学习式布料仿真多 **专精单一场景**（常见为人体着装），或需 **每场景单独模型**。
  - GNN 类方法 **推理成本随顶点/边数增长**，高密度网格与实时性冲突。
  - 训练多依赖 **离散碰撞检测（DCD）**，快运动下 **tunneling**；**连续碰撞检测（CCD）** 需 **无穿透高质量 GT**，公开数据集往往不满足。
- **对 wiki 的映射：**
  - [ClothTransformer](../../wiki/entities/paper-clothtransformer-unified-latent-cloth-simulation.md) — 三大局限与统一 latent Transformer 定位

### 2) 方法：Spatial Encoder → Temporal Transformer → Spatial Decoder

- **链接：** <https://arxiv.org/abs/2605.27852> §3；<https://yucrazing.github.io/clothtransformer/>
- **摘录要点：**
  - **自回归任务：** 给定当前布料状态 \((X_t, V_t)\)、rest shape 与 **lookahead 碰撞体网格** \(C_{t+1}\)，预测下一帧顶点 \(X_{t+1}\)。
  - **Spatial Encoder：** 2 层 GNN 提取 cloth 顶点 token + 碰撞三角形 token；**learnable query cross-attention** 压缩为 **固定 \(K=1024\) latent tokens**。
  - **Temporal Transformer：** 12 层、768 dim、block-causal masking，在 latent 空间演化动力学。
  - **Spatial Decoder：** rest-pose 顶点作 query cross-attend latent → GNN 细化 → 3D 坐标；**场景无关**（人体/夹爪/刚体均编码为通用三角形 token）。
- **对 wiki 的映射：**
  - [ClothTransformer](../../wiki/entities/paper-clothtransformer-unified-latent-cloth-simulation.md) — 架构与 Mermaid 管线

### 3) 可微 CCD 与两阶段训练目标

- **链接：** <https://arxiv.org/abs/2605.27852> §3.3–3.4
- **摘录要点：**
  - **CCD 模块：** detect-then-regress；训练期 **Self-VF / Self-EE** 可微 CCD loss；推理期五类 primitive 接触 **CCD 后处理**。
  - **Pretrain：** \(\mathcal{L}_{mse} + \mathcal{L}_{contact}\)（160k steps）；**Finetune：** 加 \(\mathcal{L}_{ccd}\)（40k steps）；rollout curriculum 1→5 步。
  - 总训练约 **300 NVIDIA H200 GPU·h**；默认 \(N_{latents}=1024\) 推理 **~4.9 ms/frame**（Human Garment 消融，RTX 4090 上跨分辨率实验另报）。
- **对 wiki 的映射：**
  - [ClothTransformer](../../wiki/entities/paper-clothtransformer-unified-latent-cloth-simulation.md) — CCD 与训练日程
  - [Contact Dynamics](../../wiki/concepts/contact-dynamics.md) — 碰撞/穿透语境交叉引用

### 4) 无穿透三场景数据集与实验

- **链接：** <https://arxiv.org/abs/2605.27852> §4–5
- **摘录要点：**
  - **GT 仿真器：** GIPC（IPC 系 GPU 无穿透求解器）；240 帧/序列，\(\Delta t=1/60\) s。
  - **三子集：** Human Garment（SMPL 着装 T 恤/裙，56 seq / 13.4k frames）；Robotic Manip.（1000 布料网格 + 夹爪抓取，240k frames）；Diverse Object Collision（Objaverse 刚体 + 自由落体，240k frames）；合计 **2056 序列 / 493.4k frames**。
  - **基线：** SOTA GNN（HOOD/ContourCraft 骨干）、MAT、LayersNet；**单模型** 在三场景 test 上 MVE **6.5–15 cm** 量级，基线 **31–149 cm**。
  - **可扩展性：** 在 ~3.6k 顶点训练、最高 **40k 顶点测试** 仍优于基线；核心 Transformer 成本 \(O(N_{latents}^2)\) 与网格规模解耦。
- **对 wiki 的映射：**
  - [ClothTransformer](../../wiki/entities/paper-clothtransformer-unified-latent-cloth-simulation.md) — 数据集表与定量结果
  - [Deform360](../../wiki/entities/paper-deform360-deformable-visuotactile-dataset.md) — 真实可变形体（布/绳）数据对照
  - [Manipulation](../../wiki/tasks/manipulation.md) — 机器人抓取布料场景交叉引用

### 5) 开源状态（截至 2026-07-20 项目页核查）

- **链接：** <https://github.com/YuCrazing/ClothTransformer>；<https://huggingface.co/datasets/YuCrazing1/ClothTransformer-dataset>
- **摘录要点：**
  - **数据集已发布：** Hugging Face `YuCrazing1/ClothTransformer-dataset`（README 标注 2026-07-17 上线）。
  - **代码仓存在但极简：** GitHub 仅 README + `.gitignore`，**未见训练/推理实现与预训练权重**（部分开源 / 待补全）。
  - 项目页以论文演示、架构图与 BibTeX 为主，**未单独列出权重下载**。
- **对 wiki 的映射：**
  - [ClothTransformer 项目页](../sites/yucrazing-clothtransformer-github-io.md)
  - [YuCrazing/ClothTransformer 仓库](../repos/YuCrazing-ClothTransformer.md)

## 引用（arXiv BibTeX）

```bibtex
@misc{zhang2026clothtransformerunifiedlatentspacetransformers,
      title={ClothTransformer: Unified Latent-Space Transformers for Scalable Cloth Simulation},
      author={Yu Zhang and Yidi Shao and Wenqi Ouyang and Yushi Lan and Zhexin Liang and Chengrui Wu and Xudong Xu and Xingang Pan},
      year={2026},
      eprint={2605.27852},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2605.27852},
}
```
