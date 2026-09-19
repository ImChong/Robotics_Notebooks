# Seen2Scene: Completing Realistic 3D Scenes with Visibility-Guided Flow

> 来源归档（ingest）

- **标题：** Seen2Scene: Completing Realistic 3D Scenes with Visibility-Guided Flow
- **类型：** paper
- **arXiv：** 2603.28548
- **出处：** ECCV 2026
- **项目页：** <https://quan-meng.github.io/projects/seen2scene/>
- **论文：** <https://arxiv.org/abs/2603.28548>
- **Hugging Face Paper：** <https://huggingface.co/papers/2603.28548>
- **代码：** <https://github.com/quan-meng/seen2scene>
- **入库日期：** 2026-09-19
- **一句话说明：** 首个直接在**不完整真实 3D 扫描**上训练的 flow matching 场景补全/生成方法；用 **visibility-guided flow matching** 在稀疏 TSDF 上 mask 未知区域，支持 layout box / 文本 / 部分扫描多条件输入。

## 核心论文摘录

### 1) 问题与动机

- **痛点：** 既有 3D 场景生成/补全多依赖**完整合成 3D** 训练，与真实扫描的**部分可见、未知区域**分布不匹配。
- **核心主张：** 在真实不完整扫描上直接学习，用**可见性引导**显式 mask 相机未观测到的 TSDF 区域，使 flow matching 只在已知/可推断 token 上有效训练。

### 2) 方法四段（论文 Fig. 概览）

| 模块 | 作用 |
|------|------|
| **(a) Masked sparse VAE** | 部分扫描 TSDF patch → latent \(z\)；未知体素不参与编码 |
| **(b) Sparse transformer \(\mathcal{G}_\psi\)** | 以 3D layout boxes \(\mathcal{B}\) 为条件，在 surface/empty token 上做 **masked flow matching** |
| **(c) ControlNet 微调** | 注入部分扫描 \(v_p\)，专用于 **scan completion** |
| **(d) 多条件生成** | 同一生成器可适配 **text** 或 **layout** 条件，从零生成场景 |

**表示：** 稀疏网格上的 **TSDF**；骨干为 **sparse transformer** + **fVDB / TorchSparse** 稀疏算子。

### 3) 训练数据与基准

- **数据：** 3D-FRONT、ScanNet++、ARKitScenes（经 VDBFusion 融合为 TSDF/VDB；渲染/导出管线见官方仓）。
- **对比基线（文内）：** **SG-NN**、**NKSR** 于 ScanNet++ / ARKitScenes 补全质量。
- **任务：** (1) 部分扫描补全；(2) layout 条件 patch 生成；(3) text→layout（LLM）→3D 生成。

### 4) 开源与复现（步骤 2.5 核查，2026-09-19）

| 资产 | URL | 状态 |
|------|-----|------|
| 代码 | <https://github.com/quan-meng/seen2scene> | **已开源**（MIT；含 train/infer 入口 `python -m seen2scene.main`） |
| 预训练权重 | <https://huggingface.co/MQ66/seen2scene> | **已发布** |
| 样本数据 | <https://huggingface.co/datasets/MQ66/seen2scene-FRONT-3D> | **已发布**（1000 场景子集，可直接跑 inference） |
| VDBFusion fork | <https://github.com/quan-meng/vdbfusion> | 数据融合依赖 |

**训练链：** VAE → Flow Matching generator → ControlNet（completion）；checkpoint 层级 `experiments/auto_encoder/AE_LOG/generator/GEN_LOG/control/CONTROL_LOG`。

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-seen2scene.md`](../../wiki/entities/paper-seen2scene.md)
- 项目页：[`sources/sites/seen2scene-project.md`](../sites/seen2scene-project.md)
- 仓库：[`sources/repos/seen2scene.md`](../repos/seen2scene.md)
- 方法交叉：[generative-world-models.md](../../wiki/methods/generative-world-models.md)、[embodied-perception-six-spatial-representations.md](../../wiki/concepts/embodied-perception-six-spatial-representations.md)
