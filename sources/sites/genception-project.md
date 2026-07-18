# GenCeption 项目页（genception.github.io）

> 来源归档（ingest 配套站点）

- **URL：** <https://genception.github.io/>
- **对应论文：** [Video Generation Models are General-Purpose Vision Learners](https://arxiv.org/abs/2607.09024)（arXiv:2607.09024，ECCV 2026）
- **代码：** 待发布（截至 **2026-07-18** 复核：项目页 Header/Footer/Resources 区 **无** GitHub、Hugging Face、Zenodo 等链接；论文 PDF 亦未列公开仓库）
- **入库日期：** 2026-07-15
- **开源状态复核：** 2026-07-18（arXiv 仍为 v1；无新版本）
- **一句话说明：** 官方落地页：text-to-video 预训练 → feed-forward 统一视频感知范式演示、多任务结果视频、架构动画、涌现泛化案例与 BibTeX。

## 页面要点（2026-07 快照）

### Overview / TL;DR

1. 推动 CV 从 **任务专用时代** 走向 **通用视觉智能**（类比 NLP 从专用到 LLM）。
2. **GenCeption** 将预训练 **视频生成模型** 改造成 **单步 feed-forward** 统一感知模型，**文本指令** 切换任务。
3. 三支柱：**生成式视觉预训练**、**统一前向架构**、**SOTA + 涌现行为**（数据效率、sim-to-real、OOD）。

### Methodology 要点

- 预训练 **video diffusion** 捕获 **时空世界先验** 与 **视觉–语言对齐**。
- Post-training 以 **合成数据为主**，将多步扩散改为 **feed-forward**。
- **Paradigm Shift** 图：专用 CV 模型 → 统一 generalist vision model。

### Results 要点

- **SOTA：** 与 DepthAnything3、D4RT、VGGT-Ω、SAM3、Sapiens、DAVID、Genmo、Lotus-2 等 **单任务专家** 可比或更优；区分 **Specialist**（单任务训）与 **Generalist**（多任务联合训）。
- **数据效率：** 同等微调数据下 **视频生成骨干** 优于 V-JEPA、VideoMAE V2；初步 **尺度律**；**7×–500× 更少数据** 可达 D4RT / VGGT-Ω 量级。

### Architecture

- 输入：**RGB 视频 + 文本 prompt（指定输出模态）**。
- **稠密任务：** RGB ambient 空间统一，latent 空间监督。
- **稀疏任务：** 向 DiT 追加 **learnable tokens**。
- 基座形态来自 **text-to-video diffusion（DiT）**。

### Video Any-Task / VLM 能力演示

- 同一模型在 **深度、法线、分割、4D 人体关键点、相机位姿** 等任务间 **无缝切换**。
- **语言指代分割：** 理解颜色、空间关系、运动；对 **未见物体**（如 rocket）泛化。

### Emergent Behaviors

| 现象 | 训练设定 | 测试泛化 |
|------|----------|----------|
| Sim-to-real | 纯合成视频 | 真实 footage，细节超渲染质量 |
| 多实例 | 合成单物体 | 真实多实例场景 |
| OOD 类别 | 仅人类 | 动物、机器人等 |

### 机构（页脚）

Google DeepMind、University of Toronto、University College London、University of Oxford、MIT、Lund University（*Work done while at Google DeepMind*）

### BibTeX

```bibtex
@inproceedings{wang2026genception,
  title     = {Video Generation Models are General-Purpose Vision Learners},
  author    = {Wang, Letian and Zhang, Chuhan and Kabra, Rishabh and Uijlings, Jasper and
               Waslander, Steven and Zisserman, Andrew and Carreira, Joao and He, Kaiming and
               Andriluka, Misha and Bazavan, Eduard Gabriel and Zanfir, Andrei and Sminchisescu, Cristian},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## 对 wiki 的映射

- [GenCeption](../../wiki/entities/genception.md)
- [生成式视觉预训练](../../wiki/concepts/generative-vision-pretraining.md)
