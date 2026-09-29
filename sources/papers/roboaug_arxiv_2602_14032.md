# RoboAug（区域对比式语义数据增广）

> 来源归档（ingest）

- **标题：** RoboAug: One Annotation to Hundreds of Scenes via Region-Contrastive Data Augmentation for Robotic Manipulation
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2602.14032>
- **机构：** 北京人形机器人创新中心（X-Humanoid）；慕尼黑工业大学（TUM）；香港城市大学（CityU）；北京航空航天大学（Beihang）；北京大学（PKU）
- **项目页：** <https://x-roboaug.github.io/>
- **代码：** 截至入库日项目页标注 **Code (Coming Soon)**（论文摘要承诺将开源 RoboAug-D 与多任务真机数据集）
- **入库日期：** 2026-09-29
- **一句话说明：** 单任务 IL 下仅需 **一帧 bbox** 标注，经 GroundingDINO + DINOv2 一次性匹配与 SAM2 传播得稠密 mask，Stable Diffusion v3 全图背景合成 + 前景合成扩场景，并 plug-and-play **区域对比损失** 约束视觉编码器；35k+ 真机 rollout 上三因子 OOD 成功率相对无增广基线 UR **0.09→0.47**、AgileX **0.16→0.60**、天工 2.0 **0.19→0.67**。

## 核心摘录（MVP）

### 1) 任务相关区域提取（one-shot）

- **摘录要点：** 每条轨迹首帧作 anchor；**单张参考图** 手工标 \(K\) 个 bbox → DINOv2 得参考 embedding；GroundingDINO 在其余轨迹首帧出候选框，余弦相似度 argmax 对齐类别；SAM2 跟踪传播为全轨迹像素级 \(M_{\text{task}}\)。
- **对 wiki 的映射：**
  - [RoboAug](../../wiki/entities/paper-roboaug.md)

### 2) 语义数据增广（非 inpainting 全背景）

- **摘录要点：** ChatGPT 生成 500 条桌面材质 prompt 库；**Stable Diffusion v3** 生成完整背景 \(I_{\text{bg}}\)，再 \(I_{\text{aug}} = M_{\text{task}} \odot I + (1-M_{\text{task}}) \odot I_{\text{bg}}\)，避免 inpainting 在机械臂遮挡下的几何伪影。
- **对 wiki 的映射：**
  - [RoboAug](../../wiki/entities/paper-roboaug.md)

### 3) 区域对比策略学习（RCL）

- **摘录要点：** 对增广 batch 中每类物体裁 \(I_{\text{obj}}\)，经共享视觉编码器；全图特征 \(z\) 经空间自注意力加权到 \(z_{\text{obj}}\) 得 \(z_{\text{att}}\)；**监督对比损失** \(\mathcal{L}_{\text{RC}}\) 与 IL 损失联合，架构无关、即插即用。
- **对 wiki 的映射：**
  - [RoboAug](../../wiki/entities/paper-roboaug.md)
  - [Manipulation 任务](../../wiki/tasks/manipulation.md)

### 4) RoboAug-D 与 VFM 检测瓶颈

- **摘录要点：** **RoboAug-D**：33 任务、7576 轨迹、73,749 关键帧、366,835 bbox、46 类；GroundingDINO / LLMDet 在操纵视角 mAP@0.5 显著偏低；one-shot 匹配相对 SOTA 检测器 mAP 提升 **34.6% / 25.0%**（论文 Fig.3）。
- **对 wiki 的映射：**
  - [RoboAug](../../wiki/entities/paper-roboaug.md)
  - [RoboAug 项目页](../sites/x-roboaug-project.md)

### 5) 真机泛化评测（35k trials）

- **摘录要点：** UR-5e、AgileX Cobot Magic 2.0、**天工 2.0**；三因子（3 背景 × 4 光照 × 3 干扰物）组合 OOD 平均成功率 RoboAug **0.47 / 0.60 / 0.67** vs GenAug **0.31 / 0.34 / 0.42** vs 无增广 **0.09 / 0.16 / 0.19**；单因子可达 170  unseen 背景等。
- **对 wiki 的映射：**
  - [RoboAug](../../wiki/entities/paper-roboaug.md)

## 当前提炼状态

- [x] 项目页开源核查（Code/Dataset **Coming Soon**；论文承诺后续开源）
- [x] wiki 映射：`wiki/entities/paper-roboaug.md` 新建
