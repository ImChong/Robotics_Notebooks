---
type: entity
tags:
  - paper
  - world-action-model
  - manipulation
status: complete
updated: 2026-11-02
arxiv: "2610.37091"
project_id: gawm-compact-latent-world-action-model
related:
  - ../overview/senlanke-weekly-2026-10-26-30-manipulation-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/gawm-compact-latent-world-action-model_arxiv_2610_37091.md
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md
summary: "GAWM：面向多任务机器人操作的紧凑潜在世界动作模型（arXiv:2610.37091）：使用冻结的 DINOv3 Encoder，将每个相机视角压缩成少量 Token。轻量预测器根据任务条件预测两个未来时间尺度的视觉 Latent，再由动作 Decoder 生成动作块。模型参数量低于 1"
---

# GAWM：面向多任务机器人操作的紧凑潜在世界动作模型

**GAWM**（*GAWM: A Compact Latent World-Action Model for Multi-Task Robot Manipulation*，[arXiv:2610.37091](https://arxiv.org/abs/2610.37091)）收录于 senlanke「具身运控lab」10-26–10-30 前沿论文动态（Manipulation）。本页依据**周报条目 + arXiv 摘要页**整理，流程图依据论文方法部分（见参考来源），未逐页核对实验数字；指标与结论以论文为准。

## 一句话定义

**GAWM：面向多任务机器人操作的紧凑潜在世界动作模型——使用冻结的 DINOv3 Encoder，将每个相机视角压缩成少量 Token。轻量预测器根据任务条件预测两个未来时间尺度的视觉 Latent，再由动作 Decoder 生成动作块。模型参数量低于 100M，无需机器人或视频预训练，在 RoboTwin 2.0 上超过论文比较的部分十亿参数 WAM。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界–动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |

## 为什么重要

- 周报所述待解决的问题：大型 WAM 通常依赖数十亿参数的视觉语言或视频生成模型，训练成本和推理延迟较高。

## 流程总览

```mermaid
flowchart TD
  A["多相机 RGB o_t + 机器人状态 s_t + 任务标识 k"] --> B["冻结 DINOv3 ViT 编码每个视角<br/>融合 3 个中间层 patch 特征"]
  B --> C["查询适配器 A：64 个可学习查询 → 每视角 64 个视觉 token<br/>叠加相机嵌入，堆叠成 z_t（双相机 K=128）"]
  G["任务码 g_k：由任务 k 的示范计算，经 MLP 映射<br/>(沿用 LiLa-WAM)"]
  C --> D["任务条件潜变量预测器 F（4–6M 参数）<br/>并行预测两个未来时域 h1 = H/2、h2 = H 的 token<br/>残差更新：LN(z_t + W_o r_j)"]
  G --> D
  C --> E["动作解码器 D（ACT 风格）<br/>H 个动作查询 attend 记忆 M_t = [z_t; 预测 token ×2; 状态 token ×2; 任务 token]"]
  D --> E
  G --> E
  S["状态 s_t"] --> E
  E --> F["MLP 直接回归整段动作块 â（tanh，ℓ1 损失）<br/>确定性回归，无迭代采样、无动作搜索"]
  D -.仅训练.-> H["Patch 解码器 G：预测 token → 未来帧 patch 特征<br/>与冻结 DINOv3 目标做余弦损失 L_fut"]
  H -.-> I["总损失 L = L_act + λ_fut L_fut + λ_cur L_cur<br/>+ λ_reg (L_div + L_var) + λ_tmp L_tmp"]
  F -.-> I
```

图依据：论文 §III-A–E / Fig. 2、式 (1)–(6)。

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.37091](https://arxiv.org/abs/2610.37091) · [PDF](https://arxiv.org/pdf/2610.37091) |
| **作者（arXiv）** | Liu, Yang, Liu, Xiaoli, Li, Liuzhao, Song, Yuduo, Liu, Muyuan, Gao, Xiang |
| **机构（周报所述，未逐条核实）** | Genisom AI。 |
| **arXiv 评论栏** | （无） |
| **开源** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02） |
| **周报所述创新点** | 使用冻结的 DINOv3 Encoder，将每个相机视角压缩成少量 Token。轻量预测器根据任务条件预测两个未来时间尺度的视觉 Latent，再由动作 Decoder 生成动作块。模型参数量低于 100M，无需机器人或视频预训练，在 RoboTwin 2.0 上超过论文比较的部分十亿参数 WAM。 |

### arXiv 摘要节选（英文原文）

> World-action models (WAMs) incorporate the prediction of future observations into robot policy learning. However, most WAMs build on billion-parameter vision-language or video-generation backbones, which incur high training cost and inference latency.

## 源码运行时序图

**不适用**：arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布），无可对齐的官方代码流程。

## 实验与评测

- 周报与摘要给出的数字/场景已写入上表「创新点」与摘要节选；其余实验细节、基线与成功率定义请回到论文正文。
- 读法：先对齐任务设定、传感器与成功定义（仿真/真机），再解读 headline 数字，勿跨论文直接比较。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见 [Manipulation周报技术地图](../overview/senlanke-weekly-2026-10-26-30-manipulation-technology-map.md)，按方向浏览同期论文 |
| **开源状态** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布） |

## 结论

**GAWM：面向多任务机器人操作的紧凑潜在世界动作模型。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-26-30-manipulation-technology-map](../overview/senlanke-weekly-2026-10-26-30-manipulation-technology-map.md)
- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)

## 参考来源

- [gawm-compact-latent-world-action-model_arxiv_2610_37091.md](../../sources/papers/gawm-compact-latent-world-action-model_arxiv_2610_37091.md)
- [wechat_senlanke_weekly_manipulation_2026-10-26_30.md](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
- [arXiv:2610.37091](https://arxiv.org/abs/2610.37091)
- [arXiv PDF](https://arxiv.org/pdf/2610.37091)：已通读方法部分（实验与附录以略读为准）：方法 §III-A–E（问题形式化、视觉编码、任务条件预测器、预测监督、动作解码器与训练目标）与 §IV-A 实验设定

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.37091)
