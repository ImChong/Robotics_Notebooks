---
type: entity
tags:
  - paper
  - bimanual
  - manipulation
status: complete
updated: 2026-10-29
arxiv: "2610.29716"
project_id: reconstruction-augmented-bimanual-pretraining-rgb-human-video
project: https://spatial-dex.github.io/
related:
  - ../overview/senlanke-weekly-2026-10-19-23-manipulation-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
  - ../overview/embodied-station-2026-10-29-58-papers-technology-map.md
sources:
  - ../../sources/papers/reconstruction-augmented-bimanual-pretraining-rgb-human-video_arxiv_2610_29716.md
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md
  - ../../sources/blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md
summary: "Recon-Augmented Bimanual Pretrain：利用三维重建增强的视觉运动预训练，从纯 RGB 人类视频学习双臂灵巧操作（arXiv:2610.29716）：将三维重建引入视觉运动预训练，使模型能够利用纯 RGB 视频中的三维操作信息学习双臂灵巧能力。"
---

# Recon-Augmented Bimanual Pretrain：利用三维重建增强的视觉运动预训练，从纯 RGB 人类视频学习双臂灵巧操作

**Recon-Augmented Bimanual Pretrain**（*Reconstruction-Augmented Visuomotor Pre-Training of 3D-Grounded Bimanual Dexterity from RGB-Only Human Videos*，[arXiv:2610.29716](https://arxiv.org/abs/2610.29716)）收录于 senlanke「具身运控lab」10-19–10-23 前沿论文动态（Manipulation）。本页依据**周报条目 + arXiv 摘要页**整理，流程图依据论文方法部分（见参考来源），未逐页核对实验数字；指标与结论以论文为准。

## 一句话定义

**Recon-Augmented Bimanual Pretrain：利用三维重建增强的视觉运动预训练，从纯 RGB 人类视频学习双臂灵巧操作——将三维重建引入视觉运动预训练，使模型能够利用纯 RGB 视频中的三维操作信息学习双臂灵巧能力。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |
| IL | Imitation Learning | 模仿学习 |

## 为什么重要

- 周报所述待解决的问题：人类 RGB 视频缺少机器人动作标签和直接可用的三维操作状态，限制了双臂策略从大规模人类视频中学习。

## 流程总览

```mermaid
flowchart TD
    subgraph 数据["数据管线: RGB-only 人类视频"]
        V[RGB-only 人类视频] --> H[现成手部估计器<br/>3D 手部 mesh 与手腕轨迹]
        V --> S[单目重建<br/>场景点云与相机位姿, 尺度未知]
        H --> K[手部尺度标定<br/>手部点与 MANO mesh 对齐<br/>求整段视频共享尺度 + 逐帧刚体变换]
        S --> K
        K --> C[坐标系对齐<br/>未来几何与手腕轨迹变换到当前相机系]
        C --> T[对齐后的深度观测<br/>手腕/手指动作目标<br/>未来几何目标]
    end
    subgraph 策略["3D-Grounded 策略 PaliGemma 初始化"]
        R[RGB 图像块特征 x] --> F
        D[深度按图像块采样并反投影<br/>Sonata 联合编码, 池化回图像块得几何网格 g] --> F[渐进式双门控残差融合<br/>f = x + λ·Po α⊙x + β⊙g<br/>λ 从 0 线性 warm-up 到 1]
        F --> X[共享 VLM 上下文 c<br/>融合特征 + 语言指令 + 本体状态]
        X --> W[Wrist Expert 扩散 Transformer<br/>预测双手腕轨迹]
        W --> G[Finger Expert 扩散 Transformer<br/>以手腕计划与 c 为条件预测手指动作]
        X --> P[Geometry Predictor 仅训练时使用<br/>预测未来几何, Chamfer 损失]
    end
    T --> R
    T --> D
    T -. 监督 .-> W
    T -. 监督 .-> G
    T -. 目标 .-> P
    G --> A[预训练 100 万人类视频片段<br/>少量机器人示范微调<br/>传感器深度走同一几何编码器]
    A --> Y[部署: 当前 RGB-D + 指令 + 机器人状态<br/>输出双臂动作块, 移除前向预测头]
```

图依据：论文 §III-A–C / Fig.2 / Fig.3 / 式(1)–(5)

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.29716](https://arxiv.org/abs/2610.29716) · [PDF](https://arxiv.org/pdf/2610.29716) |
| **作者（arXiv）** | Xingyi He, Zhiwen Qiu, Adhitya Polavaram, Yunhao Cao, Cory Fan, Julius Samwer, Saleh Kalantari, Kuan Fang |
| **机构（周报所述，未逐条核实）** | 待核实（作者：Xingyi He、Zhiwen Qiu、Kuan Fang 等）。 |
| **arXiv 评论栏** | Project page: this https URL |
| **开源** | arXiv 摘要页/评论给出项目页或外链（https://spatial-dex.github.io/）；未发现代码仓库链接（2026-10-26） |
| **周报所述创新点** | 将三维重建引入视觉运动预训练，使模型能够利用纯 RGB 视频中的三维操作信息学习双臂灵巧能力。 |

### arXiv 摘要节选（英文原文）

> We present SpatialDex, a framework that learns 3D-grounded bimanual dexterity from RGB-only human videos through reconstruction-augmented visuomotor pre-training. The central idea is to learn human behavior together with its spatial context.

## 源码运行时序图

**不适用**：arXiv 摘要页/评论给出项目页或外链（https://spatial-dex.github.io/）；未发现代码仓库链接，无可对齐的官方代码流程。

## 实验与评测

- 周报与摘要给出的数字/场景已写入上表「创新点」与摘要节选；其余实验细节、基线与成功率定义请回到论文正文。
- 读法：先对齐任务设定、传感器与成功定义（仿真/真机），再解读 headline 数字，勿跨论文直接比较。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见 [Manipulation周报技术地图](../overview/senlanke-weekly-2026-10-19-23-manipulation-technology-map.md)，按方向浏览同期论文 |
| **开源状态** | arXiv 摘要页/评论给出项目页或外链（https://spatial-dex.github.io/）；未发现代码仓库链接 |

## 结论

**Recon-Augmented Bimanual Pretrain：利用三维重建增强的视觉运动预训练，从纯 RGB 人类视频学习双臂灵巧操作。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-19-23-manipulation-technology-map](../overview/senlanke-weekly-2026-10-19-23-manipulation-technology-map.md)
- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)

## 参考来源

- [reconstruction-augmented-bimanual-pretraining-rgb-human-video_arxiv_2610_29716.md](../../sources/papers/reconstruction-augmented-bimanual-pretraining-rgb-human-video_arxiv_2610_29716.md)
- [wechat_senlanke_weekly_manipulation_2026-10-19_23.md](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md)
- [arXiv:2610.29716](https://arxiv.org/abs/2610.29716)
- [arXiv PDF](https://arxiv.org/pdf/2610.29716)：已通读方法部分（实验与附录以略读为准）：摘要、§I 引言、§II 相关工作、§III 方法（A–C）、§IV-A 实验设定开头

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.29716)
