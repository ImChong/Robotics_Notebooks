---
type: entity
tags:
  - paper
  - world-action-model
  - manipulation
status: complete
updated: 2026-10-29
arxiv: "2610.28613"
project_id: physical-canvas-vision-tactile-world-model
project: https://zebinx.github.io/PhysCanvas_HomePage/
related:
  - ../overview/senlanke-weekly-2026-10-19-23-manipulation-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
  - ../overview/embodied-station-2026-10-29-58-papers-technology-map.md
sources:
  - ../../sources/papers/physical-canvas-vision-tactile-world-model_arxiv_2610_28613.md
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md
  - ../../sources/blogs/wechat_embodied_station_58_papers_diverse_demonstrations_2026-10-29.md
summary: "Physical Canvas：面向机器人策略评估与学习的视觉触觉世界模型（arXiv:2610.28613）：Physical Canvas 将视觉与触觉共同用于世界建模，研究通过多模态物理预测支持机器人策略评估和策略学习。"
---

# Physical Canvas：面向机器人策略评估与学习的视觉触觉世界模型

**Physical Canvas**（*Physical Canvas: A Vision Tactile World Model For Robot Policy Evaluation And Learning*，[arXiv:2610.28613](https://arxiv.org/abs/2610.28613)）收录于 senlanke「具身运控lab」10-19–10-23 前沿论文动态（Manipulation）。本页依据**周报条目 + arXiv 摘要页**整理，流程图依据论文方法部分（见参考来源），未逐页核对实验数字；指标与结论以论文为准。

## 一句话定义

**Physical Canvas：面向机器人策略评估与学习的视觉触觉世界模型——Physical Canvas 将视觉与触觉共同用于世界建模，研究通过多模态物理预测支持机器人策略评估和策略学习。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WAM | World Action Model | 世界–动作联合建模 |
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |

## 为什么重要

- 周报所述待解决的问题：仅根据视觉预测操作结果，难以完整表达接触丰富任务中的物理交互。机器人策略评估还需要考虑接触状态及其变化。

## 流程总览

```mermaid
flowchart TD
    subgraph 画布["统一多模态画布 Unified Canvas"]
        O[多视角 RGB / 触觉图像 / 标记点位移 / 任务状态<br/>位移 xyz 映射为 RGB 通道, 状态以颜色与纹理编码] --> X[Rasterize: 摄像头横向排列, 各模态图块纵向堆叠成画布 x]
        AC[关节目标序列 a_t:t+H<br/>URDF + 相机内外参投影为各视角 action image m] --> ZA[视频 VAE 编码并 tokenize 得 z_a]
        X --> Z[视频 VAE 编码并 tokenize 得画布 token z]
        Z --> F[去噪时位置对齐相加 z_a<br/>任务状态 token 位置不加动作]
        ZA --> F
        F --> W[Wan 2.2 扩散 Transformer<br/>参考 token r 输入 + 初始 RGB 锚 z_anc 交叉注意力<br/>flow matching 训练]
        W --> P[预测未来 H 帧多模态画布]
    end
    subgraph 后训练["一致性感知后训练 冻结奖励模型"]
        P --> DEC[由预测速度恢复干净画布估计并 VAE 解码]
        DEC --> RS[State Reward Model<br/>BCE 对齐真实任务结果]
        DEC --> RC[Scene Reward Model<br/>最大化场景保持得分]
        DEC --> RM[Modality Reward Model<br/>最大化 RGB-触觉-位移兼容得分]
        RS --> U[更新世界模型<br/>并保留 flow matching MSE 损失]
        RC --> U
        RM --> U
    end
    subgraph 应用["策略评估与学习"]
        P --> E[闭环策略 rollout<br/>策略读生成观测并输出动作, 比较世界模型与真实成功率]
        P --> G[想象交互 + State Reward Model 打分<br/>GRPO 仅更新策略]
    end
```

图依据：论文 §3.2–3.4 / Fig.3

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.28613](https://arxiv.org/abs/2610.28613) · [PDF](https://arxiv.org/pdf/2610.28613) |
| **作者（arXiv）** | Zebin Xing, Yuhang Zheng, Yupeng Zheng, Yichen Zhang, Zichen Li, Shangqing Zhou, Yujie Zang, Yuchen Liu, Zhongpu Xia, Haoran Li, Qichao Zhang |
| **机构（周报所述，未逐条核实）** | 待核实（作者：Zebin Xing、Yuhang Zheng、Yichen Zhang 等）。 |
| **arXiv 评论栏** | （无） |
| **开源** | arXiv 摘要页/评论给出项目页或外链（https://zebinx.github.io/PhysCanvas_HomePage/）；未发现代码仓库链接（2026-10-26） |
| **周报所述创新点** | Physical Canvas 将视觉与触觉共同用于世界建模，研究通过多模态物理预测支持机器人策略评估和策略学习。 |

### arXiv 摘要节选（英文原文）

> Evaluating and post-training robot policies through physical interaction is costly and difficult to scale, motivating action-conditioned world models as interactive simulators. However, visually plausible predictions can misrepresent contact dynamics, providing unreliable feedback for policy evaluation and learning.

## 源码运行时序图

**不适用**：arXiv 摘要页/评论给出项目页或外链（https://zebinx.github.io/PhysCanvas_HomePage/）；未发现代码仓库链接，无可对齐的官方代码流程。

## 实验与评测

- 周报与摘要给出的数字/场景已写入上表「创新点」与摘要节选；其余实验细节、基线与成功率定义请回到论文正文。
- 读法：先对齐任务设定、传感器与成功定义（仿真/真机），再解读 headline 数字，勿跨论文直接比较。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见 [Manipulation周报技术地图](../overview/senlanke-weekly-2026-10-19-23-manipulation-technology-map.md)，按方向浏览同期论文 |
| **开源状态** | arXiv 摘要页/评论给出项目页或外链（https://zebinx.github.io/PhysCanvas_HomePage/）；未发现代码仓库链接 |

## 结论

**Physical Canvas：面向机器人策略评估与学习的视觉触觉世界模型。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-19-23-manipulation-technology-map](../overview/senlanke-weekly-2026-10-19-23-manipulation-technology-map.md)
- [manipulation](../tasks/manipulation.md)
- [vla](../methods/vla.md)

## 参考来源

- [physical-canvas-vision-tactile-world-model_arxiv_2610_28613.md](../../sources/papers/physical-canvas-vision-tactile-world-model_arxiv_2610_28613.md)
- [wechat_senlanke_weekly_manipulation_2026-10-19_23.md](../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-19_23.md)
- [arXiv:2610.28613](https://arxiv.org/abs/2610.28613)
- [arXiv PDF](https://arxiv.org/pdf/2610.28613)：已通读方法部分（实验与附录以略读为准）：摘要、§1 引言、§3 方法（3.1–3.4）、§4.1 实验设定

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.28613)
