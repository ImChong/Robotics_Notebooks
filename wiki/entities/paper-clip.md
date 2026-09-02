---
type: entity
tags:
  - paper
  - vlm
  - contrastive-learning
  - vision-language
  - openai
status: complete
updated: 2026-09-02
arxiv: "2103.00020"
code: https://github.com/openai/CLIP
related:
  - ./clip.md
  - ./paper-dinov2.md
  - ./paper-openvla.md
  - ../overview/vla-wm-reading-roadmap-14-papers-technology-map.md
sources:
  - ../../sources/papers/clip_arxiv_2103_00020.md
  - ../../sources/blogs/wechat_embodied_ai_lab_vla_wm_reading_roadmap_2026-09-02.md
  - ../../sources/repos/openai-clip.md
summary: "CLIP（arXiv:2103.00020，OpenAI）：图文对比预训练对齐双编码器；VLA 视觉–语言对齐基石。openai/CLIP 已开源。模型实体见 clip.md。"
---

# CLIP：自然语言监督下的可迁移视觉模型

**CLIP**（*Learning Transferable Visual Models From Natural Language Supervision*，[arXiv:2103.00020](https://arxiv.org/abs/2103.00020)，[代码](https://github.com/openai/CLIP)）由 **OpenAI** 提出：用超大规模图文对做 **对比学习**，让匹配图像与文本在嵌入空间靠近。模型/工程实体见 [clip](./clip.md)；**本页是论文 canonical 节点**。

## 一句话定义

**先把「看见」和「读到」拉到同一空间，VLA 才谈得上用语言条件化视觉。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CLIP | Contrastive Language-Image Pre-Training | 本工作 |
| VLM | Vision-Language Model | 后续多模态模型族 |
| ZS | Zero-shot | 无任务微调的分类/检索 |
| SigLIP | Sigmoid Language-Image Pretraining | 语言对齐后继 |

## 为什么重要

- 纳入 [VLA/WM 阅读路线](../../sources/blogs/wechat_embodied_ai_lab_vla_wm_reading_roadmap_2026-09-02.md) 的视觉–语言基座。
- 几乎所有早期 VLA 都直接或间接用 CLIP / 其变体作视觉塔。
- 零样本识别支撑「训练未见物体」叙事（见 [RT-2](./paper-rt-2.md)）。
- **已开源** 推理接口。

## 核心信息

| 项 | 内容 |
|----|------|
| **机构** | 开放人工智能（OpenAI） |
| **结构** | 图像编码器 + 文本编码器 |
| **损失** | 双向对比交叉熵 |
| **能力** | 零样本分类与开放词汇检索 |
| **开源** | **已开源** [openai/CLIP](https://github.com/openai/CLIP) |

### 流程总览

```mermaid
flowchart LR
  img[图像] --> ie[图像编码器]
  txt[文本] --> te[文本编码器]
  ie --> sim[点积 / 温度]
  te --> sim
  sim --> ce[对比损失]
```

## 评测

- 零样本 ImageNet 等分类接近有监督基线（原文表）。
- 机器人侧价值是 **开放词汇对齐**，不是操作成功率。
- 几何细节常弱于 [DINOv2](./paper-dinov2.md)。

## 结论

**VLA 阅读从 CLIP 开始：没有图文对齐，语言指令只是字符串。**

- 对比学习比在图像标签上监督学习更适合开放世界
- 零样本能力解释「为何能认出训练没见过的杯子」
- 操作还要几何特征，单 CLIP 塔往往不够
- [OpenVLA](./paper-openvla.md) 用 SigLIP+DINOv2 补齐
- API/工程细节读 [clip 实体](./clip.md)

## 源码运行时序图

```mermaid
sequenceDiagram
    autonumber
    actor Dev as 开发者
    participant Repo as openai/CLIP
    participant IE as 图像编码器
    participant TE as 文本编码器
    Dev->>Repo: pip/clone + 权重
    Dev->>IE: 编码图像
    Dev->>TE: 编码候选文本
    IE-->>Dev: 相似度 / 零样本标签
```

## 局限与风险

- **空间弱：** 深度、接触、细粒度几何不是 CLIP 训练目标。
- **偏见与数据：** 网页图文噪声进入下游 VLA。
- **不是策略：** 不能当动作模型用。

## 与其他工作对比

| 工作 | 相对本页 |
|------|----------|
| [DINOv2](./paper-dinov2.md) | 自监督几何特征更强 |
| [OpenVLA](./paper-openvla.md) | 双塔消费 CLIP 后继 |
| [RT-2](./paper-rt-2.md) | 把对齐后的 VLM 接到动作 |

## 关联页面

- [CLIP 模型实体](./clip.md)
- [DINOv2](./paper-dinov2.md)
- [OpenVLA](./paper-openvla.md)
- [VLA/WM 14 篇路线](../overview/vla-wm-reading-roadmap-14-papers-technology-map.md)

## 推荐继续阅读

- [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- [openai/CLIP](https://github.com/openai/CLIP)

## 参考来源

- [clip_arxiv_2103_00020](../../sources/papers/clip_arxiv_2103_00020.md)
- [具身智能研究室 VLA/WM 阅读路线](../../sources/blogs/wechat_embodied_ai_lab_vla_wm_reading_roadmap_2026-09-02.md)
- [openai-clip](../../sources/repos/openai-clip.md)
