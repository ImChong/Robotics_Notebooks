---
type: method
tags: [data-engine, vlm, labeling, data-collection, machine-learning]
status: complete
updated: 2026-04-21
related:
  - ../concepts/embodied-scaling-laws.md
  - ../methods/vla.md
  - ../queries/demo-data-collection-guide.md
sources:
  - ../../sources/papers/perception.md
summary: "自动化标注流水线（Auto-labeling Pipelines）利用大视觉语言模型（VLM）自动为海量机器人原始轨迹生成文本描述和成功率标签，极大地降低了具身基础模型的数据准备成本。"
---

# Auto-labeling Pipelines (自动化标注流水线)

**自动化标注流水线** 是构建具身基础模型（Foundation Models）的关键基础设施。它解决了具身智能中最昂贵的环节：**让数据具备语义（Semantic Grounding）**。

## 核心任务

在海量的机器人交互数据（来自遥操作、自主探索或仿真）中，流水线需要自动完成以下标注：

1. **指令生成**：根据视频序列，反向推导出该动作对应的语言指令（如“把红色的球放进篮子里”）。
2. **成功率判定**：自动识别轨迹是否完成了既定目标，作为强化学习的稀疏奖励。
3. **关键帧提取**：识别轨迹中的接触点、转折点，用于训练 [Action Chunking](./action-chunking.md)。

## 主要技术路线

现代流水线通常采用 **Teacher-Student 架构**：

- **VLM Teacher**：使用顶尖的多模态大模型（如 GPT-4V, Gemini 1.5 Pro），输入轨迹视频。
- **Prompt Engineering**：设计精细的提示词，要求模型输出标准化的 JSON 描述。
- **后处理**：将 VLM 输出的文本描述与物理传感器数据（关节力矩、IMU）进行对齐校验。

## 代表性项目

- **Auto-RT (DeepMind)**：利用 VLM 在真实的办公环境中自主选择任务、自主执行并自动生成训练数据。
- **CLAW (宇树 G1 数据管线)**：通过物理仿真和组合原子动作，自动生成带精准语言标签的全身运动轨迹。见 [CLAW](./claw.md)。
- **ROSS (Robot Open-world Success Supervision)**：利用视觉语言模型作为通用的奖励函数。
- **SceneVerse++ (BIGAI 等)**：从互联网无标注视频做 SfM 与稠密重建，再自动生成实例分割、[3D 空间 VQA](../concepts/3d-spatial-vqa.md) 与 [VLN](../tasks/vision-language-navigation.md) 监督；侧重 **3D 场景理解** 数据引擎而非单条轨迹描述。见 [SceneVerse++](../entities/sceneverse-pp.md)。

## 带来的优势

- **极低的标注成本**：相比于昂贵的人工标注，API 调用的成本几乎可以忽略不计。
- **数据多样性**：模型可以自动描述轨迹中的环境细节（如“在昏暗的光线下推开木门”），丰富了预训练分布。

## 关联页面
- [具身规模法则 (Scaling Laws)](../concepts/embodied-scaling-laws.md)
- [VLA (Vision-Language-Action Models)](./vla.md)
- [SceneVerse++（互联网 3D 场景数据）](../entities/sceneverse-pp.md)
- [演示数据采集指南](../queries/demo-data-collection-guide.md)

## 参考来源
- Ahn, M., et al. (2024). *Auto-RT: Embodied AI with Foundation Models*.
- [Google DeepMind Blog on Scaling Robot Learning](https://deepmind.google/discover/blog/scaling-robot-learning-with-auto-rt/).
- [SceneVerse++ 原始资料](../../sources/repos/sceneverse-pp.md) — 网页视频重建与 3D 自动标注流水线案例（CVPR 2026）
