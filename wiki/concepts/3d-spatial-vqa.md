---
type: concept
tags: [vqa, 3d-scene-understanding, vlm, spatial-reasoning, embodied-ai]
summary: "3D 空间 VQA 要求模型根据多视图或视频观测回答关于物体几何关系、距离、方位与房间尺度等问题，是检验视觉–语言模型空间推理能力的关键任务。"
updated: 2026-05-07
status: complete
related:
  - ../entities/sceneverse-pp.md
  - ../tasks/vision-language-navigation.md
  - ../methods/vla.md
sources:
  - ../../sources/repos/sceneverse-pp.md
---

# 3D 空间 VQA（3D Spatial Visual Question Answering）

**3D 空间 VQA**：在 **三维室内场景** 条件下，模型需要结合视觉观测与自然语言问题，推理物体间 **几何关系**（远近、相对方位、计数、尺度、路径顺序等）并给出答案——常见形式包括选择题与数值题。

## 为什么重要？

- **与传统 2D VQA 的差异**：单张 RGB 缺乏可靠尺度与遮挡背后的几何；3D 空间 VQA 逼迫模型利用 **多视图一致性**、**深度/几何先验** 或显式 **场景图**，更接近机器人「在房间里弄明白方位」的需求。
- **与 VLM / VLA 的关系**：网页图文训练的 VLM 擅长语义与外观，但对「左转后哪个物体更近」类 **度量空间关系** 往往偏弱；需要 3D 监督数据（自动或人工）补足。
- **评测载体**：例如 **VSI-Bench** 等基准将 egocentric 室内扫描转为多类型空间 QA，用于对比不同训练数据来源（真实扫描、合成、互联网重建等）。

## 核心结构（典型管线）

1. **场景表示**：点云/网格、或从多视图 lift 得到的 **3D 场景图**（实例节点 + 空间关系边）。
2. **命题生成**：模板或 LLM/VLM 基于场景图生成 QA 对，控制题型（计数、相对距离、朝向、房间大小、路线规划等）。
3. **模型**：在通用 VLM 上 LoRA/SFT，输入仍为图像序列或视频帧，但标签强调 **空间推理** 而非仅物体识别。

自动构造数据时要注意 **域差距**：互联网重建场景与实验室扫描设备采集的分布不同，可能出现「某类题型涨、某类题型掉」——需与数据来源与过拟合一起讨论。

## 常见误区

- **误区**：「接了深度估计就等于会做 3D VQA。」深度网络给出像素级距离，但 **关系推理**（多物体约束、组合计数）仍是独立难点。
- **误区**：「只在 ScanNet 等标注扫描上训练就足够泛化。」真实部署会遇到家具分布、户型与相机轨迹差异；互联网规模数据的价值在于 **覆盖与多样性**，但仍需警惕标注噪声。

## 与其他页面的关系

- **数据实体**：[SceneVerse++](../entities/sceneverse-pp.md) 从网页视频构建大规模空间 VQA，并与 VSI-Bench 格式对齐以验证迁移。
- **兄弟任务**：[视觉–语言导航（VLN）](../tasks/vision-language-navigation.md) 同样依赖「语言 + 空间」，但更强调 **动作序列** 而非单一答案。
- **模型范式**：[VLA](../methods/vla.md) 若承载高层「去哪、找什么」的语义，底层仍需可靠几何感知；3D 空间 VQA 可作为 **空间推理能力** 的中间监督或评测。

## 参考来源

- [SceneVerse++ 原始资料归档](../../sources/repos/sceneverse-pp.md)
- Chen et al., *Lifting Unlabeled Internet-level Data for 3D Scene Understanding* (arXiv:2604.01907) — 3D spatial VQA 数据生成与 VSI-Bench 实验设定

## 关联页面

- [SceneVerse++](../entities/sceneverse-pp.md)
- [视觉–语言导航（VLN）](../tasks/vision-language-navigation.md)
- [VLA](../methods/vla.md)

## 推荐继续阅读

- VSI-Bench 原文与数据卡片（SceneVerse++ 论文中作为主要 3D 空间 VQA 评测之一）
- VLM-3R 等「从 3D 场景生成推理数据」的相关工作（可与 SceneVerse++ 对照）
