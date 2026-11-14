---
type: entity
project_id: emvit-online-metric-depth
tags: [paper, robotics, ingest-2026-11]
status: draft
updated: 2026-11-14
arxiv: "2611.14958"
project: https://emvit-mde.github.io/
related:
  - ./paper-target-aware-gaussian-langmap.md
  - ./paper-trace-conversational-humanoid-motion.md
original:
  - ../../sources/papers/emvit-online-metric-depth_arxiv_2611_14958.md
sources:
  - ../../sources/papers/emvit-online-metric-depth_arxiv_2611_14958.md
  - ../../sources/sites/emvit-online-metric-depth.md
summary: "在线将单目相对深度校正到米制尺度"
---

# EM-ViT：机器人运动辅助的单目深度在线尺度适配

## 一句话定义

**把相对深度的局部结构细化与全局尺度预测解耦，用稀疏深度监督、可选机器人自运动和梯度匹配在线调整单目深度。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| MDE | Monocular Depth Estimation | 单目深度估计 |
| ViT | Vision Transformer | 视觉 Transformer |
| EM | Ego-Motion | 机器人自身运动 |
| LiDAR | Light Detection and Ranging | 激光测距 |

## 为什么重要

在线将单目相对深度校正到米制尺度，关注现有控制、数据或机器人学习方法在复杂任务中的失效点。这里单独建论文实体，避免与其他论文共用详情页；当前是待完成全文与源码复核的草稿。

## 核心机制

读取基础模型的仿射不变逆深度；dRefNet 细化视差图、ViT 预测有效基线尺度；借助稀疏 LiDAR、光度重建和梯度匹配，并可选 odomNet 优化历史相机位姿。

## 方法流程（待全文逐节复核）

以下流程仅依据已核对的论文 HTML 方法描述，不代替源代码调用图。

```mermaid
flowchart LR
  rgb[当前图像] --> prior[单目深度基础模型]
  prior --> dref[视差结构细化]
  rgb --> scale[有效基线尺度预测]
  lidar[稀疏 LiDAR 与历史运动] --> adapt[在线自监督适配]
  adapt --> dref
  adapt --> scale
  dref --> depth[米制深度图]
  scale --> depth
```

## 源码运行时序图

**待核实，不绘制未经代码阅读确认的时序图。** 目前未确认可运行官方源码，待项目源文件发布或访问恢复后补充。

## 实验与评测

论文在 KITTI 等室内外公开序列评估深度精度与适应速度；结果针对有稀疏测距/运动信息支持的在线校准，不是完全无标定的单目绝对尺度恢复。

## 局限与工程实践

- 动态物体、光度重建退化和稀疏 LiDAR 噪声会影响自监督；项目页 Code 链接在核查时仍指向占位仓库，不能声称已发布运行代码。
- **源码状态：** 项目页 Code 链接指向 github.com/YOUR REPO HERE 占位地址，截至 2026-11-14 不能判定代码开放。
- **原文状态：** 本轮已核查 arXiv 部分章节，尚未通读全部正文/附录，保持 draft，不宣称健康度满分。

## 结论

**工程上应分别评估深度相对结构误差与绝对尺度误差，而非只看合并指标。**

1. 使用与论文一致的任务、硬件和评测协议复验，不跨实验口径比较数据。
2. 把方法改善的指标与额外的数据、计算或控制成本分开记录。
3. 对未经独立核查的复现入口，暂不声称完成验证。

## 关联页面

- [目标感知的 3D Gaussian 语言地图免训练细化](./paper-target-aware-gaussian-langmap.md)
- [TRACE：具有参考保持约束的对话式人形动作编辑](./paper-trace-conversational-humanoid-motion.md)

## 参考来源

- [论文归档](../../sources/papers/emvit-online-metric-depth_arxiv_2611_14958.md)
- [官方项目页归档](../../sources/sites/emvit-online-metric-depth.md)

## 推荐继续阅读

- [arXiv 原文](https://arxiv.org/abs/2611.14958)
- [官方项目页](https://emvit-mde.github.io/)
