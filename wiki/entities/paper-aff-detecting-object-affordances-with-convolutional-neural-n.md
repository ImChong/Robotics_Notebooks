---
type: entity
tags: [paper, curated-index, awesome-affordance-learning, affordance-learning, embodied-ai]
status: complete
updated: 2026-11-26
project_id: aff-detecting-object-affordances-with-convolutional-neural-n
url: "https://www.csc.liv.ac.uk/~anguyen/assets/pdfs/2016_IROS_Affordance.pdf"
venue: "IROS 2016"
summary: "IROS 2016 论文，采用无全连接层的对称 13 层 VGG16 编码–解码 CNN 从 RGB-D 图像实时预测像素级物体部件 affordance，在 UMD 数据集取得 0.770 平均 F_beta^w（90 ms/帧），并经 Mean Shift 聚类与最小外接矩形框驱动 WALK-MAN 人形机器人实现 92.2% 实机抓取成功率。"
related:
  - ../entities/paper-affordance-learning-embodied-ai-survey.md
  - ../overview/affordance-awesome-technology-map.md
  - ../tasks/manipulation.md
  - ../methods/vla.md
sources:
  - ../../sources/papers/affordance_awesome_detecting-object-affordances-with-convolutional-neural-n.md
  - ../../sources/papers/affordance_awesome_catalog.md
  - ../../sources/repos/awesome-affordance-learning.md
---

# Detecting Object Affordances with Convolutional Neural Networks

**Detecting Object Affordances with Convolutional Neural Networks**（IROS 2016，Istituto Italiano di Tecnologia）收录于 [Awesome-Affordance-Learning](https://github.com/hq-King/Awesome-Affordance-Learning)，属于具身 AI affordance 学习综述的配套文献。本页基于论文全文完成结构化精读与核实。

## 一句话定义

早期将**无全连接层深度卷积编码–解码器（基于 VGG16 对称 13 层 SegNet 结构）**引入 RGB-D 像素级物体 affordance 分割的工作：通过端到端深度多模态特征学习在 UMD 数据集上将平均 $F_\beta^w$ 提升至 **0.770**（较手工特征 HMP 高 21.3 个百分点），并以 **90 ms** 实时推理驱动全尺寸人形机器人 **WALK-MAN** 完成 9 类物体实机抓取。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CNN | Convolutional Neural Network | 卷积神经网络，本文采用 13 层编码器 + 13 层池化索引上采样解码器 |
| HHA | Horizontal disparity, Height above ground, Angle with gravity | 将深度图编码为水平视差、离地高度与重力法线夹角的三通道表示 |
| HMP / SRF | Hierarchical Matching Pursuit / Structured Random Forests | UMD 原基线使用的手工层次匹配追踪特征与结构化随机森林 |
| CRF | Conditional Random Field | 条件随机场，DeepLab 末层用于空间平滑，本文省略以保实时性 |

## 为什么重要

- 在此之前，物体部件 affordance 检测主流依赖手工几何特征（如 Myers et al. 的 HMP 与 SRF），不仅特征工程繁琐，且在 `grasp`、`w-grasp`、`cut` 等细小功能区精度骤降，从未在真实大型机器人上闭环验证。
- 本文首次证明无全连接层的对称编码–解码 CNN 能同时兼顾全类别稳定分割精度与机器人控制所需的实时推理速度（90 ms/帧），并给出了从像素级 affordance 掩码聚类到矩形抓取框的简洁实机落地范式。

## 核心信息

| 字段 | 内容 |
|------|------|
| 出处 | IROS 2016 |
| 作者与机构 | Anh Nguyen, Dimitrios Kanoulas, Darwin G. Caldwell, Nikos G. Tsagarakis（IIT） |
| 清单分组 1 | Perception · Scope: Object；Perception Subcategory: Visual |
| 论文与项目页 | <https://www.csc.liv.ac.uk/~anguyen/assets/pdfs/2016_IROS_Affordance.pdf> · <https://sites.google.com/site/affordancecnn/> |

## 核心机制

1. **多模态数据表征与类别平衡训练**
   - 对比三种输入模态：`CNN-RGB`、`CNN-RGBD`（RGB 拼接原始深度）与 `CNN-RGBHHA`（RGB 拼接三通道 HHA 编码）。
   - 训练时对 480×640 图像中心裁剪至 240×320，使用按类别像素频率反比加权的交叉熵损失应对类别不平衡（SGD，batch=10，lr=0.001 每 50,000 次迭代衰减 10 倍，Caffe 训练约 3 天）；测试时在全图上滑动窗口取最大概率类。
2. **无全连接层对称卷积编码–解码网络**
   - 参考 **SegNet**，编码器采用 **VGG16 的 13 层卷积**（Conv + BatchNorm + ReLU + 2×2 非重叠最大池化），**完全丢弃全连接层**（避免 134M 参数量级造成的训练困难与推理缓慢）；
   - 解码器对称设置 13 层卷积，复用编码阶段记录的**池化索引（memorized pooled indices）**做非线性上采样，末层通过多类 Softmax 输出 $k$ 通道像素级 affordance 概率图。
3. **Mean Shift 聚类与人形机器人矩形抓取框生成**
   - 在身高 1.85 m、7-DoF 手臂、5 指单电机欠驱动手的全尺寸人形机器人 **WALK-MAN**（MultiSense-SL 1024×1024 RGB-D 相机 + YARP + OpenSoT 全身控制）上部署：
   - 提取抓取相关 affordance（`grasp`、`w-grasp`、`contain`、`support`）高置信像素，使用 **Mean Shift** 聚类去除离散噪点（仅保留点数 > 100 的有效簇），对簇点求凸包并拟合**最小外接矩形框**以解算末端执行器抓握中心、方向与尺度。

### 方法流程图

```mermaid
flowchart LR
    A["MultiSense-SL / Kinect RGB-D 图像"] --> B["多模态输入构造 (RGB / RGB-D / RGB-HHA)"]
    B --> C["13 层 VGG16 编码器 (Conv+BN+ReLU+MaxPool 记录池化索引)"]
    C --> D["13 层对称解码器 (池化索引上采样 + Softmax 概率图)"]
    D --> E["滑动窗口像素级 7 类 Affordance 掩码"]
    E --> F["Mean Shift 聚类 (过滤 <=100 点噪声簇)"]
    F --> G["凸包提取与最小外接矩形拟合"]
    G --> H["WALK-MAN (OpenSoT) 7-DoF 全身运动规划与欠驱动手抓取"]
```

## 评测与指标

### 1. UMD 数据集对比（~30,000 对 RGB-D，17 类 105 件工具，7 类标签，$F_\beta^w$ ↑，Table I）

| 方法 | grasp | w-grasp | cut | contain | support | scoop | pound | **Average** |
|------|---|---|---|---|---|---|---|---|
| SRF（手工特征） | 0.314 | 0.285 | 0.412 | 0.635 | 0.429 | 0.481 | 0.666 | 0.460 |
| HMP（手工特征） | 0.367 | 0.373 | 0.415 | 0.810 | 0.643 | 0.524 | 0.767 | 0.557 |
| DeepLab（+全连接 CRF） | 0.620 | 0.730 | 0.600 | **0.900** | 0.600 | **0.800** | **0.880** | 0.733 |
| CNN-RGBHHA | 0.673 | 0.652 | 0.685 | 0.716 | 0.663 | 0.635 | 0.701 | 0.675 |
| CNN-RGB | **0.719** | **0.769** | **0.737** | 0.817 | 0.780 | 0.744 | 0.794 | 0.766 |
| **CNN-RGBD（本文最优）** | 0.714 | 0.767 | 0.723 | 0.819 | **0.803** | 0.757 | 0.806 | **0.770** |

- **实时性**：在 NVIDIA Titan X GPU 上单图前向推理约 **90 ms**。

### 2. WALK-MAN 人形机器人实机抓取实验（9 类物体各 20 次试验，Table II）

- **平均抓取成功率**：**92.2%**（举起并空中保持 15 秒算成功；总周期 ~45 s，其中视觉检测与矩形拟合仅 ~1 s）。
- 分项成功率与所用 affordance：Bottle (`w-grasp`) **100%**、Hammer (`grasp`) **100%**、Scissors (`grasp`) **100%**、Turner (`support`) **100%**、Comb (`grasp`) **95%**、Ruler (`grasp`) **90%**、Saw (`grasp`) **90%**、Headphone (`grasp`) **80%**、Cup (`contain`) **75%**。

## 与其他工作对比

- **对比手工特征基线（HMP、SRF）**：`CNN-RGBD` 平均 $F_\beta^w$ 绝对领先 HMP **+21.3%**（0.770 vs 0.557）、领先 SRF **+31.0%**（0.770 vs 0.460），消除了手工特征在 `grasp/w-grasp/cut` 上的剧烈掉点。
- **对比 DeepLab**：DeepLab 虽在 `contain/scoop/pound` 三个大块区域最高，但在其他类别波动较大，平均（0.733）仍低于无 CRF 后处理的 `CNN-RGBD`（0.770）。同分组横向对照见 [技术地图](../overview/affordance-awesome-technology-map.md)。

## 常见误区或局限

1. **HHA 深度编码在桌面数据集上反而负优化**：`CNN-RGBHHA`（0.675）大幅低于 `CNN-RGBD`（0.770）与 `CNN-RGB`（0.766），原因是 UMD 物体均放置于平坦桌面，单目深度估计重力方向频繁失准从而破坏 HHA 通道。
2. **矩形框抓取受限于物理手掌尺寸**：若检测出的 affordance 面积超出欠驱动机械手张合范围或闭合几何不匹配，则抓取失败（Cup 仅 75% 即受限于手掌尺寸与 `contain` 区域宽度）。
3. **缺乏部件间语义关系建模且真机需补采数据**：真机部署前需将 UMD 与自建额外数据联合重训，且网络独立分类各像素类别，尚未建模同物体不同功能部件间的组合约束。项目页仅提供实验视频展示，未附完整开源训练代码仓库。

## 结论

**本文奠定了基于端到端卷积编码–解码器的 RGB-D 像素级 affordance 实时检测范式，并完成了早期全尺寸人形机器人的闭环抓取验证。**

- 在桌面操作场景中，直接拼接原始深度（`CNN-RGBD`）往往比依赖重力估计的 `HHA` 编码更稳健，且类别加权损失能显著均衡小尺寸手柄类（`grasp`/`cut`）的召回率。
- 去掉全连接层并用池化索引上采样，既保证 90 ms 实时推理，又能配合 Mean Shift 滤波 + 最小外接矩形快速输出机器人可执行的平面抓握参数。
- 向现代 6-DoF 灵巧操作迁移时，需克服最小矩形框对抓取尺寸和桌面垂直假设的限制（后续如 FAKP-Net、AffordanceNet 等工作逐步转向区域建议或 3D 关键点建模）。

## 关联页面

- [Affordance 学习综述](../entities/paper-affordance-learning-embodied-ai-survey.md)
- [Awesome-Affordance-Learning 技术地图](../overview/affordance-awesome-technology-map.md)
- [manipulation.md](../tasks/manipulation.md)、[vla.md](../methods/vla.md)

## 参考来源

- [`sources/papers/affordance_awesome_detecting-object-affordances-with-convolutional-neural-n.md`](../../sources/papers/affordance_awesome_detecting-object-affordances-with-convolutional-neural-n.md) — 本条目策展摘录
- [`sources/papers/affordance_awesome_catalog.md`](../../sources/papers/affordance_awesome_catalog.md) — 列表总表
- [`sources/repos/awesome-affordance-learning.md`](../../sources/repos/awesome-affordance-learning.md)
- 论文：<https://www.csc.liv.ac.uk/~anguyen/assets/pdfs/2016_IROS_Affordance.pdf>

## 推荐继续阅读

- [Awesome-Affordance-Learning 仓库](https://github.com/hq-King/Awesome-Affordance-Learning)
- [原文](https://www.csc.liv.ac.uk/~anguyen/assets/pdfs/2016_IROS_Affordance.pdf)
