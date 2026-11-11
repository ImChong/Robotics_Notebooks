---
type: entity
project_id: sam-vtp
tags:
  - paper
  - navigation
  - perception
  - traversability
  - segmentation
status: complete
updated: 2026-11-11
arxiv: "2611.12099"
code: https://github.com/lucashoefs/sam_vtp
original:
  - ../../sources/papers/sam-vtp_arxiv_2611_12099.md
related:
  - ../concepts/terrain-adaptation.md
  - ../concepts/image-segmentation-taxonomy.md
  - ../concepts/sim2real.md
  - ../concepts/visual-foundation-model-trends.md
  - ../concepts/hierarchical-quadruped-navigation-stack.md
  - ../concepts/embodied-perception-six-spatial-representations.md
sources:
  - ../../sources/papers/sam-vtp_arxiv_2611_12099.md
summary: "SAM-VTP（arXiv:2611.12099）：微调 SAM2 视频记忆通路做无提示的连续可通行性分割，并融合深度测得的坡度 / 粗糙度 / 高度；TartanGround 合成评测加 JetAcker 实机闭环，依赖深度传感器。"
---

# SAM-VTP（arXiv:2611.12099）

**SAM-VTP**（*SAM-VTP: Video Memory and Measured Geometry for Continuous Traversability Segmentation*，Curtin University，[arXiv:2611.12099](https://arxiv.org/abs/2611.12099)，代码 [lucashoefs/sam_vtp](https://github.com/lucashoefs/sam_vtp)）来自公众号文章《机器人动作块太长，什么时候该重新规划？》（2026-11-11 的 53 篇论文速览）。

## 一句话定义

**在 SAM2 的视频记忆通路上做微调，逐帧输出 [0,1] 的像素级连续可通行性，并把深度测得的坡度、粗糙度、高度融合进图像特征；无需提示、位姿或里程计。**（论文摘要与 Sec. III-A）

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SAM2 | Segment Anything Model 2 | 带视频记忆的分割基础模型，本文的骨干 |
| VTP | Video Traversability Prediction | 本文方法名 |
| SAM-TP | SAM traversability predictor（来自 GeNIE） | 逐帧 SAM2 基线，不用记忆通路 |
| fJacc | fuzzy (Ruzicka) Jaccard | 连续分数上的 Jaccard，论文主指标（Eq. 1） |
| ECE | Expected Calibration Error | 期望校准误差，论文改成软目标版本 |
| RGB-D | RGB + Depth | 彩色加深度输入 |
| FPS | Frames Per Second | 每秒帧数 |

## 为什么重要

- 逐帧可通行性在视角变化、遮挡下会闪烁，进而让代价地图不稳（Sec. I，Fig. 1）；本文把记忆放进图像空间的稠密预测器内部。
- 连续目标把「能不能过」的阈值推迟到部署时，由平台自己选工作点（Sec. I、III-C）。
- 给出了 Orin Nano 上 21.4 FPS 的实机闭环结果（Table I、III），说明这条路线可上小型边缘设备。

## 核心原理与流程总览

```mermaid
flowchart LR
  A[RGB 帧序列] --> B[Hiera-T 加 FPN 图像编码]
  D[深度图与内参] --> G[几何编码 斜率 粗糙度 高度 距离 有效性]
  G --> B
  B --> M[记忆注意力 7 帧记忆库]
  M --> C[掩码解码器 学习到的固定提示 单 token]
  C --> R[全分辨率残差头]
  R --> O[逐像素可通行性 0 到 1]
  C --> E[记忆编码器]
  E --> M
  O --> P[规划器 弧线规划加纯追踪]
```

- **记忆（Sec. III-A2）：** 记忆库保存最近 N=7 帧的特征与预测；记忆注意力、记忆编码器、解码器微调，图像编码器以 0.1× 学习率解冻。关闭 SAM2 的三 mask 输出，只监督单 token（原因：质量头无可通行性监督，排序任意会引起闪烁）。
- **提示（Sec. III-A1）：** 用 4 个学习的稀疏 token 与稠密嵌入替代点击提示，任务里「感兴趣区域永远是可驾驶地面」。
- **测得几何（Sec. III-A3）：** 由深度算出坡度、局部高度粗糙度、高度、log 距离、有效性，编码后加到最粗层特征，同时送入记忆注意力与解码器；训练时随机丢弃该分支，但推理撤掉深度损失 0.061 fJacc。
- **监督目标（Sec. III-C）：** y = 语义先验（手工查找表，七档 0-1）× 坡度衰减（12°→35°，下限 0.3）× 粗糙度衰减（0.03→0.20 m，下限 0.4）；高出局部地形 0.55 m 或超出 (-1.0, 2.5) m 的像素记 0。论文明说这是语义 + 几何「可驾驶类型与形状」的估计，不是物理交互测量。
- **损失（Sec. III-A4）：** Ruzicka 损失 + BCE + Focal(γ=3) + 时序一致项，权重 (1.2, 0.8, 0.7, 0.5)。
- **训练设置（Sec. III-B, III-D）：** TartanGround 136 条轨迹 / 24 个环境，94 / 20 / 22 划分；T=24、stride 2；AdamW 3e-5；RTX 5090。

### 关键结果

| 对比 | 数字 | 出处 |
|------|------|------|
| fJacc，SAM-TP → SAM-VTP | 0.908 → 0.924（种子差 <0.001） | Table I |
| 二值 IoU | 0.854 → 0.886 | Table I |
| Flicker（帧间 fJacc 变化） | 0.0175 → 0.0120，文称 −31% | Table I, Sec. IV |
| 吞吐 SAM-VTP | 296 / 30.1 / 21.4 FPS（5090 / AGX Orin / Orin Nano） | Table I |
| 消融（较短预算）基线 / +几何 / 仅记忆 / 两者 | 0.875 / 0.900 / 0.894 / 0.914 | Table II |
| 消融：撤深度 / 二值目标 | 0.853 / 0.733 | Table II |
| 无可行路径占比，铺装 | 32.4% → 9.9%（15 次运行） | Table III |
| 无可行路径占比，粗糙地形 | 20.6% → 3.0%（21 次运行，10 对配对） | Table III |

## 工程实践

- **输入约定（仓库 README，静态读）：** 深度须是沿光轴的平面 z-depth（米）；内参须是你自己相机在标定分辨率下的，默认为 TartanGround 相机（640×640, 90°）；二者写错不会报错而是给出看似合理的错误地图。
- **部署：** 帧需按采集顺序输入，序列间调用 `reset_memory()`；CUDA Graph 回放是 CUDA 上的默认路径；论文称 SAM2 原生不支持 CUDA Graph（Sec. III-D）。README 称整体 41.2 M 参数，几何编码器约 128 k。
- **换平台：** 目标绑定一个间隙包络，平台差异大需重新生成标签（Sec. V）；仓库提供 `sam-vtp-labels` 标签管线，文档中假设约 0.15 m 轮半径、0.20 m 离地间隙、30° 稳定坡（docs/labels.md）。
- **实机栈：** JetAcker + RealSense D455f + Orin Nano 8 GB（JetPack 6.2），ROS 2 部署栈在另一仓库 sam_vtp_jetacker（未核实）。
- 仓库 MIT 许可；权重在 Hugging Face（未核实）。本页未运行任何代码。

## 与文章说法的核对

文章原话：「结合视频记忆与深度测得的坡度、粗糙度等信息，持续预测可通行区域。合成数据评测之外有实地导航；仍依赖深度质量、机器人通行尺度，不能仅凭语义判断能否通过。」

| 文章说法 | 核对 | 结论 |
|----------|------|------|
| 视频记忆 + 深度测得坡度 / 粗糙度，持续预测 | Sec. III-A：7 帧记忆库，斜率、粗糙度、高度等几何由深度测量 | 与论文一致 |
| 合成数据评测之外有实地导航 | 精度评测仅 TartanGround 合成，对照自家管线生成的目标（Sec. V）；实机为 Table III 两个场地 15 + 21 次运行 | 一致，但应补充：实地部分只测规划行为，不测 mask 质量；且不是同一指标 |
| 依赖深度质量 | 论文显示依赖「有深度」：撤去推理深度 −0.061 fJacc（Table II，Sec. V）；未做深度噪声 / 失真的实验 | 「依赖深度」成立；「深度质量」影响是推测，论文未测 |
| 依赖机器人通行尺度 | Sec. V：目标绑定间隙包络，平台差异大需重新生成目标 | 与论文一致 |
| 不能仅凭语义判断能否通过 | Sec. III-C：目标 = 语义先验 × 几何调制，称无法仅由语义或几何推出（Fig. 3 图注）；Table II 撤几何或撤深度均下降 | 论文有消融支持，但其「真值」本身由同样几何规则生成，存在循环风险（论文 Sec. III-A3 列出三项预防，不是消除） |
| 文章未提的数字 | 摘要：无路径次数 3.3× / 6.9× 更少；Orin Nano 21.4 FPS | 文章未提，不构成矛盾 |

附加提醒（作者自述与本页判断分开）：
- 作者自述：实机对比 SAM-TP 用 RGB、SAM-VTP 用 RGB-D，故不能分离记忆与深度（Table III 标题）；分离在合成消融中完成（Table II，较短训练预算）。
- 作者自述：运行由操作员终止，SAM-TP 通常在安全门触发后被取回（Sec. IV）；粗糙地形的进度率与距离并没有在所有配对上分离（8/10、7/10）。

## 局限

**论文自己承认（Sec. V、IV）：**
- 部署需要深度传感器；撤深度损失 0.061 fJacc。
- 目标承诺一个间隙包络，换平台需重新生成而不只是重设阈值。
- 序列初始化有短暂预热（冷启动）。
- 全部分割精度来自合成数据、对照自家管线的目标；闭环只测规划行为。
- 管线需要逐像素语义图，仿真给，真实影像没有，仿真外使用需要语义分割器在环。
- 语义查找表数值是手工的序数值，未校准（Sec. III-C）。

**本页观察到的未验证之处（判断，非论文结论）：**
- 实机样本小：15 次与 21 次运行，粗糙地形仅 10 对配对；Table I 中 WVN / WayFASTER 为单次运行。
- 闭环中 RGB 与 RGB-D 的对比混了两个因素（见上）；论文做了回放以补偿，但回放也只有每场地三次试验。
- 没有深度噪声 / 缺失 / 反光表面的鲁棒性实验（推测其会影响效果，论文未测）。
- 语义先验在实机推理时并不提供，模型要从图像推断；但训练目标的语义来自仿真真值，真实场景下的语义偏差未量化。
- 吞吐表的 SAM-TP 用的是论文自己优化的实现，GeNIE 原实现慢 7×，速度对比取决于实现口径。
- 仓库 Citation 为 TBD、权重与部署栈未核实。

## 关联页面

- [地形适应（Terrain Adaptation）](../concepts/terrain-adaptation.md)
- [图像分割分类（Image Segmentation Taxonomy）](../concepts/image-segmentation-taxonomy.md)
- [Sim2Real](../concepts/sim2real.md)
- [视觉基础模型趋势](../concepts/visual-foundation-model-trends.md)
- [分层四足导航栈](../concepts/hierarchical-quadruped-navigation-stack.md)
- [具身感知的六种空间表示](../concepts/embodied-perception-six-spatial-representations.md)

## 参考来源

- [sam-vtp_arxiv_2611_12099.md](../../sources/papers/sam-vtp_arxiv_2611_12099.md)
- [arXiv:2611.12099](https://arxiv.org/abs/2611.12099)
- 公众号文章《机器人动作块太长，什么时候该重新规划？》（2026-11-11）
- [GitHub：lucashoefs/sam_vtp](https://github.com/lucashoefs/sam_vtp)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2611.12099v1)
- [SAM 2 官方仓库](https://github.com/facebookresearch/sam2)
- [地形适应](../concepts/terrain-adaptation.md)
