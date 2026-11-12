---
type: entity
project_id: pls-gs
tags:
  - paper
  - 3dgs
  - gaussian-splatting
  - lidar
  - scaffold-gs
  - mapping
  - qut
status: complete
updated: 2026-11-11
arxiv: "2611.10699"
code: https://github.com/k-harikumar/PLS-GS
original:
  - ../../sources/papers/pls-gs_arxiv_2611_10699.md
related:
  - ./paper-compactsplat.md
  - ./paper-reduced-3dgs.md
  - ./paper-gaussian-lic2.md
  - ./paper-rcl-2308-04079-3d-gaussian-splatting-for-real-time-radiance-fie.md
  - ./paper-pai-2410-12262-3dgaussiansplattinginrobotics.md
  - ../concepts/lidar-sensing.md
  - ../concepts/lidar-slam-survey-2026.md
sources:
  - ../../sources/papers/pls-gs_arxiv_2611_10699.md
summary: "PLS-GS（QUT，WACV 2027 预印本；arXiv:2611.10699）：在相机加 LiDAR 的户外场景里，用位置加颜色 6D 层级高斯混合当 Scaffold-GS 的锚点支架，在各分量曲面系里解码神经高斯并加马氏距离支撑损失；GauU-Scene V2 五场景上精度更高、模型更小、渲染更快，但 PSNR 低于基线，代码尚未发布。"
---

# PLS-GS（arXiv:2611.10699）

**PLS-GS**（*PLS-GS: Probabilistic Lidar Scaffolds for Compact, Surface-Aligned Gaussian Splatting*，[arXiv:2611.10699](https://arxiv.org/abs/2611.10699)，Harikumar、Fontan、Peynot、Hernández、Sünderhauf，QUT Centre for Robotics 与 Emesent）是一篇 3D 高斯泼溅（3DGS）方法论文，首页标注为 WACV 2027 预印本。

## 一句话定义

**把 LiDAR 加相机点云拟合成一棵位置加颜色的 6D 层级高斯混合树，用叶分量做 Scaffold-GS 的锚点并固定其位置与曲面朝向，让渲染用的神经高斯沿曲面展开、被马氏距离损失限制在曲面附近，从而得到更紧凑、更贴近测量表面的 3DGS。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| 3DGS | 3D Gaussian Splatting | 用各向异性 3D 高斯基元做可微光栅化渲染的场景表示 |
| LiDAR | Light Detection and Ranging | 激光雷达，提供测量级几何 |
| GMM | Gaussian Mixture Model | 高斯混合模型，用 EM 拟合点集 |
| HGMM | Hierarchical Gaussian Mixture Model | 层级 GMM，逐层细分子分量，分辨率随局部点密度自适应 |
| EM | Expectation-Maximisation | 期望最大化，GMM 的拟合算法 |
| SfM | Structure-from-Motion | 运动恢复结构，论文用 COLMAP 的稀疏点补 LiDAR 覆盖不到的区域 |
| PSNR / SSIM / LPIPS | 峰值信噪比 / 结构相似性 / 学习感知距离 | 渲染质量指标 |
| Ch. | Chamfer distance | 双向最近点距离，论文以米为单位 |
| FPS | Frames Per Second | 渲染帧率 |

## 为什么重要

- 机器人与测绘平台常同时有相机和高密度 LiDAR，纯图像 3DGS 只间接约束几何，基元冗余且会偏离真实表面（作者自述，Sec. 1）。
- 现有 Scaffold-GS 一类方法的锚点是离散体素、由图像点播种；LiDAR 监督类方法的基元仍自由漂移、逐基元存储（Sec. 1–2）。PLS-GS 的主张是连续、自适应、由测量几何生成的支架。
- 对想了解「几何先验如何换紧凑度」的读者，这篇给出了可对照的数字与消融（见下）。属于方法论文，不是机器人闭环应用论文，论文内没有机器人任务实验。

## 核心原理与流程总览

```mermaid
flowchart LR
  A[彩色点云 LiDAR 加 SfM] --> B[位置加颜色 6D 标准化]
  B --> C[GPU 流式 EM 拟合 HGMM 叶层级 7]
  C --> D[LiDAR 混合与 SfM 混合马氏距离门去冗余]
  D --> E[每个叶分量一个锚点 位置与曲面系固定]
  E --> F[曲面系解码 k 个神经高斯 mu = x + U 乘 o 乘 s]
  F --> G[光度损失加马氏距离支撑损失]
  G --> H[高残差区域拟合局部 HGMM 生长新锚点]
  H --> F
```

1. **概率支架（Sec. 4.1，5.3，S-I.2）：** 点云 `[位置, 颜色]` 六维标准化后拟合 HGMM，每节点分 K=8 个子分量，叶层级为 7，丢弃支撑点不足（LiDAR 最少 32 点，SfM 最少 1 点）和特征值长宽比超过 20 的分量。LiDAR 与 SfM 各拟合一棵，SfM 分量若落入某 LiDAR 分量的 χ²（0.99 置信度、6 自由度）马氏门内就删除，LiDAR 骨干从不删除。联合颜色的理由：能把空间相邻但外观不同的表面（路标与沥青）分开。
2. **曲面系解码（Sec. 4.2）：** 协方差特征分解 Σ = U diag(λ) Uᵀ，前两列张成切平面，第三列对齐法向；神经高斯中心为 `x + U (o ⊙ s)`，不透明度、协方差、颜色沿用 Scaffold-GS 的共享 MLP，颜色均值作 logit 空间的加性基底。锚点位置与曲面系训练中固定。
3. **支撑损失（Sec. 4.3）：** 对各神经高斯在局部系下按特征值归一的位移平方求平均，沿切向便宜、沿法向昂贵；总损失 `L = L_photo + λs·L_support`，采用 λs = 1e-3。
4. **概率生长（Sec. 4.4，S-I.3）：** 累积视空间位置梯度，超阈值的神经高斯代表持续光度残差，对其拟合局部 6D HGMM，叶分量成为新锚点并与已有支架去重；生长锚点可被剪枝，测量几何锚点保留。补充材料给出：第 500 次迭代起累积梯度，第 1500 至 15000 次迭代每 100 次生长一次。

## 工程实践

- **硬性输入假设：** 需要与 SfM（COLMAP）坐标系配准好的彩色 LiDAR 点云。作者称 GauU-Scene V2 是其所知唯一满足此条件的大尺度公开基准（Sec. 5.1）；LiDAR 覆盖不到处靠 SfM 稀疏点填补（Sec. 4.1）。
- **实验设置（Sec. 5.1，5.3）：** GauU-Scene V2 五场景（面积 0.8 至 1.56 km²，LiDAR 点 27M 至 94M，相机 424 至 1215 张），图像长边 800 像素，每 8 张留 1 张测试；基于公开 Scaffold-GS 实现，30K 迭代，单张 24 GB GPU，k=10，特征维 32，关闭逐图像外观嵌入（S-I.3）。
- **预处理成本（Table S1，均值）：** HGMM 拟合一次性约 26:04（分:秒）、峰值显存约 11.15 GB；含拟合的总时间均值 43:49，高于 Scaffold-GS 的 25:47（SfM）和 25:09（LiDAR），只有纯训练循环更短（17:45）。
- **几何评测协议（Sec. 5.4，S-I.4）：** 沿用 CityGaussianV2 的可见性裁剪加阈值，τ 为各场景真值降采样最近邻距离的 1.5 倍（0.314 至 0.338 m），直接取神经高斯均值而非网格抽取。作者明确说这样的分数与他人用网格采样报的数字不可比。
- **超参敏感性：** λs 太大伤渲染（Table 3：1e-1 时 College 的 PSNR 23.85 dB）；k 增大精度与模型体积同步增加（Table S3，k=5 到 20 精度 0.368 到 0.397、体积 164.5 到 258.6 MB）。
- **代码状态：** 论文摘要称代码可得，但仓库 README（读于 2026-11-11）写明发布「in preparation and coming soon」，当前无可运行代码，复现只能照论文与补充材料的参数自行实现。论文里的链接 `kharikumar/PLS-GS` 与实际仓库 `k-harikumar/PLS-GS` 不一致。

## 实验结果（作者报告，GauU-Scene V2 五场景均值）

| 指标 | Scaffold-GS（SfM） | Scaffold-GS（LiDAR） | PLS-GS | 出处 |
|------|------|------|------|------|
| 精度 P | 0.312 | 0.330 | 0.484 | Table 1 |
| 召回 R | 0.189 | 0.245 | 0.191 | Table 1 |
| F1 | 0.234 | 0.279 | 0.269 | Table 1 |
| Chamfer（m） | 0.899 | 0.820 | 0.703 | Table 1 |
| PSNR（dB） | 27.10 | 26.48 | 25.45 | Table 1 |
| 终锚点数 | 1368K | 1308K | 460K | Table 1 |
| 模型大小（MB） | 432.5 | 413.2 | 172.9 | Table 1 |
| FPS | 114.0 | 114.4 | 190.4 | Table 1 |

- 论文称 PLS-GS 每个场景精度最高、Chamfer 最低（Sec. 6.1）；召回与 F1 随容量增长，LiDAR 基线锚点是其 2.8 倍，因此在这两项上更高，作者因此把召回解释为「覆盖而非放置」（Sec. 5.4，6.1）。这是作者的解读，需读者自行判断。
- 匹配容量（Table 2，同一批锚点）：精度 0.314 到 0.484，Chamfer 1.167 到 0.703 m，PSNR 25.66 到 25.45 dB，FPS 272.0 到 190.4，即同容量下几何明显更好、画质略低、速度更慢（作者归因于曲面系渲染公式，Sec. 6.2）。
- Table 3（College）：光度生长使锚点增加 3.23 倍，概率生长增加 1.06 倍；去掉各向异性初始化后生长版精度由 0.363 降到 0.312。

## 与文章说法的核对

对照公众号文章《机器人动作块太长，什么时候该重新规划？》（2026-11-11 的 53 篇论文速览）第 42 条。

| 文章说法 | 论文核对 |
|----------|----------|
| 「以激光点云的概率支架约束高斯位置和方向」 | 基本相符，但要补一层：锚点位置与曲面朝向由 HGMM 固定，约束神经高斯的是 Eq. 10 的曲面系偏移加 Eq. 11 的马氏距离支撑损失（Sec. 4.2–4.3），并非直接约束每个高斯的朝向。支架输入是 LiDAR 加 SfM 的彩色点云，不是纯激光点云。 |
| 「用表面结构先验压缩场景表示」 | 相符。Table 1：模型 413.2 到 172.9 MB（42%），锚点 2.8 倍更少；College 在图 1 中为 35%。 |
| 「户外数据验证了紧凑性」 | 相符但范围窄：只在 GauU-Scene V2 的 5 个航测大场景上验证，图像降采样到长边 800 像素，只有静态场景，无机器人在线任务（Sec. 5.1）。 |
| 「依赖配准点云」 | 相符。方法需要与 COLMAP 坐标系配准的彩色 LiDAR（Sec. 5.1，S-I.2）；作者没有在局限节写这一条，是作者在数据集说明里交代的前提，「依赖」是对其的合理解读。 |
| 「渲染质量仍与更自由的表示存在取舍」 | 相符。Table 1 PSNR 比两个基线低 1.03 至 1.65 dB，局限节自认「rendering quality remains below Scaffold-GS with unconstrained photometric growth」（Sec. 7）；匹配容量下差距收窄到 0.21 dB。 |
| 文章给出的代码链接 `k-harikumar/PLS-GS` | 仓库存在，但 README 写代码「coming soon」，截至 2026-11-11 尚未开源代码；文章未提这一点。 |

文章未写但值得注意的：召回与 F1 均值上两个基线更高（Table 1），论文用容量差解释，匹配容量后才反过来（Table 2）。

## 局限

**论文自述（Sec. 7）：**
- 渲染质量仍低于带无约束光度生长的 Scaffold-GS；计划利用层级结构更自适应地细化、研究压缩，并与渲染深度损失比较，后续扩展到 4D 高斯。

**我读到的未验证或需谨慎之处（读者判断，非作者结论）：**
- 只有一个数据集（GauU-Scene V2），五个场景、图像降到 800 像素；没有真机、在线或动态场景实验，也没有与 LI-GS、ARSGaussian 的直接对比，作者称两者无公开实现（Sec. 5.2）。基线只有 Scaffold-GS 两种初始化。
- 几何评测直接取神经高斯均值，LiDAR 作真值同时又是方法的输入，几何指标与 LiDAR 一致只说明贴近该传感器测量，不等于贴近真实表面；作者也说明分数不可与网格采样的结果比（Sec. 5.4）。
- 召回与 F1 的叙事依赖作者对「容量」的解读；主表里这两项 PLS-GS 并未占优。
- 速度：匹配容量下 PLS-GS 更慢（272 对 190 FPS）；总时间含 HGMM 预处理反而高于基线（Table S1）。
- 本页未取得代码（未发布），所有数字都是论文自报，未复现。项目页未访问，未核实。
- 论文性质：方法论文，非机器人任务论文，与机器人落地的距离需要自行评估。

## 与其他工作对比

| 工作 | 关注点 | 与本文的区别 |
|------|--------|--------------|
| [Reduced 3DGS](./paper-reduced-3dgs.md) | 训练后剪枝、球谐自适应与量化压缩基元 | 事后压缩；PLS-GS 从测量几何生成支架，在训练前就限制锚点数量与位置 |
| [CompactSplat](./paper-compactsplat.md) | 前馈 3DGS 按纹理分区与层级采样自适应分配高斯 | 纯图像前馈、关注推理期稀疏度；PLS-GS 是逐场景优化，依赖配准彩色 LiDAR |
| [Gaussian-LIC2](./paper-gaussian-lic2.md) | 实时 LiDAR-惯性-相机高斯 SLAM，LiDAR 深度监督建图 | 在线 SLAM 系统；PLS-GS 是离线重建，LiDAR 以 HGMM 支架而非深度损失的形式进入，论文未与其对比 |

## 结论

**PLS-GS 把配准彩色点云拟合成 6D 层级高斯混合并用作 Scaffold-GS 的固定锚点支架，以略低的画质换来更贴近测量表面、体积约 42% 的更紧凑 3DGS。**

1. GauU-Scene V2 五场景均值：精度 0.484、Chamfer 0.703 m，均优于两个 Scaffold-GS 基线；模型 172.9 MB、190.4 FPS（Table 1）。
2. PSNR 低 1.03 至 1.65 dB，召回 / F1 低于 LiDAR 基线；匹配容量后几何优势保留，画质差距收窄到 0.21 dB（Table 2）。
3. 概率生长只增加 1.06 倍锚点，远少于光度生长的 3.23 倍（Table 3）。
4. 局限：单数据集、静态离线场景、LiDAR 同时作输入与几何真值、含预处理总时间高于基线，代码尚未发布（Sec. 5、7，Table S1）。

## 关联页面

- [CompactSplat：前馈紧凑 3D 高斯](./paper-compactsplat.md)
- [Reduced 3DGS：压缩 3DGS 基元数](./paper-reduced-3dgs.md)
- [Gaussian-LIC2：LiDAR-惯性-相机高斯映射](./paper-gaussian-lic2.md)
- [3D Gaussian Splatting 原始论文页](./paper-rcl-2308-04079-3d-gaussian-splatting-for-real-time-radiance-fie.md)
- [3D 高斯泼溅在机器人中的应用综述](./paper-pai-2410-12262-3dgaussiansplattinginrobotics.md)
- [LiDAR 传感](../concepts/lidar-sensing.md)
- [LiDAR SLAM 综述](../concepts/lidar-slam-survey-2026.md)

## 参考来源

- [pls-gs_arxiv_2611_10699.md](../../sources/papers/pls-gs_arxiv_2611_10699.md)
- [arXiv:2611.10699](https://arxiv.org/abs/2611.10699)
- 代码仓库：<https://github.com/k-harikumar/PLS-GS>（README 读于 2026-11-11）

## 继续阅读

- [arXiv PDF（v1）](https://arxiv.org/pdf/2611.10699v1)
- Scaffold-GS（CVPR 2024）、GauU-Scene V2（arXiv:2404.04880）、CityGaussianV2（ICLR 2025）：本文的基线、数据集与评测协议来源，均见论文参考文献
- [CompactSplat](./paper-compactsplat.md)
