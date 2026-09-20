# panogs_slam_arxiv_2609_17387

> 来源归档（ingest）

- **标题：** PanoGS-SLAM: Panoramic 3D Gaussian Splatting SLAM
- **短名：** PanoGS-SLAM
- **类型：** paper / slam / 3dgs / panoramic / dense-mapping / monocular
- **来源：** arXiv abs / PDF / HTML
- **原始链接：**
  - <https://arxiv.org/abs/2609.17387>
  - <https://arxiv.org/pdf/2609.17387>
  - <https://arxiv.org/html/2609.17387v1>
- **作者：** Yongqi Mao, Hao Shi, Yufan Zhang, Zhonghua Yi, Xiangfei Guo, Kaiwei Wang†（† 通讯）
- **机构：** 浙江大学（ZJU）；国防科技大学（NUDT）；蚂蚁集团（Ant Group，Hao Shi 第二单位）
- **版本：** arXiv:2609.17387v1（Submitted 2026-09-15）
- **入库日期：** 2026-09-20
- **一句话说明：** 首个在 **球面域** 直接做可微渲染与位姿联合优化的 **全景 3DGS 稠密 SLAM**；提出 **球面一致光度损失 L_pano**（补偿 ERP 面积畸变）与 **深度引导高斯初始化 DGIS**；PALVIO / SynPano 上轨迹与渲染全面优于几何与 GS 基线，前端 **15 iter** 收敛（MonoGS ~100 iter），SynPano room3 **7 FPS**。

## 核心摘录

### 1) 问题与动机
- 现有 **3DGS-SLAM**（GS-SLAM、MonoGS、Photo-SLAM 等）多基于 **窄 FoV 针孔**，光度梯度集中在窄视锥 → 旋转可观性弱、平移–旋转耦合强、优化病态；快速运动与大视角变化下跟踪不稳。
- 增量建图时，新观测区高斯支撑不足也会拖垮前端（MonoGS 进入大 unseen 区域时典型退化）。
- **全景/360°** 可在球面上连续分布光度约束，理论上扩大收敛 basin、改善 conditioning；但 ERP 像素面积非均匀，直接 L1/L2 会放大极区噪声。
- 既有全景 3DGS（360-GS、ODGS、OmniGS）偏 **离线重建**，未解决 **在线位姿估计 + 全局一致 SLAM**。

### 2) 方法要点
1. **球面 ERP 相机模型：** 3D 高斯经球坐标 \((\phi,\theta)\) 映射到 ERP 像素；2D 协方差用 ERP Jacobian（参考 ODGS 系全景光栅），在球域 splat，**不** 拆成透视 crop。
2. **前端 Tracking：** 固定高斯图，只优化当前帧位姿；最小化 **Panoramic Loss** \(L_{pano}=\sum_u \cos\theta(u)\|I_{render}(u)-I_{gt}(u)\|\)（单位球面积一致加权）；位姿由上一帧初始化；关键帧策略沿用 MonoGS（共视 + 位移）。
3. **预处理 / 深度：** 首帧用预训练 **全景单目深度 BiFuse [31]** 得初始深度图。
4. **后端 Mapping：** 在关键帧窗口内联合优化关键帧位姿、随机采样的非关键帧位姿与高斯参数；同样用 \(L_{pano}\)。
5. **DGIS（Depth-Guided Gaussian Initialization）：**
   - **Map init：** RGB-D 下采样；**单位球等面积均匀采样** 生成高斯，缓解 ERP 极区冗余。
   - **Gaussian insertion：** 关键帧按 **opacity 图** 自适应密度插入；深度由当前帧渲染深度初始化，无效像素取最近邻 + 小扰动。
6. **GS 管理：** cloning / splitting / pruning（MonoGS 风格）；关键帧窗口外 prune unseen GS。

### 3) 实验（论文报告摘要）
| 基准 | 指标 | 主要对照 | PanoGS-SLAM 读法 |
|------|------|----------|------------------|
| **PALVIO [14]**（10 seq，真实 PAL） | ATE RMSE (m) ↓ | P2U-SLAM 次佳 ~0.073–0.110；MonoGS 多 seq **>1 m** 或失败 | **Ours** ID01–10：**0.055–0.102**（表 I 全 bold）；例 ID02 **0.055** vs MonoGS **1.490** |
| **SynPano [13]**（5 room，合成 ERP） | ATE RMSE (m) ↓ | P2U-SLAM room1 **0.020**；MonoGS room2 **1.190** | **Ours** room1–5：**0.003–0.018**（room3 **0.003**） |
| **渲染（针孔 120°×3 视图平均）** | PSNR / SSIM / LPIPS | MonoGS / Photo-SLAM | PALVIO：**23.48 / 0.88 / 0.30**；SynPano：**30.58 / 0.91 / 0.22**（表 III，均优于 GS 基线） |
| **消融 SynPano** | ATE (cm) | w/o \(L_{pano}\) **5.19**；w LGS **0.84**；w/o DGIS **1.27** | **Ours 0.78** cm（表 IV） |
| **FoV 控制 SynPano** | ATE (cm) @ FoV | MonoGS @120° room2 **118.98** | Ours：120°→360° 单调改善（room2：59.64→11.29→…→**0.45** @360°） |
| **收敛 / 实时** | iter / FPS | MonoGS ~**100 iter** 才稳定 | **15 iter** 收敛；SynPano room3 **7.04 FPS**（表 VI；MonoGS 同场景 **5.07→1.19 FPS** 随 FoV 变） |

- **基线：** ORB-SLAM3、VINS-Mono、P2U-SLAM、LF-VISLAM（几何）；Photo-SLAM、MonoGS（GS-SLAM）。SynPano 无 IMU，VINS/LF-VISLAM 不适用。
- **公平性：** 对针孔基线将全景 **投影为虚拟 pinhole** 或 **PAL 环带图**；PanoGS 原生 ERP 输入。

### 4) 开源核查（步骤 2.5）
- **项目页：** 截至 2026-09-20 **未发现** 独立 `*.github.io` 项目页（arXiv 为主入口）。
- **GitHub：** 检索 `PanoGS-SLAM` **0 仓库**；论文 Abstract 写 *「The source code will be made publicly available.»*
- **结论：** **宣称将开源 / 待发布** — 无官方 URL；wiki `## 源码运行时序图` 标不适用。

## 对 wiki 的映射

- 升格 [PanoGS-SLAM 论文实体](../../wiki/entities/paper-panogs-slam.md)
- 交叉 [PanoLOG / G²PS](../../wiki/entities/paper-panolog-ggps.md)（离线全景 3DGS）、[Gaussian-LIC2](../../wiki/entities/paper-gaussian-lic2.md)（LIC 3DGS-SLAM）、[UniSim-SLAM](../../wiki/entities/paper-unisim-slam.md)、[导航·SLAM 栈](../../wiki/overview/navigation-slam-autonomy-stack.md)

## 当前提炼状态

- [x] 摘要 + 方法 + PALVIO/SynPano 表 + 开源边界
- [x] wiki 实体页与交叉引用
- [ ] 代码放出后补 `sources/repos/` 与源码运行时序图
