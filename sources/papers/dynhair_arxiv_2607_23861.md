# dynhair_arxiv_2607_23861

> 来源归档（ingest）

- **标题：** Head Avatars with Dynamic Explicit Hair（DynHair）
- **类型：** paper
- **来源：** arXiv:2607.23861（2026-07-26 预印本）；**ECCV 2026**
- **作者：** Vanessa Sklyarova, Haonan Chen, Berna Kabadayi, Tobias Kirschstein, Zicong Fan, Xi Wang, Gerard Pons-Moll, Matthias Nießner, Marc Pollefeys, Michael J. Black, Justus Thies
- **机构：** 苏黎世联邦理工（ETH Zürich）、马克斯·普朗克研究所（MPI）、慕尼黑工业大学（TUM）、达姆施塔特工业大学（TU Darmstadt）、微软（Microsoft）等
- **入库日期：** 2026-09-16
- **最后更新：** 2026-09-16
- **项目页：** <https://dynhair.is.tue.mpg.de/>
- **代码：** <https://github.com/Vanessik/DynHair>（MIT；截至入库日为占位仓库）
- **一句话说明：** 多视角视频学习 **显式发丝级** 动态人头化身：发丝对齐 3DGS + LSTM–FiLM 头发形变器（条件于头部角速度/加速度/相对重力），与 GHA 式上半身 3DGS 联合可微渲染；自重演与跨主体驱动；GitHub 已建仓但尚无训练脚本。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / §I）

- **链接：** <https://arxiv.org/abs/2607.23861>
- **痛点：** 现有 Gaussian / NeRF 人头化身多把头发当作头部隐式纹理，**缺少物理可信的动态**（惯性、重力、甩动）；物理仿真需手工调参且难微分；从观测学习动力学的 capture-based 方法多用非结构化 Gaussian 或体积，strand 细节与跨主体泛化不足。
- **DynHair 主张：** **发丝多段线 + 发丝对齐 3D Gaussian** 表示外观；**LSTM** 编码 BFM 头部运动历史的角速度 \(\omega\)、加速度 \(\alpha\)、相对重力 \(\mathbf{g}\)，经 **FiLM** 调制每点发丝特征后由 **MLP** 预测相对 canonical 发型的位移；与 **上半身非结构化 3DGS**（GHA 式表情/姿态条件）联合优化。
- **对 wiki 的映射：**
  - [DynHair 实体](../../wiki/entities/paper-dynhair.md)
  - [Teleoperation](../../wiki/tasks/teleoperation.md)（数字人 / telepresence 上游）
  - [SHELLS](../../wiki/entities/paper-shells-layered-surface-sampling.md)（多视角人头重建对照）

### 2) 表示与训练（§III）

- **Canonical 发型：** Im2Haircut [52] PCA 先验初始化；每缕 **L=40** 点、约 **11k** 缕；静态阶段 \(\mathcal{L}_{\text{static}}\)（Im2Haircut + PCA/长度/平滑/前后视一致）。
- **运动条件：** 滑动窗 **T=5** 帧；\(\mathbf{c}_\tau=[\omega_\tau,\alpha_\tau,\mathbf{g}_\tau]\in\mathbb{R}^9\) 经 positional encoding → LSTM → \(\mathbf{z}_t\) → FiLM → 每点位移 \(\Delta p_{ij}\cdot\rho_j\)（根部衰减 \(\rho_j\)）。
- **渲染：** 发丝线段中点放置 Gaussian，主轴沿发丝方向；与上半身 Gaussian 拼接后 **可微 splatting** 输出 RGB、分割、方向图。
- **损失：** \(\mathcal{L}_{\text{photo}}\)（RGB/SSIM/LPIPS）+ \(\mathcal{L}_{\text{hair}}\)（轮廓 recall 偏重、方向、穿透、**弹性拉伸**）+ 发色时空正则。
- **训练：** 1024²；单卡 A100 **320k** iter；HHAvatar 3 场景 + 自采 **15 相机 72 FPS 4K** 新数据（22 动作、10 项头发动力学）。
- **对 wiki 的映射：**
  - [Generative World Models](../../wiki/methods/generative-world-models.md)（3DGS 数字人支线）

### 3) 实验数字（§IV，表 1 节选）

**自重演（3 被试测试集平均）：**

| 方法 | PSNR↑ | FID↓ | hair IoU↑ | tIoU_hair↑ | tLPIPS_ex | hair vel.×10³↑ |
|------|-------|------|-----------|------------|-----------|----------------|
| GaussianAvatars | 20.17 | 45.73 | — | — | -0.0230 | 0.40 |
| GHA | 22.33 | 36.25 | — | — | -0.0127 | 2.17 |
| Maya 仿真* | 19.39 | 62.64 | 0.776 | 0.925 | 0.0183 | 2.38 |
| **DynHair** | 21.60 | **30.06** | **0.878** | **0.936** | **0.0045** | **2.41** |

- **读法：** PSNR/SSIM 略低于 GHA 是 strand 几何约束的代价；**FID / tLPIPS_ex / 头发速度** 更反映动力学目标。基线 tLPIPS_ex 为负 → 过度平滑僵硬。
- **跨主体重演：** 相对运动条件可把驱动序列迁移到另一发型化身。
- **应用：** 发丝修剪、改色仍保留动力学；单目在固定 tracker 下可学习形变（项目页 4/1 视角消融）。

### 4) 消融要点（表 2）

- 去掉 **加速度** 或 **相对重力** → 点头时发丝不能自然下垂；去掉 **FiLM** / 换 MLP 编码器 → 运动僵硬（tLPIPS_ex 变负）；去掉 **弹性损失** → VER 暴涨、物理指标崩溃。
- **对 wiki 的映射：**
  - [DynHair 实体](../../wiki/entities/paper-dynhair.md) §结论

### 5) 开源核查（步骤 2.5，2026-09-16）

| 项 | 状态 |
|----|------|
| 项目页 | <https://dynhair.is.tue.mpg.de/> — 方法图、视频、消融齐全；摘要写 data and code available |
| GitHub | <https://github.com/Vanessik/DynHair> — **MIT**；`master` 仅 README/LICENSE/.gitignore，**无可辨识训练/推理脚本** |
| 结论 | **部分开源 / 待发布完整代码** — 官方仓已建但为占位；复现需等待完整发布或联系作者 |

## 其他公开资料

- 项目页演示与消融：<https://dynhair.is.tue.mpg.de/>
